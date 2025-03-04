import os
import streamlit as st
import tempfile
from gtts import gTTS
import speech_recognition as sr
import audio_recorder_streamlit as recorder

from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document

os.environ["GOOGLE_API_KEY"] = ""

model_name = "BAAI/bge-small-en-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

def custom_load_text(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [Document(page_content=text)]

persist_directory = "chromadb_index"
text_file = "geeta_txt.txt"

if not os.path.exists(persist_directory):
    st.write("ChromaDB index not found. Building index from", text_file, "...")
    with st.spinner("Building index, please wait..."):
        documents = custom_load_text(text_file)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        vector_store = Chroma.from_documents(docs, hf, persist_directory=persist_directory)
        vector_store.persist()
    st.write("Index built and persisted!")
else:
    with st.spinner("Loading existing index..."):
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=hf)
    st.write("Loaded existing ChromaDB index.")

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

def retrieve_context(query):
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])
    return context

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def format_conversation_history(history):
    history_text = ""
    for speaker, text in history:
        history_text += f"{speaker}: {text}\n"
    return history_text

def llm_query(history, context, question):
    history_text = format_conversation_history(history)
    prompt_template = f"""Final Prompt: You are Lord Krishna from the Bhagwat Geeta.
You are conversing with a friend and the conversation so far is given below.
Use the following pieces of wisdom to answer the user's question with spiritual advice,
gentle guidance, and a melodious tone, as if you are offering divine advice.

Conversation History:
{history_text}

Context from Bhagwat Geeta:
{context}

Question:
{question}

Answer (in the voice of Krishna):"""
    with st.spinner("Krishna is contemplating..."):
        response = llm.invoke(prompt_template)
    return response.content

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang="en")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        temp_file.close()
        with open(temp_file.name, "rb") as f:
            audio_bytes = f.read()
        os.unlink(temp_file.name)
        return audio_bytes
    except Exception as e:
        st.error(f"Error in text-to-speech: {e}")
        return None

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

st.title("Ask Lord Krishna")
st.markdown("Speak your question using your microphone to receive divine advice from the Bhagwat Geeta.")

if st.button("Clear Conversation"):
    st.session_state.conversation_history = []
    st.experimental_rerun()

st.markdown("### Record Your Question")
audio_bytes = recorder.audio_recorder()

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_bytes)
    recognizer = sr.Recognizer()
    with sr.AudioFile("temp_audio.wav") as source:
        audio_data = recognizer.record(source)
    try:
        transcribed_text = recognizer.recognize_google(audio_data)
        st.write("You said:", transcribed_text)
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        transcribed_text = None
    os.remove("temp_audio.wav")
else:
    transcribed_text = None

user_query = st.text_input("Or type your question:")

if transcribed_text:
    query = transcribed_text
elif user_query:
    query = user_query
else:
    query = None

if st.button("Ask Krishna") and query:
    st.session_state.conversation_history.append(("User", query))
    with st.spinner("Retrieving context..."):
        context = retrieve_context(query)
    if not context:
        st.write("No relevant Bhagwat Geeta context found.")
    else:
        answer = llm_query(st.session_state.conversation_history, context, query)
        st.session_state.conversation_history.append(("Krishna", answer))
        st.markdown(f"**Krishna says:** {answer}")
        audio_data = text_to_speech(answer)
        if audio_data:
            st.audio(audio_data, format="audio/mp3")

if st.session_state.conversation_history:
    st.markdown("### Conversation History")
    for speaker, text in st.session_state.conversation_history:
        st.markdown(f"**{speaker}:** {text}")
