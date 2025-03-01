# conversation_manager.py

from transformers import pipeline
from rag_retriever import retrieve_context, store_conversation_history, retrieve_conversation_history
from fallback_api import generate_fallback_answer
from gtts import gTTS
import os
from playsound import playsound
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_response(query, context):
    """
    Use a QA model to extract an answer from the provided context.
    """
    try:
        # Initialize the QA pipeline with a pre-trained model
        qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", device=0)
        
        # Generate the response using the QA model
        response = qa_pipeline(question=query, context=context)
        answer = response.get("answer", "")
        
        # Log the response for debugging
        logger.info(f"QA Model Response: {answer}")
        
        return answer
    except Exception as e:
        logger.error(f"Error in generate_response: {e}")
        return ""

def text_to_speech(text):
    """
    Convert text to speech using gTTS and play the audio.
    """
    try:
        # Generate speech from text
        tts = gTTS(text=text, lang="en")
        
        # Save the speech to a temporary file
        temp_file = "response.mp3"
        tts.save(temp_file)
        
        # Play the audio file
        playsound(temp_file)
        
        # Remove the temporary file after playing
        os.remove(temp_file)
    except Exception as e:
        logger.error(f"TTS error: {e}")

def process_conversation(query):
    try:
        logger.info(f"User Query: {query}")
        
        # Retrieve context and generate response
        context = retrieve_context(query)
        if context:
            answer = generate_response(query, context)
            if answer:
                logger.info(f"Generated Answer: {answer}")
                text_to_speech(answer)
                store_conversation_history(query, answer)
                return answer
        
        # Fallback if no context or answer is found
        logger.info("No relevant context found. Using fallback API.")
        answer = generate_fallback_answer(query)
        logger.info(f"Fallback Answer: {answer}")
        text_to_speech(answer)
        store_conversation_history(query, answer)
        return answer
    
    except Exception as e:
        logger.error(f"Error in process_conversation: {e}")
        return "I'm sorry, I encountered an error while processing your request."