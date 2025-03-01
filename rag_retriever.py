# rag_retriever.py

import chromadb
from sentence_transformers import SentenceTransformer
import wikipedia

class ChromaRetriever:
    def __init__(self, collection_name, documents):
        """
        Initialize the ChromaDB retriever using default client settings.
        :param collection_name: Name of the Chroma collection.
        :param documents: List of document strings.
        """
        self.documents = documents
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # Compute document embeddings.
        self.embeddings = self.model.encode(documents, convert_to_numpy=True)
        
        # Create a Chroma client using default settings (in-memory by default).
        self.client = chromadb.Client()
        
        # Create a collection in the Chroma database.
        self.collection = self.client.create_collection(name=collection_name)
        
        # Add documents to the collection.
        ids = [f"doc{i}" for i in range(len(documents))]
        self.collection.add(
            ids=ids,
            embeddings=self.embeddings.tolist(),
            metadatas=[{"text": doc} for doc in documents],
            documents=documents
        )
    
    def retrieve(self, query, top_k=1, threshold=0.5):
        """
        Retrieve the most relevant document from the collection.
        :param query: The user query string.
        :param top_k: Number of top results to return.
        :param threshold: Maximum acceptable distance for a match.
        :return: The document text if a match is found, otherwise None.
        """
        query_embedding = self.model.encode(query, convert_to_numpy=True).tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        distances = results['distances'][0]
        documents = results['documents'][0]
        if distances and distances[0] < threshold:
            return documents[0]
        return None

def get_external_context(query, sentences=2):
    """
    Fallback external retrieval using the Wikipedia API.
    Returns a summary of the topic if found.
    """
    try:
        summary = wikipedia.summary(query, sentences=sentences)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            # If the query is ambiguous, take the first option.
            summary = wikipedia.summary(e.options[0], sentences=sentences)
            return summary
        except Exception:
            return None
    except Exception:
        return None

# Initialize a global retriever instance with sample documents.
sample_documents = [
    "New Delhi is the capital of India. It is located in the northern part of the country and serves as the center of government, culture, and commerce. New Delhi is known for its historical sites and vibrant urban life."
    # You can add more documents here.
]

global_chroma_retriever = ChromaRetriever("document_collection", sample_documents)

# Initialize a global conversation history collection
conversation_history_collection = global_chroma_retriever.client.create_collection(name="conversation_history")

def retrieve_context(query):
    """
    Retrieve context from the vector database.
    If no relevant document is found, fall back to an external API (Wikipedia).
    """
    context = global_chroma_retriever.retrieve(query)
    if context:
        return context
    return get_external_context(query)

def store_conversation_history(query, response):
    """
    Store the conversation history in the ChromaDB collection.
    """
    # Generate embeddings for the query and response
    query_embedding = global_chroma_retriever.model.encode(query, convert_to_numpy=True).tolist()
    response_embedding = global_chroma_retriever.model.encode(response, convert_to_numpy=True).tolist()
    
    # Store the conversation in the history collection
    conversation_history_collection.add(
        ids=[f"conv{len(conversation_history_collection.get()['ids'])}"],
        embeddings=[query_embedding],
        metadatas=[{"query": query, "response": response}],
        documents=[response]
    )

def retrieve_conversation_history(query, top_k=5):
    """
    Retrieve the most relevant conversation history based on the query.
    """
    query_embedding = global_chroma_retriever.model.encode(query, convert_to_numpy=True).tolist()
    results = conversation_history_collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results['documents'][0]