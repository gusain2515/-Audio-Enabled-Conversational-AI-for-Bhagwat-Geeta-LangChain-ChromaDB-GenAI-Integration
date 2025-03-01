from transformers import pipeline
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_fallback_answer(query):
    """
    Use a generative model as a fallback when no relevant context is found.
    This version uses a more advanced model (flan-t5-large) for better responses.
    """
    try:
        # Initialize the text generation pipeline with a better model
        # Using flan-t5-large, a more powerful and instruction-tuned model
        text_generator = pipeline("text2text-generation", model="google/flan-t5-large", device=0)
        
        # Create a structured prompt to guide the model's response
        prompt = f"Answer the following question in a clear and concise manner: {query}"
        
        # Generate the response using the text generation model
        response = text_generator(prompt, max_length=100, do_sample=True, temperature=0.7)
        
        # Extract the generated text from the response
        generated_text = response[0]['generated_text']
        
        # Log the generated text for debugging
        logger.info(f"Fallback API Response: {generated_text}")
        
        # Return the generated text
        return generated_text
    
    except Exception as e:
        # Log any errors that occur during the fallback process
        logger.error(f"Fallback API error: {e}")
        return "I'm sorry, I cannot generate an answer at this time."