import speech_recognition as sr

def read_audio_microphone():
    """
    Listen for audio input from the microphone and return the recorded audio data.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak now...")
        # Optionally adjust for ambient noise:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)
    return audio

# Example usage in main.py
from stt_service import transcribe_audio
from conversation_manager import process_conversation

def main():
    # Use microphone input instead of a file.
    audio = read_audio_microphone()
    
    # Transcribe the audio input to obtain the query.
    query = transcribe_audio(audio)
    if query:
        print("User:", query)
        answer = process_conversation(query)
        print("Agent:", answer)
    else:
        print("Could not transcribe audio.")

if __name__ == "__main__":
    main()
