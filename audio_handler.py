import speech_recognition as sr

def read_audio_file(audio_file_path):
    """
    Read audio from a file and return the audio data.
    """
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio = recognizer.record(source)
        return audio
    except Exception as e:
        print("Error reading audio file:", e)
        return None
