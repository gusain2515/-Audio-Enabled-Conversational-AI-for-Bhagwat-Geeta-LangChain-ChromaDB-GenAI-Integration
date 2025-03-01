import speech_recognition as sr

def transcribe_audio(audio):
    """
    Convert audio to text using Google Speech Recognition.
    """
    recognizer = sr.Recognizer()
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        print("STT request error:", e)
        return ""