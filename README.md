# Ask Lord Krishna

Ask Lord Krishna is a Streamlit web application that brings ancient wisdom to modern times. This application allows users to ask questions using either voice or text and receive spiritual advice inspired by the Bhagwat Geeta, all in the melodious voice of Lord Krishna.

## Overview

The project leverages multiple technologies:
- **Speech Recognition:** Converts your spoken questions into text.
- **Contextual Retrieval:** Uses a local ChromaDB index built from the Bhagwat Geeta text to retrieve relevant context.
- **Generative AI:** Integrates with Google Generative AI (Gemini-2.0 Flash) to generate responses in the voice of Lord Krishna.
- **Text-to-Speech:** Uses gTTS to convert Krishna's advice into spoken audio.

## Features

- **Audio and Text Input:** Ask your question by recording audio or typing.
- **Conversational History:** Displays a running history of your conversation with Lord Krishna.
- **Dynamic Context Retrieval:** Automatically retrieves relevant passages from the Bhagwat Geeta to provide context-aware responses.
- **Interactive Experience:** Enjoy a seamless interaction that blends ancient wisdom with modern AI capabilities.

## Installation

### Prerequisites

- Python 3.7 or higher
- A valid Google API Key for the Generative AI service

### Dependencies

Install the necessary Python packages with:

```bash
pip install streamlit gtts SpeechRecognition audio_recorder_streamlit langchain chromadb huggingface-hub langchain_google_genai
