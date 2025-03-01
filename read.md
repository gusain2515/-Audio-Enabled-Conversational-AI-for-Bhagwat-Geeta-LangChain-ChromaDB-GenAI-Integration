# Voice-Enabled Conversational AI with Retrieval-Augmented Generation 
Overview
Purpose: Provide context-aware answers using a hybrid approach (pre-defined knowledge + generative AI) with voice interaction.

Target Audience: Developers, AI enthusiasts, and researchers interested in conversational AI systems.

Key Technologies:

Python, Hugging Face Transformers, ChromaDB

SpeechRecognition, gTTS, Wikipedia API

Models: deepset/roberta-base-squad2, google/flan-t5-large

✨ Features
Real-Time Speech-to-Text with microphone input

Retrieval-Augmented Generation (RAG) using ChromaDB vector storage

Fallback Mechanism to FLAN-T5 generative model when context is missing

Intent Detection for structured actions (e.g., appointment booking)

Conversation History tracking with vector embeddings

Text-to-Speech responses with auto-cleanup

graph TD
    A[Microphone Input] --> B(STT Service)
    B --> C{Intent Detected?}
    C -->|Yes| D[Execute Intent]
    C -->|No| E[RAG Retriever]
    E --> F{Context Found?}
    F -->|Yes| G[QA Model Answer]
    F -->|No| H[Fallback Generator]
    G/H --> I[TTS Response]
