# AI-Buddy# Project Companion AI

A personalized conversational AI designed to function as a supportive companion, mentor, and friend. Built using Python, Chainlit, Google Gemini, and SQLite.

## Current Status (Updated - April 10, 2025)

*   **Core Structure:** Project structure established with Python.
*   **UI:** Basic chat interface implemented using Chainlit.
*   **LLM:** Connected to the Google Gemini API (specifically `gemini-1.5-flash-latest`) for language understanding and generation.
*   **Memory:** Persistent memory system implemented using SQLite.
    *   Database schema includes tables for `user_profile`, `conversation_summaries`, and `ai_insights`.
    *   Implemented saving & retrieval for all three memory types:
        *   **User Profile Facts:** Extracts and stores key facts about the user.
        *   **Conversation Summaries:** Generates and stores summaries of interaction turns.
        *   **AI Insights:** Generates and stores AI's reflective insights on the conversation/user.
*   **Interaction:** Basic text-based request-response loop functioning.
*   **Prompt Engineering:** System prompt refined iteratively to encourage more natural conversation, better context awareness, subtle memory usage, and reduced repetition.
*   **API Key Handling:** Uses `.env` file (ignored by Git) for API key management via `python-dotenv`.

*(Note: STT/TTS functionality was attempted but paused due to framework issues)*.

## Goals (from Briefing Document)

*   Develop an emergent, evolving personality.
*   Maintain robust long-term memory of interactions and user details.
*   Enable natural, multimodal conversation (Text, TTS, STT).
*   Future goals: Visual input, computer control, avatar integration.

## Setup & Running

1.  Clone the repository.
2.  Create a Python virtual environment: `python -m venv venv`
3.  Activate the environment: `source venv/bin/activate` (Linux/macOS) or `.\venv\Scripts\activate` (Windows)
4.  Install dependencies: `pip install -r requirements.txt`
5.  Create a `.env` file in the root directory and add your Google API key: `GOOGLE_API_KEY="YOUR_API_KEY"`
6.  Run the Chainlit app: `chainlit run main.py -w`
