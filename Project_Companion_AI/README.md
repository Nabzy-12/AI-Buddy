# AI-Buddy# Project Companion AI

A personalized conversational AI designed to function as a supportive companion, mentor, and friend. Built using Python, Chainlit, Google Gemini, and SQLite.

## Current Status (Initial Setup - April 10, 2025)

*   **Core Structure:** Project structure established with Python.
*   **UI:** Basic chat interface implemented using Chainlit.
*   **LLM:** Connected to the Google Gemini API (specifically `gemini-1.5-flash-latest`) for language understanding and generation.
*   **Memory:** Initial persistent memory system implemented using SQLite.
    *   Database schema includes tables for `user_profile`, `conversation_summaries`, and `ai_insights`.
    *   **Implemented Feature:** Basic extraction and storage of user profile facts (e.g., `user_name`) from conversations. The AI can recall these facts across sessions.
*   **Interaction:** Basic request-response loop functioning.

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