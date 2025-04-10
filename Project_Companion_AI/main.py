# main.py
import sys
import os
# import io # No longer needed for audio buffer

# --- Add project root to Python path ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- End path addition ---

import chainlit as cl
from companion_ai import memory, llm_interface # No longer importing audio_utils
# import speech_recognition as sr # No longer needed
import asyncio # Keep for potential future async operations

# --- Initialization ---
# Database Init
if not os.path.exists(memory.DB_PATH): print("Database not found, initializing..."); memory.init_db()
else: print("Database found."); memory.init_db()

# --- TTS/STT Setup Removed ---
# tts_engine = audio_utils.initialize_tts() # Removed
# stt_recognizer = sr.Recognizer() # Removed

# --- Chainlit Event Handlers ---

@cl.on_chat_start
async def start_chat():
    """Greets the user."""
    # Removed audio buffer/state initialization
    # Removed ChatProfile/Action setup

    greeting = "Companion AI activated. How can I help you today?" # Reverted greeting
    await cl.Message(content=greeting).send()

    # Removed TTS call for greeting
    # if tts_engine: audio_utils.speak_text(greeting)


@cl.on_message
async def handle_message(message: cl.Message):
    """Handles incoming text messages."""
    user_message = message.content
    print(f"\n--- User Message Received: {user_message} ---")

    # --- 1. Retrieve Memory ---
    profile_facts = memory.get_all_profile_facts()
    latest_summaries = memory.get_latest_summary(n=5)
    latest_insights = memory.get_latest_insights(n=5)
    memory_context = { "profile": profile_facts, "summaries": latest_summaries, "insights": latest_insights }

    # --- 2. Generate Response ---
    print("Generating AI response...")
    ai_response_text = llm_interface.generate_response(user_message, memory_context)
    print(f"AI Response Generated: '{ai_response_text[:100]}...'")

    # --- 3. Send Response ---
    print("Sending text response to UI...")
    await cl.Message(content=ai_response_text).send()

    # --- Speak Response Removed ---
    # if tts_engine: audio_utils.speak_text(ai_response_text)

    # --- 4. Update Memory ---
    print("Updating memory...")
    # (Memory update logic unchanged)
    extracted_facts = llm_interface.extract_profile_facts(user_message, ai_response_text)
    if extracted_facts:
        print(f"Attempting to save extracted facts: {extracted_facts}")
        for key, value in extracted_facts.items():
             if isinstance(key, str) and isinstance(value, str) and key and value: memory.upsert_profile_fact(key, value); print(f"  Saved fact: {key} = {value}")
             else: print(f"  Skipped invalid fact: Key={key}, Value={value}")
    summary_text = llm_interface.generate_summary(user_message, ai_response_text)
    if summary_text: print(f"Attempting to save summary: {summary_text}"); memory.add_summary(summary_text); print("  Summary saved.")
    else: print("  No summary generated or saved.")
    insight_text = llm_interface.generate_insight(user_message, ai_response_text, memory_context)
    if insight_text: print(f"Attempting to save insight: {insight_text}"); memory.add_insight(insight_text); print("  Insight saved.")
    else: print("  No insight generated or saved.")
    print("--- Memory Update Complete ---")

# --- Audio Handlers Removed ---
# @cl.on_audio_start ... removed
# @cl.on_audio_chunk ... removed
# @cl.on_audio_end ... removed


# --- To Run ---
# 1. Ensure venv is active
# 2. Ensure .env file exists with GOOGLE_API_KEY
# 3. Ensure dependencies are installed: pip install -r requirements.txt
# 4. Run: chainlit run main.py -w