# main.py
import sys
import os

# --- Add project root to Python path ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- End path addition ---

import chainlit as cl
from companion_ai import memory, llm_interface, audio_utils # Import audio_utils
import speech_recognition as sr # Import SpeechRecognition

# --- Initialization ---
# Database Init
if not os.path.exists(memory.DB_PATH):
    print("Database not found, initializing...")
    memory.init_db()
else:
    print("Database found.")
    memory.init_db() # Safe to call again

# TTS Init
tts_engine = audio_utils.initialize_tts()

# --- STT Setup ---
stt_recognizer = sr.Recognizer()
stt_microphone = sr.Microphone()
# Adjust recognizer sensitivity to ambient noise once at the start
try:
    print("Adjusting microphone for ambient noise... Speak normally during this.")
    with stt_microphone as source:
        stt_recognizer.adjust_for_ambient_noise(source, duration=1.0) # Listen for 1 sec
    print(f"Ambient noise adjustment complete. Threshold: {stt_recognizer.energy_threshold:.2f}")
except Exception as e:
    print(f"Could not perform initial ambient noise adjustment: {e}")
    print("STT might be less accurate.")


# --- Chainlit Event Handlers ---

@cl.on_message
async def handle_message(message: cl.Message):
    """Called every time the user sends a message."""

    user_message = message.content
    print(f"\n--- User Message Received: {user_message} ---")

    # --- 1. Retrieve Memory ---
    profile_facts = memory.get_all_profile_facts()
    latest_summaries = memory.get_latest_summary(n=5)
    latest_insights = memory.get_latest_insights(n=5)

    memory_context = {
        "profile": profile_facts,
        "summaries": latest_summaries,
        "insights": latest_insights
    }

    # --- 2. Generate Response ---
    print("Generating AI response...")
    ai_response_text = llm_interface.generate_response(user_message, memory_context)
    print(f"AI Response Generated: '{ai_response_text[:100]}...'")

    # --- 3. Send Response ---
    print("Sending text response to UI...")
    msg = cl.Message(content=ai_response_text)
    await msg.send()

    # --- Speak Response ---
    if tts_engine:
        print("Sending response to TTS...")
        audio_utils.speak_text(ai_response_text)
        print("Finished TTS call.")
    else:
        print("TTS engine not available, skipping speech.")

    # --- 4. Update Memory ---
    print("Updating memory...")
    # 4a. Facts
    extracted_facts = llm_interface.extract_profile_facts(user_message, ai_response_text)
    if extracted_facts:
        print(f"Attempting to save extracted facts: {extracted_facts}")
        for key, value in extracted_facts.items():
            if isinstance(key, str) and isinstance(value, str) and key and value:
                memory.upsert_profile_fact(key, value)
                print(f"  Saved fact: {key} = {value}")
            else:
                print(f"  Skipped invalid fact: Key={key}, Value={value}")
    # 4b. Summary
    summary_text = llm_interface.generate_summary(user_message, ai_response_text)
    if summary_text:
        print(f"Attempting to save summary: {summary_text}")
        memory.add_summary(summary_text)
        print("  Summary saved.")
    else:
        print("  No summary generated or saved.")
    # 4c. Insight
    insight_text = llm_interface.generate_insight(user_message, ai_response_text, memory_context)
    if insight_text:
        print(f"Attempting to save insight: {insight_text}")
        memory.add_insight(insight_text)
        print("  Insight saved.")
    else:
        print("  No insight generated or saved.")
    print("--- Memory Update Complete ---")


# --- STT Action Button Callback ---
@cl.action_callback("stt_listen")
async def on_action(action: cl.Action):
    """Handles the 'Speak' button click."""
    print(f"\n--- Action received: {action.name} ---")
    await cl.Message(content="Listening... Please speak clearly.").send() # Feedback

    try:
        with stt_microphone as source:
            print("STT Listening for audio...")
            audio_data = stt_recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("STT Audio received, attempting recognition...")

        try:
            text = stt_recognizer.recognize_google(audio_data)
            print(f"STT Recognition successful: '{text}'")
            await cl.Message(author="User", content=text).send()

        except sr.UnknownValueError:
            print("STT Error: Google Web Speech could not understand audio")
            await cl.ErrorMessage(content="Sorry, I couldn't understand what you said.").send()
        except sr.RequestError as e:
            print(f"STT Error: Could not request results from Google Web Speech service; {e}")
            await cl.ErrorMessage(content="Sorry, I couldn't reach the speech recognition service.").send()
        except Exception as e:
            print(f"STT Error: An unexpected error occurred during recognition: {e}")
            await cl.ErrorMessage(content="Sorry, an unexpected error occurred during speech recognition.").send()

    except sr.WaitTimeoutError:
         print("STT Error: No speech detected within the timeout.")
         await cl.ErrorMessage(content="I didn't hear anything. Please try again.").send()
    except Exception as e:
        print(f"STT Error: Could not access microphone or listen: {e}")
        await cl.ErrorMessage(content=f"Error accessing microphone: {e}").send()

# --- Define the Action Button for the UI & Greet ---
@cl.on_chat_start
async def start_chat_and_actions():
    """Called when a new chat session starts. Sets up actions and greets."""
    # Setup Action Button *** SENT FIRST NOW ***
    actions = [
        cl.Action(
            name="stt_listen",
            value="listen",
            label="🎤 Speak",
            payload={}
        )
    ]
    await cl.ChatSettings(actions=actions).send()

    # Greet the user *** SENT SECOND NOW ***
    greeting = "Companion AI activated. How can I help you today?"
    await cl.Message(content=greeting).send()
    # Speak the greeting if TTS is available
    if tts_engine:
        audio_utils.speak_text(greeting)


# --- To Run ---
# (Instructions unchanged)