# companion_ai/audio_utils.py

import pyttsx3
import time # Import time for potential small delays

engine = None # Global engine variable

def initialize_tts():
    """Initializes the TTS engine."""
    global engine
    if engine is None:
        try:
            print("Initializing TTS engine...")
            engine = pyttsx3.init()

            # --- Optional Engine Configuration ---
            # Rate
            rate = engine.getProperty('rate')
            engine.setProperty('rate', rate - 40) # Slow down slightly (default is often 200)

            # Volume
            # volume = engine.getProperty('volume')
            # engine.setProperty('volume', 1.0) # Max volume

            # Voice (List available voices and choose one)
            voices = engine.getProperty('voices')
            print("Available Voices:")
            for i, voice in enumerate(voices):
                print(f"  {i}: ID: {voice.id} - Name: {voice.name} - Lang: {voice.languages}")

            # --- !! SELECT VOICE HERE !! ---
            # Try selecting a specific voice by index or ID if the default isn't great
            # Example (use index): Set index based on the printed list (e.g., 0 for first, 1 for second)
            voice_index_to_use = 0 # <<< CHANGE THIS INDEX if needed
            if voices and len(voices) > voice_index_to_use:
                 engine.setProperty('voice', voices[voice_index_to_use].id)
                 print(f"Using Voice: {voices[voice_index_to_use].name}")
            else:
                 print("Could not find specified voice index, using default.")
            # Example (use ID - copy ID from the printed list):
            # desired_voice_id = "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_DAVID_11.0" # Example Windows ID
            # engine.setProperty('voice', desired_voice_id)
            # ---------------------------------

            print("TTS engine initialized.")
        except Exception as e:
            print(f"Error initializing TTS engine: {e}")
            engine = None # Ensure engine is None if init fails
    return engine

def speak_text(text_to_speak: str):
    """Uses the initialized TTS engine to speak the given text."""
    global engine
    if engine is None:
        print("TTS engine not initialized. Cannot speak.")
        return

    try:
        # Simple blocking call for now
        print(f"TTS attempting to speak: '{text_to_speak[:50]}...'") # Log first 50 chars
        engine.say(text_to_speak)
        engine.runAndWait() # Blocks until speech is finished
        # Add a tiny delay AFTER speaking to prevent issues if called rapidly (optional)
        # time.sleep(0.1)
        print("TTS finished speaking.")
    except RuntimeError as e:
        # Sometimes runAndWait can cause issues if called improperly
        print(f"TTS runtime error: {e}. Attempting recovery (or skip).")
        # Potentially try re-initializing or just ignore for this attempt
        # engine = None # Force re-init next time? Careful with recursion.
    except Exception as e:
        print(f"Error during TTS speech: {e}")

# Optional: Clean shutdown (might be needed in complex apps)
# def shutdown_tts():
#     global engine
#     if engine:
#         try:
#             engine.stop()
#         except Exception as e:
#             print(f"Error stopping TTS engine: {e}")
#         engine = None