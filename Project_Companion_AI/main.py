# main.py (vNext with Listening State Management)

import asyncio
import base64
import io
import os
import sys
import traceback
import argparse
import json

import cv2
import pyaudio
import PIL.Image
import mss

from google import genai
from dotenv import load_dotenv

# --- Project Specific Imports ---
from companion_ai import llm_interface
from companion_ai import memory
# --- End Project Specific Imports ---

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
types = None

if not GOOGLE_API_KEY:
    print("ERROR: GOOGLE_API_KEY not found. Please set it in your .env file.")
    sys.exit(1)

if 'GOOGLE_API_KEY' not in os.environ:
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# ... (rest of the initial setup remains the same) ...
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
MODEL = "models/gemini-2.5-flash-preview-native-audio-dialog"
DEFAULT_VIDEO_MODE = "none" # Changed default to none as we focus on audio

client = genai.Client(api_key=GOOGLE_API_KEY, http_options={"api_version": "v1beta"})
from google.generativeai import types as direct_types_import
types = direct_types_import

LIVE_CONFIG = {
    "response_modalities": ["AUDIO"],
    "input_audio_transcription": {}
}

pya = pyaudio.PyAudio()
AUDIO_STREAM_SENTINEL = object()

class AudioLoop:
    def __init__(self, video_mode=DEFAULT_VIDEO_MODE):
        self.video_mode = video_mode
        self.audio_in_queue = asyncio.Queue()
        self.outgoing_payload_queue = asyncio.Queue(maxsize=100) # Increased size
        self.session = None
        self.audio_stream_in = None
        self.audio_stream_out = None
        self._running = True
        self.llm_iface = llm_interface
        self.mem = memory
        self.ai_response_trigger_task = None
        
        # --- NEW: State flag for managing the echo issue ---
        self.is_ai_speaking = False
        
        try:
            self.mem.init_db()
            print("INFO: Database initialized.")
        except Exception as e_db_init:
            print(f"ERROR: Failed to initialize database: {e_db_init}")

    # ... send_text_trigger, camera, and screen methods remain unchanged ...
    async def send_text_trigger(self):
        print("\nINFO: Press 'q' then Enter in terminal to quit.")
        while self._running:
            try:
                user_command = await asyncio.to_thread(input)
                if user_command.lower() == 'q':
                    self._running = False
                    if self.session: print("INFO: User typed 'q'. Initiating shutdown.")
                    if self.session and hasattr(self.session, 'close_send'):
                        await self.session.close_send()
                    break
            except (RuntimeError, asyncio.CancelledError):
                print("INFO: Text input trigger loop ending due to shutdown.")
                break
            except Exception as e:
                print(f"ERROR in send_text_trigger: {e}")
                self._running = False
                break
        print("INFO: Text input trigger loop finished.")

    # --- THIS IS THE FIRST MAJOR CHANGE ---
    async def listen_microphone(self):
        print("INFO: Starting microphone listening...")
        # ... (microphone opening logic is the same) ...
        try:
            mic_info = pya.get_default_input_device_info()
            device_index = mic_info.get("index")
            self.audio_stream_in = await asyncio.to_thread(
                pya.open, format=FORMAT, channels=CHANNELS, rate=SEND_SAMPLE_RATE,
                input=True, input_device_index=device_index, frames_per_buffer=CHUNK_SIZE
            )
        except Exception as e:
            print(f"ERROR: Could not open microphone: {e}"); self._running = False; return

        print("INFO: Microphone opened. Streaming audio to Gemini...")
        try:
            while self._running and self.audio_stream_in and not self.audio_stream_in.is_stopped():
                try:
                    data = await asyncio.to_thread(self.audio_stream_in.read, CHUNK_SIZE, exception_on_overflow=False)
                    
                    # *** THE CORE LOGIC CHANGE IS HERE ***
                    # Only send audio to the AI if the AI is NOT currently speaking.
                    if self._running and not self.is_ai_speaking:
                        await self.outgoing_payload_queue.put({"audio": {"data": data}})
                        
                except IOError as e:
                    if hasattr(e, 'errno') and e.errno == pyaudio.paInputOverflowed:
                        print("WARNING: Microphone input overflowed."); continue
                    print(f"ERROR: Microphone read error: {e}"); self._running = False; break
                except Exception as e:
                    print(f"ERROR: Unexpected in listen_microphone: {e}"); self._running = False; break
        finally:
            # ... (cleanup is the same) ...
            if self.audio_stream_in:
                try:
                    if self.audio_stream_in.is_active(): self.audio_stream_in.stop_stream()
                    self.audio_stream_in.close()
                except Exception as e_close: print(f"Error closing input stream: {e_close}")
            print("INFO: Microphone listening stopped.")
            
    # --- THIS IS THE SECOND MAJOR CHANGE ---
    async def play_audio_output(self):
        print("INFO: Starting audio playback...")
        try:
            self.audio_stream_out = await asyncio.to_thread(
                pya.open, format=FORMAT, channels=CHANNELS, rate=RECEIVE_SAMPLE_RATE,
                output=True, frames_per_buffer=CHUNK_SIZE*2
            )
        except Exception as e: print(f"ERROR: Could not open audio output stream: {e}"); self._running = False; return
        
        try:
            while self._running and self.audio_stream_out and not self.audio_stream_out.is_stopped():
                try:
                    # *** THE CORE LOGIC CHANGE IS HERE ***
                    
                    # 1. Set state to SPEAKING before we start playing
                    self.is_ai_speaking = True
                    
                    audio_chunk = await self.audio_in_queue.get()
                    if audio_chunk is None or not self._running:
                        self.audio_in_queue.task_done(); break
                    
                    # 2. Write the audio chunk to the speaker
                    await asyncio.to_thread(self.audio_stream_out.write, audio_chunk)
                    self.audio_in_queue.task_done()
                    
                    # 3. After playing, if the queue is empty, set state back to NOT SPEAKING
                    if self.audio_in_queue.empty():
                        self.is_ai_speaking = False

                except Exception as e:
                    print(f"ERROR during audio playback write: {e}")
                    self.is_ai_speaking = False # Ensure flag is reset on error
                    break
        except asyncio.CancelledError:
            print("INFO: play_audio_output task cancelled.")
        finally:
            self.is_ai_speaking = False # Final safety reset
            # ... (cleanup is the same) ...
            if self.audio_stream_out:
                try:
                    if self.audio_stream_out.is_active(): self.audio_stream_out.stop_stream()
                    self.audio_stream_out.close()
                except Exception as e_close: print(f"Error closing output stream: {e_close}")
            print("INFO: Audio playback stopped.")


    # ... (send_realtime_payloads, _trigger_ai_response_after_delay, receive_from_gemini, run_main_loop, __main__) ...
    # ... (These other methods remain the same as the last version I provided) ...
    # For completeness, I will include them here.

    async def send_realtime_payloads(self):
        print("INFO: Starting to send realtime payloads to Gemini.")
        try:
            while self._running:
                payload_item = await self.outgoing_payload_queue.get()
                if payload_item is None or not self._running:
                    self.outgoing_payload_queue.task_done(); break
                data_to_send_kwargs = {}
                if "audio" in payload_item and types:
                    data_to_send_kwargs['audio'] = types.Blob(data=payload_item["audio"]["data"], mime_type="audio/pcm")
                if data_to_send_kwargs and self.session:
                    try: await self.session.send_realtime_input(**data_to_send_kwargs)
                    except Exception as e:
                        print(f"ERROR sending realtime input: {e}"); self._running = False; break
                self.outgoing_payload_queue.task_done()
        except asyncio.CancelledError: print("INFO: send_realtime_payloads task cancelled.")
        finally: print("INFO: Realtime payload sending stopped.")


    async def _trigger_ai_response_after_delay(self, final_user_speech: str):
        try:
            await asyncio.sleep(1.2)
            print(f"INFO: Silence timer elapsed. Processing final STT: '{final_user_speech}'")
            # ... The rest of this function is identical to the previous version ...
            # 1. Retrieve memory context
            memory_context_dict = {
                "profile": await asyncio.to_thread(self.mem.get_all_profile_facts),
                "summaries": await asyncio.to_thread(self.mem.get_latest_summary, n=2),
                "insights": await asyncio.to_thread(self.mem.get_latest_insights, n=2)
            }
            # 2. Generate AI's custom text response
            ai_custom_text_response = await asyncio.to_thread(
                self.llm_iface.generate_response, final_user_speech, memory_context_dict)
            print(f"INFO: Companion AI Custom Text: {ai_custom_text_response}")

            # 3. Send custom response to Gemini for TTS
            if ai_custom_text_response and self.session:
                print("INFO: Sending Companion AI response to Gemini for TTS...")
                await self.session.send_client_content(
                    turns=[{"role": "user", "parts": [{"text": ai_custom_text_response}]}],
                    turn_complete=True)

            # 4. Update memory in the background
            asyncio.create_task(self.update_memory_async(final_user_speech, ai_custom_text_response, memory_context_dict))

        except asyncio.CancelledError:
            print("INFO: AI response trigger was cancelled by subsequent user speech.")
        except Exception as e:
            print(f"ERROR in _trigger_ai_response_after_delay: {e}"); traceback.print_exc()

    async def update_memory_async(self, user_msg, ai_msg, context):
        """Asynchronous memory update tasks."""
        summary = await asyncio.to_thread(self.llm_iface.generate_summary, user_msg, ai_msg)
        if summary:
            await asyncio.to_thread(self.mem.add_summary, summary); print(f"INFO: Added summary: {summary[:70]}...")
        
        facts = await asyncio.to_thread(self.llm_iface.extract_profile_facts, user_msg, ai_msg)
        if facts:
            for key, value in facts.items():
                await asyncio.to_thread(self.mem.upsert_profile_fact, key, value)
            print(f"INFO: Upserted profile facts: {facts}")

        insight = await asyncio.to_thread(self.llm_iface.generate_insight, user_msg, ai_msg, context)
        if insight:
            await asyncio.to_thread(self.mem.add_insight, insight); print(f"INFO: Added AI insight: {insight[:70]}...")

    async def receive_from_gemini(self):
        print("INFO: Listening for responses from Gemini...")
        accumulated_stt = ""
        try:
            while self._running and self.session:
                # ... This function is identical to the previous version ...
                turn_iterator = self.session.receive()
                async for response in turn_iterator:
                    if not self._running: break
                    transcript_part = ""
                    is_stt_final = False
                    if (sc := getattr(response, 'server_content', None)):
                        if (it := getattr(sc, 'input_transcription', None)):
                            transcript_part = it.text.strip()
                        if sc.turn_complete: is_stt_final = True
                    
                    if transcript_part:
                        print(f"STT (part): {transcript_part}")
                        if self.ai_response_trigger_task and not self.ai_response_trigger_task.done():
                            self.ai_response_trigger_task.cancel()
                        accumulated_stt += transcript_part + " "
                    
                    if is_stt_final and accumulated_stt.strip():
                        final_speech = accumulated_stt.strip()
                        accumulated_stt = ""
                        print(f"INFO: Gemini detected pause. Starting silence timer for: '{final_speech}'")
                        if self.ai_response_trigger_task and not self.ai_response_trigger_task.done():
                            self.ai_response_trigger_task.cancel()
                        self.ai_response_trigger_task = asyncio.create_task(
                            self._trigger_ai_response_after_delay(final_speech))
                    
                    if hasattr(response, 'data') and response.data:
                        await self.audio_in_queue.put(response.data)

        except asyncio.CancelledError: print("INFO: receive_from_gemini task cancelled.")
        except Exception as e: print(f"ERROR in receive_from_gemini: {e}"); self._running = False
        finally: print("INFO: Gemini response listening stopped.")


    async def run_main_loop(self):
        print(f"INFO: Connecting to Gemini Live model: {MODEL}")
        try:
            async with client.aio.live.connect(model=MODEL, config=LIVE_CONFIG) as session:
                self.session = session
                print("INFO: Successfully connected to Gemini Live session!")
                
                # Using a TaskGroup for better error handling and cleanup
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self.receive_from_gemini())
                    tg.create_task(self.play_audio_output())
                    tg.create_task(self.listen_microphone())
                    tg.create_task(self.send_realtime_payloads())
                    tg.create_task(self.send_text_trigger()) # To handle 'q' for quit
        
        except* Exception as eg:
            print(f"FATAL: One or more tasks failed: {eg.exceptions}")
        finally:
            self._running = False
            # Cleanly stop PyAudio
            if pya: pya.terminate()
            print("INFO: Application cleanup finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # No video modes for now to focus on audio loop
    args = parser.parse_args()
    print("Starting Project Companion AI...")
    
    loop_runner = AudioLoop()
    try:
        asyncio.run(loop_runner.run_main_loop())
    except KeyboardInterrupt:
        print("\nINFO: Keyboard interrupt. Exiting.")
    finally:
        print("INFO: Main execution finished.")