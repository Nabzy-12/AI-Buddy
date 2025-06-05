# main.py (Full Integration with Custom AI Logic)

import asyncio
import base64
import io
import os
import sys
import traceback
import argparse 
import json # For llm_interface.extract_profile_facts

import cv2
import pyaudio 
import PIL.Image 
import mss 

from google import genai
from dotenv import load_dotenv 

# --- Project Specific Imports ---
from companion_ai import llm_interface # Your AI brain
from companion_ai import memory      # Your memory system
# --- End Project Specific Imports ---

load_dotenv() 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
types = None # Will be set after client init

if not GOOGLE_API_KEY:
    print("ERROR: GOOGLE_API_KEY not found. Please set it in your .env file.")
    sys.exit(1) 

if 'GOOGLE_API_KEY' not in os.environ and GOOGLE_API_KEY:
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
if GOOGLE_API_KEY:
    # This should use the key from os.environ if using genai.Client() later
    # If genai.Client() doesn't take api_key arg, this global configure is fallback
    try:
        genai.configure(api_key=GOOGLE_API_KEY) 
    except AttributeError:
        # This might happen if `from google import genai` doesn't expose `configure` directly
        # and `genai.Client()` handles the key. We'll proceed as Client can take it.
        print("INFO: genai.configure() not found on 'genai' module. Assuming genai.Client() handles API key.")


if sys.version_info < (3, 11, 0):
    try:
        import taskgroup # type: ignore
        import exceptiongroup # type: ignore
        asyncio.TaskGroup = taskgroup.TaskGroup # type: ignore
        asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup # type: ignore
    except ImportError:
        print("Warning: 'taskgroup' or 'exceptiongroup' backports not found for Python < 3.11.")

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000  
RECEIVE_SAMPLE_RATE = 24000 
CHUNK_SIZE = 1024         

MODEL = "models/gemini-2.0-flash-live-001" 
DEFAULT_VIDEO_MODE = "camera" 

try:
    # Pass api_key directly if client supports it, otherwise it uses env var or global config
    client = genai.Client(api_key=GOOGLE_API_KEY, http_options={"api_version": "v1beta"})
    
    # Try to get types from the initialized client or the genai module
    if hasattr(client, 'types'): # Some SDKs might expose it on the client instance
        types = client.types
    elif hasattr(genai, 'types'):
        types = genai.types
    else: 
        from google.generativeai import types as direct_types_import # Fallback
        types = direct_types_import
    print(f"INFO: Successfully obtained 'types' module for Gemini SDK (version {genai.__version__ if hasattr(genai, '__version__') else 'unknown'})")

except AttributeError as e_client_attr:
    sdk_version = "unknown"; _=sdk_version
    try: sdk_version = genai.__version__ 
    except: pass
    print(f"ERROR: genai.Client() failed or 'types' could not be accessed. Attribute Error: {e_client_attr}")
    print(f"Current google-genai version: {sdk_version}")
    sys.exit(1)
except Exception as e_client:
    print(f"ERROR: Unexpected error initializing genai.Client(): {e_client}")
    sys.exit(1)

LIVE_CONFIG = {
    "response_modalities": ["AUDIO", "TEXT"], 
    "input_audio_transcription": {},
    "speech_config": types.SpeechConfig( 
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
        )
    ) if types else {},
    "realtime_input_config": { # ADD THIS BACK OR MODIFY IT
        "automatic_activity_detection": {"disabled": False}, 
        "activity_handling": "NO_INTERRUPTION" # CHANGED FROM INTERRUPT_IF_NO_ACTIVITY
    }
}

if not types: # If types couldn't be imported, remove speech_config as it will fail
    print("WARNING: `types` module not available, removing speech_config from LIVE_CONFIG.")
    LIVE_CONFIG.pop("speech_config", None)


pya = pyaudio.PyAudio()

def try_put_sentinel(queue, item=None):
    # ... (same as before) ...
    if queue:
        try: queue.put_nowait(item if item is not None else AUDIO_STREAM_SENTINEL)
        except asyncio.QueueFull: print(f"Warning: Queue was full for sentinel.")
        except Exception as e: print(f"Warning: Error putting sentinel: {e}")

AUDIO_STREAM_SENTINEL = object()

class AudioLoop:
    def __init__(self, video_mode=DEFAULT_VIDEO_MODE):
        self.video_mode = video_mode
        self.audio_in_queue = asyncio.Queue() 
        self.outgoing_payload_queue = asyncio.Queue(maxsize=20) 
        self.session = None
        self.audio_stream_in = None 
        self.audio_stream_out = None 
        self._running = True
        # Store direct references to your modules for use in methods
        self.llm_iface = llm_interface 
        self.mem = memory            
        # Initialize database (this is synchronous, so call it directly here)
        try:
            self.mem.init_db()
            print("INFO: Database initialized by AudioLoop constructor.")
        except Exception as e_db_init:
            print(f"ERROR: Failed to initialize database: {e_db_init}")
            # Decide if this is a fatal error
            # sys.exit(1) 


    async def send_text_trigger(self):
        # ... (same as before) ...
        print("\nINFO: Press Enter in terminal to signal AI to respond to latest speech, or type 'q' to quit.")
        while self._running:
            try:
                user_command = await asyncio.to_thread(input, "Action (Enter to process speech / 'q' to quit) > ")
                if user_command.lower() == 'q':
                    self._running = False
                    if self.session: print("INFO: User typed 'q'.")
                    try_put_sentinel(self.outgoing_payload_queue); try_put_sentinel(self.audio_in_queue)
                    if self.session and hasattr(self.session, 'close_send') and callable(self.session.close_send):
                        print("INFO: Closing send stream to Gemini.")
                        await self.session.close_send()
                    break
                print("INFO: Enter pressed. AI will respond to next complete speech segment.")
            except RuntimeError as e: 
                if "cannot schedule new futures after shutdown" in str(e) or "Event loop is closed" in str(e): 
                    print("INFO: Text input trigger loop ending due to shutdown."); break
                else: print(f"ERROR in send_text_trigger (RuntimeError): {e}"); self._running = False; break
            except Exception as e: print(f"ERROR in send_text_trigger: {e}"); self._running = False; break
        print("INFO: Text input trigger loop finished.")


    # ... Video methods _get_   _from_camera, stream_camera_frames, _get_screen_frame, stream_screen_frames remain the same ...
    # ... (No changes needed here from your last full working code for these)
    # ... (inside AudioLoop class) ...

    def _get_frame_from_camera(self, cap): # self is needed for instance method
        ret, frame = cap.read()
        if not ret: 
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb) # CORRECTED: Was 'древний'
        img.thumbnail((1024, 1024)) # Ensure this is a tuple
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        return {"mime_type": "image/jpeg", "data": base64.b64encode(image_io.getvalue()).decode()}

    # ... (the rest of the AudioLoop class) ...

    async def stream_camera_frames(self):
        print("INFO: Starting camera streaming...")
        cap = None
        try:
            cap = await asyncio.to_thread(cv2.VideoCapture, 0)
            if not cap.isOpened(): print("ERROR: Cannot open camera"); return
            while self._running and cap.isOpened():
                frame_data = await asyncio.to_thread(self._get_frame_from_camera, cap)
                if frame_data: await self.outgoing_payload_queue.put({"image": frame_data})
                await asyncio.sleep(1.0) 
        except Exception as e: print(f"ERROR in stream_camera_frames: {e}")
        finally:
            if cap and cap.isOpened(): cap.release()
            print("INFO: Camera streaming stopped.")

    def _get_screen_frame(self, sct_instance):
        monitor_idx = 1 if len(sct_instance.monitors) > 1 else 0 
        monitor = sct_instance.monitors[monitor_idx] 
        sct_img = sct_instance.grab(monitor)
        img = PIL.Image.frombytes("RGB", (sct_img.width, sct_img.height), sct_img.rgb)
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        return {"mime_type": "image/jpeg", "data": base64.b64encode(image_io.getvalue()).decode()}

    async def stream_screen_frames(self):
        print("INFO: Starting screen streaming...")
        sct_instance = None
        try:
            sct_instance = await asyncio.to_thread(mss.mss)
            while self._running:
                frame_data = await asyncio.to_thread(self._get_screen_frame, sct_instance)
                if frame_data: await self.outgoing_payload_queue.put({"image": frame_data})
                await asyncio.sleep(1.0)
        except Exception as e: print(f"ERROR in stream_screen_frames: {e}")
        finally: print("INFO: Screen streaming stopped.")

    async def listen_microphone(self):
        # ... (same as before) ...
        print("INFO: Starting microphone listening...")
        try:
            mic_info_tuple = pya.get_default_input_device_info(), # This returns a tuple
            mic_info = mic_info_tuple[0] # Get the dict from the tuple
            device_index = mic_info.get("index")
            if not isinstance(device_index, (int, type(None))):
                 print(f"Warning: mic_info index is not an int: {device_index}. Using default.")
                 device_index = None
            self.audio_stream_in = await asyncio.to_thread(
                pya.open, format=FORMAT, channels=CHANNELS, rate=SEND_SAMPLE_RATE,
                input=True, input_device_index=device_index, frames_per_buffer=CHUNK_SIZE
            )
        except Exception as e:
            print(f"ERROR: Could not open microphone: {e}"); self._running = False 
            try_put_sentinel(self.outgoing_payload_queue); try_put_sentinel(self.audio_in_queue); return

        print("INFO: Microphone opened. Streaming audio to Gemini...")
        try:
            while self._running and self.audio_stream_in and not self.audio_stream_in.is_stopped():
                try:
                    data = await asyncio.to_thread(self.audio_stream_in.read, CHUNK_SIZE, exception_on_overflow=False)
                    if self._running: await self.outgoing_payload_queue.put({"audio": {"data": data, "mime_type": "audio/pcm"}})
                except IOError as e: 
                    if hasattr(e, 'errno') and e.errno == pyaudio.paInputOverflowed: # type: ignore
                        print("WARNING: Microphone input overflowed."); continue
                    print(f"ERROR: Microphone read error: {e}"); self._running = False; break
                except Exception as e: print(f"ERROR: Unexpected in listen_microphone: {e}"); self._running = False; break
        finally:
            if self.audio_stream_in:
                try:
                    if self.audio_stream_in.is_active(): self.audio_stream_in.stop_stream()
                    self.audio_stream_in.close()
                except Exception as e_close: print(f"Error closing input stream: {e_close}")
            print("INFO: Microphone listening stopped.")


    async def send_realtime_payloads(self):
        # ... (same as before, ensure `types` is not None for types.Blob) ...
        print("INFO: Starting to send realtime payloads to Gemini.")
        try:
            while self._running:
                payload_item = await self.outgoing_payload_queue.get()
                if payload_item is None or not self._running: 
                    self.outgoing_payload_queue.task_done(); break 
                data_to_send_kwargs = {} 
                if "audio" in payload_item and types: 
                    data_to_send_kwargs['audio'] = types.Blob( 
                        data=payload_item["audio"]["data"], mime_type=payload_item["audio"]["mime_type"]
                    )
                if "image" in payload_item and types:
                    data_to_send_kwargs['image'] = types.Blob( 
                        data=base64.b64decode(payload_item["image"]["data"]), mime_type=payload_item["image"]["mime_type"]
                    )
                if data_to_send_kwargs and self.session:
                    try: await self.session.send_realtime_input(**data_to_send_kwargs)
                    except Exception as e:
                        print(f"ERROR sending realtime input to Gemini: {e}"); self._running = False 
                        try_put_sentinel(self.audio_in_queue); break 
                self.outgoing_payload_queue.task_done()
        except asyncio.CancelledError: print("INFO: send_realtime_payloads task cancelled.")
        finally: print("INFO: Realtime payload sending stopped.")


    async def receive_from_gemini(self):
        print("INFO: Listening for responses from Gemini...")
        first_audio_chunk_of_turn = True
        accumulated_stt = ""

        try:
            while self._running and self.session:
                turn_iterator = None
                try:
                    turn_iterator = self.session.receive() 
                    async for response in turn_iterator: 
                        if not self._running: break 
                        
                        is_stt_final_for_this_message = False
                        
                        if hasattr(response, 'server_content') and response.server_content:
                            if hasattr(response.server_content, 'input_transcription') and \
                               response.server_content.input_transcription and \
                               hasattr(response.server_content.input_transcription, 'text'):
                                transcript_part = response.server_content.input_transcription.text.strip()
                                if transcript_part:
                                    print(f"STT (part): {transcript_part}")
                                    accumulated_stt += transcript_part + " " 
                            
                            if hasattr(response.server_content, 'turn_complete') and \
                               response.server_content.turn_complete and accumulated_stt.strip():
                                is_stt_final_for_this_message = True
                                print(f"INFO: Detected turn_complete from Gemini for STT input.")

                        if is_stt_final_for_this_message:
                            final_user_speech = accumulated_stt.strip()
                            accumulated_stt = "" 
                            print(f"INFO: Final User Speech (STT): {final_user_speech}")

                            if final_user_speech:
                                print("INFO: Companion AI generating response using llm_interface...")
                                try:
                                    # 1. Retrieve memory context
                                    profile_facts = await asyncio.to_thread(self.mem.get_all_profile_facts)
                                    latest_summaries = await asyncio.to_thread(self.mem.get_latest_summary, n=2)
                                    latest_insights = await asyncio.to_thread(self.mem.get_latest_insights, n=2)
                                    
                                    memory_context_dict = {
                                        "profile": profile_facts,
                                        "summaries": latest_summaries,
                                        "insights": latest_insights
                                    }

                                    # 2. Generate AI's custom text response
                                    ai_custom_text_response = await asyncio.to_thread(
                                        self.llm_iface.generate_response,
                                        final_user_speech,
                                        memory_context_dict 
                                    )
                                    print(f"INFO: Companion AI Custom Text Response: {ai_custom_text_response}")

                                    # 3. Update memory (summary, facts, insights)
                                    summary_text = await asyncio.to_thread(
                                        self.llm_iface.generate_summary,
                                        final_user_speech,
                                        ai_custom_text_response
                                    )
                                    if summary_text:
                                        await asyncio.to_thread(self.mem.add_summary, summary_text)
                                        print(f"INFO: Added summary: {summary_text[:100]}...") # Print snippet

                                    # No explicit return from your extract_profile_facts, assuming it prints
                                    # If it returns a dict:
                                    extracted_facts = await asyncio.to_thread(
                                         self.llm_iface.extract_profile_facts,
                                         final_user_speech,
                                         ai_custom_text_response
                                    )
                                    if isinstance(extracted_facts, dict):
                                        for key, value in extracted_facts.items():
                                            await asyncio.to_thread(self.mem.upsert_profile_fact, key, value)
                                        if extracted_facts: print(f"INFO: Upserted profile facts: {extracted_facts}")


                                    insight_text = await asyncio.to_thread(
                                        self.llm_iface.generate_insight,
                                        final_user_speech,
                                        ai_custom_text_response,
                                        memory_context_dict # Pass context to insight too
                                    )
                                    if insight_text:
                                        await asyncio.to_thread(self.mem.add_insight, insight_text)
                                        print(f"INFO: Added AI insight: {insight_text[:100]}...") # Print snippet

                                except Exception as e_ai_logic:
                                    print(f"ERROR in custom AI logic or memory update: {e_ai_logic}")
                                    traceback.print_exc()
                                    ai_custom_text_response = "I had a little hiccup. Could you say that again?"
                                
                                # 4. Send custom response to Gemini for TTS
                                if ai_custom_text_response and self.session:
                                    print("INFO: Sending Companion AI response to Gemini for TTS...")
                                    await self.session.send_client_content(
                                        turns=[{"role": "user", "parts": [{"text": ai_custom_text_response}]}],
                                        turn_complete=True 
                                    )
                                    first_audio_chunk_of_turn = True # Expecting new audio stream for TTS
                        
                        # Handle incoming audio data (TTS from Gemini)
                        if first_audio_chunk_of_turn and hasattr(response, 'server_content') and \
                           response.server_content and hasattr(response.server_content, 'model_turn') and \
                           response.server_content.model_turn and \
                           len(response.server_content.model_turn.parts) > 0 and \
                           hasattr(response.server_content.model_turn.parts[0], 'inline_data') and \
                           response.server_content.model_turn.parts[0].inline_data and \
                           hasattr(response.server_content.model_turn.parts[0].inline_data, 'mime_type'):
                            print(f"DEBUG: Gemini TTS audio response MIME type: {response.server_content.model_turn.parts[0].inline_data.mime_type}")
                            first_audio_chunk_of_turn = False 
                        
                        if hasattr(response, 'data') and response.data: 
                            self.audio_in_queue.put_nowait(response.data)
                        
                        if hasattr(response, 'text') and response.text:
                             # This might be an echo of the TTS text, or if we requested TEXT modality too
                            print(f"AI (direct text from Gemini, could be TTS echo): {response.text.strip()}") 

                        if hasattr(response, 'server_content') and \
                           hasattr(response.server_content, 'generation_complete') and \
                           response.server_content.generation_complete:
                            first_audio_chunk_of_turn = True 

                except asyncio.CancelledError: print("INFO: receive_from_gemini task cancelled."); break
                except Exception as e_turn: 
                    print(f"ERROR during Gemini turn processing: {e_turn}"); await asyncio.sleep(0.1) 
        except Exception as e_session: 
            print(f"ERROR in receive_from_gemini session: {e_session}"); self._running = False 
            try_put_sentinel(self.outgoing_payload_queue); try_put_sentinel(self.audio_in_queue)
        finally: print("INFO: Gemini response listening stopped.")

    async def play_audio_output(self):
        # ... (same as before, with detailed PyAudio error logging) ...
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
                    audio_chunk = await self.audio_in_queue.get()
                    if audio_chunk is None or not self._running: 
                        self.audio_in_queue.task_done(); break
                    await asyncio.to_thread(self.audio_stream_out.write, audio_chunk)
                    self.audio_in_queue.task_done() 
                except pyaudio.PaError as pa_err: # type: ignore
                    print(f"ERROR: PyAudio PaError during playback: {pa_err}")
                    host_api_error_info = ""; _=host_api_error_info
                    if hasattr(pa_err, 'hostApiErrorInfo') and pa_err.hostApiErrorInfo:  # type: ignore
                         host_api_error_info = str(pa_err.hostApiErrorInfo) # type: ignore
                    print(f"PyAudio error details: Host API error type: {pa_err.hostApiErrorType if hasattr(pa_err, 'hostApiErrorType') else 'N/A'}, Info: {host_api_error_info}") # type: ignore
                    self._running = False; break 
                except Exception as e: print(f"ERROR during audio playback write: {e}"); break 
        except asyncio.CancelledError: print("INFO: play_audio_output task cancelled.")
        finally:
            if self.audio_stream_out:
                try:
                    if self.audio_stream_out.is_active(): self.audio_stream_out.stop_stream()
                    self.audio_stream_out.close()
                except Exception as e_close: print(f"Error closing output stream: {e_close}")
            print("INFO: Audio playback stopped.")


    async def run_main_loop(self):
        # ... (same task management as before) ...
        print(f"INFO: Connecting to Gemini Live model: {MODEL} with config: {LIVE_CONFIG}")
        all_tasks_set = set()
        try:
            async with client.aio.live.connect(model=MODEL, config=LIVE_CONFIG) as session:
                self.session = session
                print("INFO: Successfully connected to Gemini Live session!")
                
                task_coroutines = [
                    self.send_text_trigger(), self.receive_from_gemini(), self.play_audio_output(),
                    self.listen_microphone(), self.send_realtime_payloads()
                ]
                if self.video_mode == "camera": task_coroutines.append(self.stream_camera_frames())
                elif self.video_mode == "screen": task_coroutines.append(self.stream_screen_frames())

                running_tasks = [asyncio.create_task(coro) for coro in task_coroutines]
                all_tasks_set.update(running_tasks)
                
                if running_tasks:
                    done, pending = await asyncio.wait(running_tasks, return_when=asyncio.FIRST_COMPLETED)
                    self._running = False 
                    for task in done:
                        try: task.result() 
                        except asyncio.CancelledError: print(f"Task (completed group) was cancelled: {task.get_name() if hasattr(task, 'get_name') else 'Unknown'}")
                        except Exception as e: print(f"ERROR in completed task ({task.get_name() if hasattr(task, 'get_name') else 'Unknown'}): {e}")
                    for task in pending: task.cancel()
                    if pending: await asyncio.gather(*pending, return_exceptions=True)
                else: print("Warning: No tasks were started.")
                print("INFO: Main interaction loop finished.")
        except AttributeError as e_attr:
            print(f"FATAL ERROR: AttributeError with Gemini SDK: {e_attr}"); sdk_version = "unknown";_ = sdk_version 
            try: sdk_version = genai.__version__
            except: pass; print(f"Current google-genai version: {sdk_version}")
        except Exception as e_connect:
            print(f"FATAL ERROR: Could not connect to Gemini Live session: {e_connect}"); traceback.print_exc()
        finally:
            self._running = False 
            try_put_sentinel(self.outgoing_payload_queue); try_put_sentinel(self.audio_in_queue)
            if pya: pya.terminate()
            print("INFO: Application cleanup finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini Live API Demo Script")
    parser.add_argument("--mode", type=str, default=DEFAULT_VIDEO_MODE, help='Video mode', choices=["camera", "screen", "none"])
    args = parser.parse_args()

    print(f"Starting Gemini Live API demo with video mode: {args.mode}")
    py_version_str = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Using Python version: {py_version_str}")
    sdk_version_to_print = "unknown"
    try: sdk_version_to_print = genai.__version__; print(f"Using google-genai (SDK) version: {sdk_version_to_print}")
    except AttributeError: print(f"Warning: Could not get google-genai version. SDK might be '{sdk_version_to_print}'.")

    loop_runner = AudioLoop(video_mode=args.mode)
    try: asyncio.run(loop_runner.run_main_loop())
    except KeyboardInterrupt: print("\nINFO: Keyboard interrupt. Exiting.")
    except Exception as e_main_run: print(f"ERROR: Unhandled exception in main: {e_main_run}"); traceback.print_exc()
    finally: print("INFO: Main execution finished.")