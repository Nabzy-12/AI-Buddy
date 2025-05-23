# Ultra-minimal main.py for testing
import chainlit as cl
from io import BytesIO # Still need this if on_audio_chunk/end expect the session var

@cl.on_chat_start
async def start():
    cl.user_session.set("audio_buffer", BytesIO()) # Keep session var init for safety
    cl.user_session.set("audio_duration_ms", 0)  # Keep session var init
    await cl.Message(content="Minimal test. Check console for MIME warnings.").send()

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    pass # Do nothing

@cl.on_audio_end
async def on_audio_end(elements: list):
    pass # Do nothing