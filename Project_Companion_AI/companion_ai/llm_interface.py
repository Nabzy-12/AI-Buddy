# companion_ai/llm_interface.py

from google import genai # Use this as per the new SDK 'google-genai' and cookbook
import os
from dotenv import load_dotenv
import json
import re

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file or environment variable.")

# For google-genai, the API key is typically handled by the Client 
# or by ensuring GOOGLE_API_KEY is in the environment.
# We don't need genai.configure() if using genai.Client() properly.

# --- SDK Client Initialization (as per cookbook for Live API, may apply more broadly) ---
try:
    # This client will be used for model interactions if GenerativeModel is not top-level on 'genai'
    sdk_client = genai.Client(api_key=API_KEY) # Pass API key directly for clarity
    # Or, if the Client doesn't take api_key and relies on env var:
    # sdk_client = genai.Client() 
except AttributeError:
    print("ERROR (llm_interface.py): genai.Client() failed. 'google-genai' library may not be installed correctly or version issue.")
    raise
except Exception as e:
    print(f"ERROR (llm_interface.py): Unexpected error initializing genai.Client(): {e}")
    raise


# --- Configuration ---
MODEL_NAME_TEXT_GEN = "gemini-1.5-flash-latest" # For text generation
generation_config_text_gen = {
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 2048,
}

safety_settings_text_gen = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# --- Model Initialization for Text Generation ---
# If GenerativeModel is not on 'genai' directly, it might be on the 'sdk_client'
# or we use the client to make requests.
# Let's assume for now we might need to get the model through the client,
# OR the client itself has a generate_content method.

# Option A: Try to get a model through the client (speculative if not in docs)
# if hasattr(sdk_client, 'get_model'):
#     text_model = sdk_client.get_model(MODEL_NAME_TEXT_GEN) # This is a guess
# else:
#     # Option B: Assume text_model still comes from top-level genai, but this failed before.
#     # This line caused the AttributeError: module 'google.genai' has no attribute 'GenerativeModel'
#     # So, we MUST NOT use genai.GenerativeModel directly if 'from google import genai' is used.
#     # text_model = genai.GenerativeModel(...) # THIS IS WRONG with 'from google import genai'

# Option C: The client itself might be used for requests, or there's another way.
# For many Google SDKs, after creating a client, you use methods on that client.
# However, the Gemini Python SDK often uses genai.GenerativeModel("model-name") AFTER configuration.
# The issue is that 'from google import genai' gives 'genai' which might not be the same
# 'genai' as 'import google.generativeai as genai'.

# THIS IS THE CORE PROBLEM: How to get a text generation model instance
# using the `google-genai` SDK (when imported as `from google import genai`)

# Let's go back to the most standard way for `google-generativeai` (even if cookbook uses different import for live client)
# because `llm_interface.py` is for general text gen.
# We will use `import google.generativeai as genai_text_sdk` for THIS FILE.
# This means this file will use the `google-generativeai` package,
# while `main.py` (for Live API) uses `google-genai`. This is messy but might be necessary
# if their APIs for text-gen vs live-api are best accessed via different top-level modules/clients.

# --- Let's try a specific import for text generation in this file ---
import google.generativeai as genai_text_sdk # Use a different alias for clarity

genai_text_sdk.configure(api_key=API_KEY) # Configure this specific SDK import

text_model = genai_text_sdk.GenerativeModel(
                                model_name=MODEL_NAME_TEXT_GEN,
                                generation_config=generation_config_text_gen,
                                safety_settings=safety_settings_text_gen
                             )
# --- End Model Initialization ---


# --- Core Function ---
def generate_response(user_message: str, memory_context: dict) -> str:
    # ... (system_prompt and prompt_context building is THE SAME as your version) ...
    system_prompt = """...""" # Keep your v6 prompt
    prompt_context = "\n--- Memory Context ---\n"
    user_name = memory_context.get("profile", {}).get("user_name", "the user") 

    if memory_context.get("profile"):
        prompt_context += f"Known facts about {user_name} (Recent):\n"
        profile_items = list(memory_context["profile"].items())[-4:]
        for key, value in profile_items: prompt_context += f"- {key}: {value}\n"
        if 'user_name' in memory_context["profile"] and 'user_name' not in dict(profile_items): prompt_context += f"- user_name: {memory_context['profile']['user_name']}\n"

    if memory_context.get("summaries"):
        prompt_context += "\nRecent conversation summaries (Last 2):\n"
        for summary in memory_context["summaries"][:2]:
            ts = summary.get('timestamp', 'N/A')
            prompt_context += f"- [{ts}] {summary['summary_text']}\n"

    if memory_context.get("insights"):
        prompt_context += "\nYour Recent AI insights (Last 2):\n"
        for insight in memory_context["insights"][:2]:
            ts = insight.get('timestamp', 'N/A')
            prompt_context += f"- [{ts}] {insight['insight_text']}\n"
    prompt_context += "--- End Memory Context ---\n"

    full_prompt = f"{system_prompt}\n{prompt_context}\nUser: {user_message}\nAI:"

    try:
        # Use the 'text_model' initialized with 'google.generativeai'
        response = text_model.generate_content(full_prompt) 
        if response.parts:
            cleaned_text = response.text.strip()
            return cleaned_text
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
             return f"[Blocked due to: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason.name}]"
        else:
            return "[No response generated or unexpected response structure]"
    except Exception as e:
        print(f"Error generating response: {e}")
        traceback.print_exc() # Print full traceback
        return "I encountered an error trying to process that. Please try again."

# --- extract_profile_facts, generate_summary, generate_insight functions ---
# These should also use 'text_model' (from 'google.generativeai')
def extract_profile_facts(user_message: str, ai_response: str) -> dict:
    # ... (your existing logic, but ensure it uses 'text_model.generate_content(...)') ...
    # Example:
    # extraction_response = text_model.generate_content(extractor_prompt)
    # (rest of your parsing logic)
    extractor_prompt = f"""...""" # Your full prompt
    try:
        extraction_response = text_model.generate_content(extractor_prompt)
        # ... (your existing JSON parsing logic from your llm_interface.py) ...
        if extraction_response.parts:
            raw_json = extraction_response.text.strip()
            if raw_json.startswith("```json"): raw_json = raw_json[7:]
            if raw_json.endswith("```"): raw_json = raw_json[:-3]
            raw_json = raw_json.strip()
            if raw_json.startswith('{') and raw_json.endswith('}'):
                extracted_data = json.loads(raw_json)
                if isinstance(extracted_data, dict): return extracted_data
                else: print(f"Extractor LLM returned valid JSON, but not a dictionary: {raw_json}"); return {}
            else: print(f"Extractor LLM did not return valid JSON: {raw_json}"); return {}
        else: # Handle blocked or empty responses
            block_reason = getattr(getattr(extraction_response, 'prompt_feedback', None), 'block_reason', None)
            if block_reason: print(f"Fact extraction blocked: {block_reason}")
            else: print("Fact extraction returned no response parts.")
            return {}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from extractor LLM: {e}"); 
        if hasattr(extraction_response, 'text'): print(f"Raw extractor output: {extraction_response.text}")
        return {}
    except Exception as e: print(f"Error during fact extraction: {e}"); traceback.print_exc(); return {}


def generate_summary(user_message: str, ai_response: str) -> str | None:
    # ... (your existing logic, but ensure it uses 'text_model.generate_content(...)') ...
    summarizer_prompt = f"""...""" # Your full prompt
    try:
        summary_response = text_model.generate_content(summarizer_prompt)
        # ... (your existing parsing logic from your llm_interface.py) ...
        if summary_response.parts:
            summary_text = summary_response.text.strip()
            if summary_text: return summary_text
            else: print("Summarizer returned empty text."); return None
        else:
            block_reason = getattr(getattr(summary_response, 'prompt_feedback', None), 'block_reason', None)
            if block_reason: print(f"Summarization blocked: {block_reason}")
            else: print("Summarization returned no response parts.")
            return None
    except Exception as e: print(f"Error during summarization: {e}"); traceback.print_exc(); return None

def generate_insight(user_message: str, ai_response: str, memory_context: dict) -> str | None:
    # ... (your existing logic, but ensure it uses 'text_model.generate_content(...)') ...
    insight_prompt = f"""...""" # Your full prompt (including context building)
    insight_context = "Relevant Context:\n" # Rebuild as per your original
    user_name = memory_context.get("profile", {}).get("user_name", "the user")
    if memory_context.get("profile"):
        profile_items = list(memory_context["profile"].items())[:3]; insight_context += f"- {user_name}'s Profile: " + ", ".join(f"{k}={v}" for k, v in profile_items) + "\n"
    if memory_context.get("summaries"):
        recent_summary = memory_context["summaries"][0]['summary_text'] if memory_context["summaries"] else "N/A"; insight_context += f"- Last Summary: {recent_summary}\n"
    if memory_context.get("insights"):
        recent_insight = memory_context["insights"][0]['insight_text'] if memory_context["insights"] else "N/A"; insight_context += f"- Your Last Insight: {recent_insight}\n"
    
    # Reconstruct the full insight prompt using your original structure
    full_insight_prompt = f"""
You are the Project Companion AI. Reflect on the *latest* user message and your response, considering the provided context about {user_name}.
Generate a concise insight (1-2 sentences) about the user's potential state, interests, goals, or the nature of the interaction itself.
Focus on observations that could help deepen the relationship or guide future conversation. Examples:
- "User seems enthusiastic about [topic]."
- "Detected a potential interest in [area] based on their question."
- "User shared a personal goal related to [goal]."
- "The conversation took a more technical/personal/reflective turn."
- "AI could follow up on [topic mentioned by user] next time."
- "User might be feeling [emotion, e.g., curious, stressed] based on their message."
Avoid generic statements. Be specific to the latest exchange.

{insight_context}
--- Latest Exchange ---
User: {user_message}
AI: {ai_response}
--- End Exchange ---

Insight:
"""
    try:
        insight_response = text_model.generate_content(full_insight_prompt)
        # ... (your existing parsing logic from your llm_interface.py) ...
        if insight_response.parts:
            insight_text = insight_response.text.strip()
            if insight_text: return insight_text
            else: print("Insight generator returned empty text."); return None
        else:
            block_reason = getattr(getattr(insight_response, 'prompt_feedback', None), 'block_reason', None)
            if block_reason: print(f"Insight generation blocked: {block_reason}")
            else: print("Insight generation returned no response parts.")
            return None
    except Exception as e: print(f"Error during insight generation: {e}"); traceback.print_exc(); return None

# if __name__ == "__main__": pass # Keep this or remove if not needed