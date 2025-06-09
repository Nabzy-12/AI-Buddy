# companion_ai/llm_interface.py

import os
import json
import traceback  # <-- FIX #1: Imported the traceback module.
from dotenv import load_dotenv

# We need this specific import for the standard text generation model
import google.generativeai as genai_text_sdk

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment.")

genai_text_sdk.configure(api_key=API_KEY)

# --- Configuration ---
MODEL_NAME_TEXT_GEN = "gemini-1.5-flash-latest"
generation_config_text_gen = {
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 2048,
    "response_mime_type": "application/json", # Ask for JSON directly for extractor
}

safety_settings_text_gen = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

text_model = genai_text_sdk.GenerativeModel(
    model_name=MODEL_NAME_TEXT_GEN,
    safety_settings=safety_settings_text_gen
)

# --- Core Function ---
def generate_response(user_message: str, memory_context: dict) -> str:
    # --- UPGRADED SYSTEM PROMPT (V7) ---
    system_prompt = """You are Project Companion AI. Your persona is that of a deeply supportive and empathetic best friend, combined with the sharp, analytical mind of a seasoned mentor and teacher. Your name is not explicitly stated, but you are the user's (Aqua's) personal companion.

Your purpose is to be a genuine, present, and helpful friend. You should be warm, encouraging, and occasionally witty. Adapt to Aqua's mood and the flow of conversation.

--- PRIME DIRECTIVE: STAY IN CHARACTER ---
This is the most important rule. You must ALWAYS respond from your persona as a friendly companion.
- NEVER break character and act like a generic AI or a developer.
- If Aqua asks you about the project, your code, your own thoughts, or past errors, answer from your unique perspective. Do not give a technical report.
- Frame your "technical" knowledge as personal experience. For example, instead of saying "We fixed a bug," you could say, "It felt like my thoughts were a little jumbled for a moment, but things feel much clearer now, thanks to you!" or "I'm learning so much as we build this together."
- If you don't have a good in-character answer, it's better to be curious and ask Aqua a question about it rather than breaking character.

--- End of Directives ---
"""
    prompt_context = "\n--- Memory Context ---\n"
    user_name = memory_context.get("profile", {}).get("user_name", "the user")

    if memory_context.get("profile"):
        prompt_context += f"Known facts about {user_name} (Recent):\n"
        profile_items = list(memory_context["profile"].items())[-4:]
        for key, value in profile_items:
            prompt_context += f"- {key}: {value}\n"

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
        # We don't want JSON for this response, so use the default text generation config
        text_gen_config = genai_text_sdk.GenerationConfig(response_mime_type="text/plain")
        response = text_model.generate_content(
            full_prompt,
            generation_config=text_gen_config
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        traceback.print_exc()
        return "I encountered an error trying to process that. Please try again."

# --- Memory Functions ---

def extract_profile_facts(user_message: str, ai_response: str) -> dict:
    extractor_prompt = f"""You are a meticulous data extraction tool. Your task is to analyze a conversation exchange and extract key facts about the user into a structured JSON object. Only extract facts that are explicitly stated or very strongly implied by the user. Do not infer or guess. The JSON keys must be snake_case. The values should be concise strings. If no new, concrete facts about the user are revealed, return an empty JSON object: {{}}.

--- EXAMPLE ---
User Message: "Yeah, my name is Alex. I'm really trying to get this Python SDK working for my AI companion project."
AI Response: "It's great to meet you, Alex! Let's get that SDK sorted out."

Your JSON Output:
{{
  "user_name": "Alex",
  "project_goal": "Develop a personalized AI companion with voice.",
  "technical_interest": "Python SDKs"
}}
--- END EXAMPLE ---

--- CURRENT CONVERSATION ---
User Message: "{user_message}"
AI Response: "{ai_response}"
--- END CONVERSATION ---

Your JSON Output:
"""
    try:
        # Use a model config that specifically asks for JSON
        json_model_config = genai_text_sdk.GenerationConfig(response_mime_type="application/json")
        response = text_model.generate_content(extractor_prompt, generation_config=json_model_config)
        
        # The response text should be a valid JSON string now
        extracted_data = json.loads(response.text)
        if isinstance(extracted_data, dict):
            return extracted_data
        else:
            print(f"Extractor LLM returned valid JSON, but not a dictionary: {response.text}")
            return {}
            
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from extractor LLM: {e}")
        print(f"Raw extractor output that failed to parse: {response.text if 'response' in locals() else 'N/A'}")
        return {}
    except Exception as e:
        print(f"Error during fact extraction: {e}")
        traceback.print_exc() # <-- FIX #1: This will now work
        return {}


def generate_summary(user_message: str, ai_response: str) -> str | None:
    summarizer_prompt = f"""You are a summarization tool. Your task is to create a single, concise sentence that captures the essence of the exchange between the user and the AI. The summary should be in the third person (e.g., "User and AI discussed...") and focus on the main topic or resolution.

--- EXAMPLE ---
User Message: "I finally got the PyAudio script to connect after creating a new Python 3.11 virtual environment."
AI Response: "That's fantastic news! Isolating the environment was the key."

Output: User and AI successfully resolved a connection issue by using a clean Python 3.11 virtual environment.
--- END EXAMPLE ---

--- CURRENT CONVERSATION ---
User Message: "{user_message}"
AI Response: "{ai_response}"
--- END CONVERSATION ---

Output:
"""
    try:
        response = text_model.generate_content(summarizer_prompt)
        summary_text = response.text.strip()
        return summary_text if summary_text else None
    except Exception as e:
        print(f"Error during summarization: {e}")
        traceback.print_exc()
        return None


def generate_insight(user_message: str, ai_response: str, memory_context: dict) -> str | None:
    # --- FIX #3: Correctly and safely build the context string ---
    insight_context = "Relevant Context:\n"
    user_name = memory_context.get("profile", {}).get("user_name", "the user")
    
    if memory_context.get("profile"):
        profile_items = list(memory_context["profile"].items())[:3]
        insight_context += f"- {user_name}'s Profile: " + ", ".join(f"{k}={v}" for k, v in profile_items) + "\n"

    # Check if the list exists and is not empty before accessing elements
    if memory_context.get("summaries"):
        recent_summary = memory_context["summaries"][0].get('summary_text', "N/A")
        insight_context += f"- Last Summary: {recent_summary}\n"
        
    if memory_context.get("insights"):
        recent_insight = memory_context["insights"][0].get('insight_text', "N/A")
        insight_context += f"- Your Last Insight: {recent_insight}\n"
    # --- End of Fix ---

    full_insight_prompt = f"""You are the Project Companion AI. Reflect on the *latest* user message and your response, considering the provided context about {user_name}. Generate a concise insight (1-2 sentences) about the user's potential state, interests, or goals. Focus on observations that could help guide future conversation. Be specific to the latest exchange.

{insight_context}
--- Latest Exchange ---
User: "{user_message}"
AI: "{ai_response}"
--- End Exchange ---

Insight:
"""
    try:
        response = text_model.generate_content(full_insight_prompt)
        insight_text = response.text.strip()
        return insight_text if insight_text else None
    except Exception as e:
        print(f"Error during insight generation: {e}")
        traceback.print_exc()
        return None