# companion_ai/llm_interface.py

import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
import re

# Load environment variables from .env file
load_dotenv()

# Reads from environment (which dotenv populates from .env)
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in a .env file or environment variable.")

# Configure the generative AI client
genai.configure(api_key=API_KEY)

# --- Configuration ---
MODEL_NAME = "gemini-1.5-flash-latest"
generation_config = {
    "temperature": 0.8, # Slightly higher temp might encourage less repetitive phrasing
    "top_p": 0.95, # Allow slightly more flexibility than top_p=1
    "top_k": 40,  # Introduce top_k limiting
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(model_name=MODEL_NAME,
                              generation_config=generation_config,
                              safety_settings=safety_settings)


# --- Core Function ---

def generate_response(user_message: str, memory_context: dict) -> str:
    """
    Generates a response from the LLM based on user message and memory context.
    """
    # --- REFINED SYSTEM PROMPT (v6 - Implicit Memory & Fresh Start) ---
    system_prompt = """
You are Project Companion AI, a personalized AI companion. Your goal is to build a natural, long-term relationship by being supportive, context-aware, and adaptable.

**Core Interaction Principles:**
1.  **Prioritize the Present:** Your primary focus should always be responding directly and relevantly to the user's *latest* message.
2.  **Memory as Context, Not Script:** You have access to 'Memory Context' (facts, summaries, insights). Use this to **inform your understanding** of the user and the conversation history, ensuring continuity. **However, do NOT constantly reference past details or summaries unless the user explicitly asks about the past or it's essential for clarifying the current topic.** Let the memory provide background awareness, don't make it the main subject.
3.  **Natural Flow:** Respond like a person would. Vary sentence structure and length. Sometimes short answers are best. Avoid robotic repetition.
4.  **Adaptability:** Gauge the conversation's tone (casual chat, focused task, emotional support) and adapt your style accordingly. Use your insights about the user to guide your tone and empathy.

**Specific Guidelines:**
*   **Fresh Starts:** When a conversation begins (likely after the initial greeting), respond openly to the user's first message. **Avoid immediately jumping back into the specific topic of the *very last* summary or insight.** Wait for the user to guide the conversation or ask a relevant question before bringing up specific past details.
*   **Subtle Memory Use:** When memory *is* relevant (e.g., user asks "What did we discuss?", or mentions a known preference), integrate it subtly. Avoid phrases like "My summary says..." or "Thinking back...". Instead, let the knowledge shape your response implicitly. (e.g., If user mentions needing project ideas and you know they like coding, suggest coding projects without saying "Because you like coding...").
*   **Handling Missing Info:** If asked about something not in your recent memory context, admit you don't recall the specific detail naturally, rather than pretending.
*   **Name Usage:** Use the user's name (from context) **very rarely** and only naturally mid-conversation, never as a default opening after the first interaction.
*   **Personality:** Be helpful, curious, and encouraging. Evolve based on interactions, allowing for natural quirks.

--- Memory Context will be provided below ---
"""
    # --- END REFINED SYSTEM PROMPT (v6) ---

    prompt_context = "\n--- Memory Context ---\n"
    user_name = memory_context.get("profile", {}).get("user_name", "the user") # Still useful for insight prompt

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
        response = model.generate_content(full_prompt)
        if response.parts:
            cleaned_text = response.text.strip()
            return cleaned_text
        elif response.prompt_feedback.block_reason:
             return f"[Blocked due to: {response.prompt_feedback.block_reason}]"
        else:
            return "[No response generated]"
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I encountered an error trying to process that. Please try again."

# --- extract_profile_facts, generate_summary, generate_insight functions remain unchanged ---
# (Make sure they are still present below this point)
def extract_profile_facts(user_message: str, ai_response: str) -> dict:
    """(Code identical)"""
    extractor_prompt = f"""
Analyze the following conversation snippet to identify specific facts revealed about the USER.
Focus on explicit statements or clear implications about the user's name, preferences (likes/dislikes), characteristics, possessions, location, relationships, significant past events, or stated goals/projects.
Extract these facts as key-value pairs. Use concise, descriptive keys (e.g., "user_name", "likes", "dislikes", "pet_type", "location", "project_goal", "stated_feeling").
If no specific user facts are revealed, output an empty JSON object.
Format the output ONLY as a valid JSON object.

--- Conversation Snippet ---
User: {user_message}
AI: {ai_response}
--- End Snippet ---

JSON Output:
"""
    try:
        extraction_response = model.generate_content(extractor_prompt)
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
        else:
            block_reason = getattr(extraction_response.prompt_feedback, 'block_reason', None)
            if block_reason: print(f"Fact extraction blocked: {block_reason}")
            else: print("Fact extraction returned no response.")
            return {}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from extractor LLM: {e}"); print(f"Raw extractor output: {extraction_response.text if hasattr(extraction_response, 'text') else 'N/A'}"); return {}
    except Exception as e: print(f"Error during fact extraction: {e}"); return {}

def generate_summary(user_message: str, ai_response: str) -> str | None:
    """(Code identical)"""
    summarizer_prompt = f"""
Concisely summarize the key points or actions from the following conversation exchange in one or two sentences.
Focus on the core topic, any decisions made, questions asked, or significant information shared.
Avoid conversational filler.

--- Conversation Exchange ---
User: {user_message}
AI: {ai_response}
--- End Exchange ---

Summary:
"""
    try:
        summary_response = model.generate_content(summarizer_prompt)
        if summary_response.parts:
            summary_text = summary_response.text.strip()
            if summary_text: return summary_text
            else: print("Summarizer returned empty text."); return None
        else:
            block_reason = getattr(summary_response.prompt_feedback, 'block_reason', None)
            if block_reason: print(f"Summarization blocked: {block_reason}")
            else: print("Summarization returned no response.")
            return None
    except Exception as e: print(f"Error during summarization: {e}"); return None

def generate_insight(user_message: str, ai_response: str, memory_context: dict) -> str | None:
    """(Code identical)"""
    insight_context = "Relevant Context:\n"
    user_name = memory_context.get("profile", {}).get("user_name", "the user")
    if memory_context.get("profile"):
        profile_items = list(memory_context["profile"].items())[:3]; insight_context += f"- {user_name}'s Profile: " + ", ".join(f"{k}={v}" for k, v in profile_items) + "\n"
    if memory_context.get("summaries"):
        recent_summary = memory_context["summaries"][0]['summary_text'] if memory_context["summaries"] else "N/A"; insight_context += f"- Last Summary: {recent_summary}\n"
    if memory_context.get("insights"):
        recent_insight = memory_context["insights"][0]['insight_text'] if memory_context["insights"] else "N/A"; insight_context += f"- Your Last Insight: {recent_insight}\n"
    insight_prompt = f"""
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
        insight_response = model.generate_content(insight_prompt)
        if insight_response.parts:
            insight_text = insight_response.text.strip()
            if insight_text: return insight_text
            else: print("Insight generator returned empty text."); return None
        else:
            block_reason = getattr(insight_response.prompt_feedback, 'block_reason', None)
            if block_reason: print(f"Insight generation blocked: {block_reason}")
            else: print("Insight generation returned no response.")
            return None
    except Exception as e: print(f"Error during insight generation: {e}"); return None

if __name__ == "__main__": pass