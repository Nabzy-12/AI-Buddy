# companion_ai/memory.py
import sqlite3
import os
from datetime import datetime

# Define the path for the database within the 'data' directory
# Assumes this script is run from the root 'Project_Companion_AI' folder context
# Or that the 'data' folder is relative to where the script using this module is located.
# For robustness, especially with Chainlit, let's define it relative to this file's location.
MODULE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(MODULE_DIR, '..')) # Go up one level from companion_ai
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
DB_PATH = os.path.join(DATA_DIR, 'companion_ai.db')

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    # Use Row factory for dictionary-like access to columns
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database and creates tables if they don't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # User Profile Table (Key-Value Store)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_profile (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Conversation Summaries Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            summary_text TEXT NOT NULL
        )
    ''')

    # AI Insights Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ai_insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            insight_text TEXT NOT NULL
        )
    ''')

    conn.commit()
    conn.close()
    print("Database initialized successfully.") # Optional: for confirmation

# --- User Profile Functions ---

def upsert_profile_fact(key: str, value: str):
    """Adds or updates a user profile fact."""
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = datetime.now()
    # Use INSERT OR REPLACE (or ON CONFLICT) for upsert behavior
    cursor.execute('''
        INSERT INTO user_profile (key, value, last_updated)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value = excluded.value,
            last_updated = excluded.last_updated;
    ''', (key, value, timestamp))
    conn.commit()
    conn.close()

def get_profile_fact(key: str) -> str | None:
    """Retrieves a specific user profile fact by key."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM user_profile WHERE key = ?", (key,))
    result = cursor.fetchone()
    conn.close()
    return result['value'] if result else None

def get_all_profile_facts() -> dict:
    """Retrieves all user profile facts as a dictionary."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT key, value FROM user_profile ORDER BY last_updated DESC")
    results = cursor.fetchall()
    conn.close()
    # Convert list of Row objects to a dictionary
    return {row['key']: row['value'] for row in results}


# --- Conversation Summary Functions ---

def add_summary(summary_text: str):
    """Adds a new conversation summary."""
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = datetime.now()
    cursor.execute("INSERT INTO conversation_summaries (timestamp, summary_text) VALUES (?, ?)",
                   (timestamp, summary_text))
    conn.commit()
    conn.close()

def get_latest_summary(n: int = 1) -> list[dict]:
    """Retrieves the latest N conversation summaries."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, summary_text FROM conversation_summaries ORDER BY timestamp DESC LIMIT ?", (n,))
    results = cursor.fetchall()
    conn.close()
    # Convert Row objects to dictionaries for easier use
    return [dict(row) for row in results]

# --- AI Insight Functions ---

def add_insight(insight_text: str):
    """Adds a new AI insight."""
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = datetime.now()
    cursor.execute("INSERT INTO ai_insights (timestamp, insight_text) VALUES (?, ?)",
                   (timestamp, insight_text))
    conn.commit()
    conn.close()

def get_latest_insights(n: int = 1) -> list[dict]:
    """Retrieves the latest N AI insights."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, insight_text FROM ai_insights ORDER BY timestamp DESC LIMIT ?", (n,))
    results = cursor.fetchall()
    conn.close()
    # Convert Row objects to dictionaries
    return [dict(row) for row in results]


# --- Initialization Call ---
# You might run this once manually or ensure it's called at application start
if __name__ == "__main__":
    # This block runs only when the script is executed directly
    # Useful for initial setup or testing
    print(f"Database path: {DB_PATH}")
    init_db()

    # Example Usage (for testing if run directly)
    upsert_profile_fact("user_name", "Alex")
    upsert_profile_fact("likes", "dogs, coding")
    print("Profile Facts:", get_all_profile_facts())
    add_summary("User discussed their project idea for a companion AI.")
    print("Latest Summary:", get_latest_summary())
    add_insight("User is enthusiastic about the project. Memory system is key.")
    print("Latest Insight:", get_latest_insights())