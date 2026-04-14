import json
import os
import time
from datetime import datetime
from pathlib import Path

# Define the log file path
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
INTERACTION_LOG_FILE = LOG_DIR / "interaction_log.jsonl"

def log_interaction(user_id, book_id, event_type, value=None, metadata=None):
    """
    Logs user interactions with books to a JSONL file for future model training.

    Args:
        user_id (int): ID of the user.
        book_id (str/int): ID of the book (Google API ID or local DB ID).
        event_type (str): Type of event ('view', 'click', 'rate', 'add_to_library').
        value (float, optional): Numeric value associated with the event (e.g., rating out of 5).
        metadata (dict, optional): Additional contextual data.
    """
    # Create the log entry
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "timestamp_unix": int(time.time()),
        "user_id": user_id,
        "book_id": str(book_id),
        "event_type": event_type,
    }

    if value is not None:
        entry["value"] = value

    if metadata is not None:
        entry["metadata"] = metadata

    # Append to JSONL file
    try:
        with open(INTERACTION_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"Failed to log interaction: {e}")
