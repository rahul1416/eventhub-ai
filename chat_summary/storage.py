import json
from collections import defaultdict

CHAT_HISTORY_FILE = "chathistory.json"

try:
    with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
        chat_data = json.load(f)
except FileNotFoundError:
    chat_data = []
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_data, f)

message_counter = defaultdict(int)
event_buffers = defaultdict(list)
summaries = {}

def append_to_history_file(entry: dict):
    chat_data.append(entry)
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_data, f, indent=2)

def get_messages_for_event(event_id: str):
    return [entry["message"] for entry in chat_data if entry["eventId"] == event_id]
