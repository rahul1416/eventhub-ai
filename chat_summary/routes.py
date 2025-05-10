from fastapi import APIRouter
import threading

from .models import ChatMessage
from .storage import append_to_history_file, message_counter, event_buffers, summaries
from .summarizer import summarize_event_chat

router = APIRouter()
MESSAGE_THRESHOLD = 1

@router.post("/")
def receive_chat(msg: ChatMessage):
    entry = msg.dict()
    append_to_history_file(entry)
    event_buffers[msg.eventId].append(msg.message)
    message_counter[msg.eventId] += 1

    if message_counter[msg.eventId] >= MESSAGE_THRESHOLD:
        message_counter[msg.eventId] = 0
        threading.Thread(target=summarize_event_chat, args=(msg.eventId,)).start()

    return {"status": "message received", "new_messages": len(event_buffers[msg.eventId])}

@router.get("/summary/{event_id}")
def get_summary(event_id: str):
    return {"eventId": event_id, "summary": summaries.get(event_id, "Summary not available yet.")}
