from pydantic import BaseModel

class ChatMessage(BaseModel):
    eventId: str
    userId: str
    message: str
    timestamp: str
