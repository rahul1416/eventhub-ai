from fastapi import FastAPI, Body
from pydantic import BaseModel
from collections import defaultdict
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import threading
import json
import os
from dotenv import load_dotenv
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import uvicorn

# === Load environment variables ===
load_dotenv()

# === Initialize FastAPI ===
app = FastAPI()

# === LLM setup (LangChain + Groq) ===
llm = ChatGroq(temperature=0.3, model_name="llama3-70b-8192")
prompt = PromptTemplate(
    input_variables=["chat"],
    template="""
You are an assistant summarizing a chat event. Write a 2–3 paragraph summary capturing main topics, names, and insights.

Chat:
{chat}

Summary:
"""
)
chain = LLMChain(llm=llm, prompt=prompt)

# === Chat data and summary logic ===
MESSAGE_THRESHOLD = 1
message_counter = defaultdict(int)
event_buffers = defaultdict(list)
summaries = {}
CHAT_HISTORY_FILE = "chathistory.json"

try:
    with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
        chat_data = json.load(f)
except FileNotFoundError:
    chat_data = []
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_data, f)

class ChatMessage(BaseModel):
    eventId: str
    userId: str
    message: str
    timestamp: str

def append_to_history_file(entry):
    chat_data.append(entry)
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_data, f, indent=2)

def summarize_event_chat(event_id: str):
    messages = [entry["message"] for entry in chat_data if entry["eventId"] == event_id]
    combined = "\n".join(messages)
    summary = chain.run(chat=combined)
    summaries[event_id] = summary
    print(f"[Summary Updated] eventId={event_id}")

@app.post("/chat")
def receive_chat(msg: ChatMessage):
    entry = msg.dict()
    append_to_history_file(entry)
    event_buffers[msg.eventId].append(msg.message)
    message_counter[msg.eventId] += 1
    if message_counter[msg.eventId] >= MESSAGE_THRESHOLD:
        message_counter[msg.eventId] = 0
        threading.Thread(target=summarize_event_chat, args=(msg.eventId,)).start()
    return {"status": "message received", "new_messages": len(event_buffers[msg.eventId])}

@app.get("/summary/{event_id}")
def get_summary(event_id: str):
    return {"eventId": event_id, "summary": summaries.get(event_id, "Summary not available yet.")}


# === Google GenAI Client for Research Endpoint ===
try:
    client = genai.Client()
    model_id = "gemini-2.0-flash"
except Exception as e:
    print(f"Error initializing Google GenAI Client: {e}")
    client = None
    model_id = None

class ResearchRequest(BaseModel):
    event_name: str
    artist_name: str
    ngo_name: str | None = None

class ResearchResponse(BaseModel):
    generated_text: str | None = None
    grounding_metadata_rendered_content: str | None = None
    error: str | None = None

@app.post("/research", response_model=ResearchResponse)
async def research_event(request: ResearchRequest):
    if not client or not model_id:
        return ResearchResponse(error="Google GenAI Client not initialized. Check API key and environment variables.")

    event_name = request.event_name
    artist_name = request.artist_name
    ngo_name = request.ngo_name

    query_parts = [
        f"Provide a comprehensive description for the event '{event_name}' featuring '{artist_name}'. "
        "Include details about the event's theme, purpose, and what attendees can expect. "
        f"Also, provide background information on '{artist_name}', treating this as potentially an individual artist, group, or crew. "
        "Include their notable work, style, and artistic background. "
    ]
    if ngo_name:
        query_parts.append(
            f"This event is organized by or benefits the NGO '{ngo_name}'. "
            f"Detail '{ngo_name}'s mission, impactful projects, and how they utilize funds to create positive change. "
            f"Explain how '{event_name}' directly supports '{ngo_name}'s specific objectives or causes. "
            f"If '{artist_name}' has any connection to '{ngo_name}' (such as being comprised of NGO staff, volunteers, or previous collaborators), "
            "highlight this relationship and how it strengthens the event's purpose. "
            "Emphasize the tangible impact attendees will make by supporting this charitable initiative. "
        )
    else:
        query_parts.append(
            "If this is a recurring event, mention any past successes or highlights. "
            "Focus on creating an engaging overview for potential attendees. "
        )

    query = "".join(query_parts)
    google_search_tool = Tool(google_search=GoogleSearch())

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=query,
            config=GenerateContentConfig(tools=[google_search_tool])
        )

        generated_text_parts = []
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                generated_text_parts.append(part.text)

        generated_text = "\n".join(generated_text_parts) if generated_text_parts else "No content parts found in the response."

        return ResearchResponse(generated_text=generated_text)

    except Exception as e:
        print(f"Error during GenAI call: {e}")
        return ResearchResponse(error=f"An error occurred: {str(e)}")

# === Start Server ===
if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY environment variable not set. Please set it in your .env file or environment.")
    else:
        print("Starting Uvicorn server on http://127.0.0.1:8000")
        print("API documentation available at http://127.0.0.1:8000/docs")
        uvicorn.run(app, host="127.0.0.1", port=8000)
