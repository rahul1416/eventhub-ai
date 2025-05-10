from fastapi import APIRouter
from .models import ResearchRequest, ResearchResponse
import os
from dotenv import load_dotenv
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

load_dotenv()
router = APIRouter()

try:
    client = genai.Client()
    model_id = "gemini-2.0-flash"
except Exception as e:
    print(f"Error initializing Google GenAI Client: {e}")
    client = None
    model_id = None

@router.post("/", response_model=ResearchResponse)
async def research_event(request: ResearchRequest):
    if not client or not model_id:
        return ResearchResponse(error="Google GenAI Client not initialized.")

    query_parts = [
        f"Provide a comprehensive description for the event '{request.event_name}' featuring '{request.artist_name}'. ",
        "Include details about the event's theme, purpose, and what attendees can expect. ",
        f"Also, provide background information on '{request.artist_name}', treating this as potentially an individual artist, group, or crew. "
    ]
    if request.ngo_name:
        query_parts.append(
            f"This event is organized by or benefits the NGO '{request.ngo_name}'. "
            "Explain the NGO's mission and how the event supports it. "
        )
    else:
        query_parts.append("If this is a recurring event, mention past highlights. ")

    query = "".join(query_parts)
    google_search_tool = Tool(google_search=GoogleSearch())

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=query,
            config=GenerateContentConfig(tools=[google_search_tool])
        )

        parts = response.candidates[0].content.parts if response.candidates else []
        generated_text = "\n".join(part.text for part in parts if hasattr(part, "text"))

        return ResearchResponse(generated_text=generated_text)

    except Exception as e:
        return ResearchResponse(error=f"An error occurred: {str(e)}")
