from pydantic import BaseModel

class ResearchRequest(BaseModel):
    event_name: str
    artist_name: str
    ngo_name: str | None = None

class ResearchResponse(BaseModel):
    generated_text: str | None = None
    grounding_metadata_rendered_content: str | None = None
    error: str | None = None
