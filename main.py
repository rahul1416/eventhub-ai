from fastapi import FastAPI
from chat_summary.routes import router as chat_router
from search_agent.routes import router as research_router
from recommendation.routes import router as recommendation_router


app = FastAPI()

app.include_router(chat_router, prefix="/chat", tags=["Chat Summary"])
app.include_router(research_router, prefix="/research", tags=["Search Agent"])
app.include_router(recommendation_router)

@app.get("/")
def root():
    return {"message": "Welcome to the Event API"}