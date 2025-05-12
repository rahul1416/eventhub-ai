from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from chat_summary.routes import router as chat_router
from search_agent.routes import router as research_router
from recommendation.routes import router as recommendation_router

app = FastAPI()

# CORS Middleware Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to specific origins like ["https://yourfrontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(chat_router, prefix="/chat", tags=["Chat Summary"])
app.include_router(research_router, prefix="/research", tags=["Search Agent"])
app.include_router(recommendation_router)

@app.get("/")
def root():
    return {"message": "Welcome to the Event API"}

