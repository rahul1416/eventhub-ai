from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from bson import ObjectId
from recommendation.recommender import recommend_events

router = APIRouter()

@router.get("/recommendations/{user_id}")
def get_recommendations(user_id: str, top_n: int = 3):
    try:
        recommendations = recommend_events(user_id=user_id, top_n=top_n)
        if not recommendations:
            return {"message": "No recommendations found"}

        # Ensure all ObjectIds are converted to str before returning
        safe_recommendations = jsonable_encoder(
            recommendations,
            custom_encoder={ObjectId: str}
        )

        return {
            "user_id": user_id,
            "recommended_events": safe_recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
