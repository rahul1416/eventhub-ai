from pymongo import MongoClient
from datetime import datetime, timezone
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from dotenv import load_dotenv
import os
from bson import ObjectId
from dateutil import parser

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MongoDB URI is missing in environment variables.")

client = MongoClient(MONGO_URI)
db = client['eventhub']
events_collection = db['events']
registrations_collection = db['registrations']
users_collection = db['users']

print("Loaded events:", events_collection.count_documents({}))
print("Loaded registrations:", registrations_collection.count_documents({}))

def parse_to_utc(dt):
    """Ensure all datetimes are UTC-aware"""
    if isinstance(dt, str):
        dt = parser.isoparse(dt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt

def get_user_past_events(user_id):
    user_oid = ObjectId(user_id)
    registrations = registrations_collection.find({"userId": user_oid})
    event_ids = [r["eventId"] for r in registrations]
    return list(events_collection.find({"_id": {"$in": event_ids}}))

def filter_candidate_events(user_id):
    user_oid = ObjectId(user_id)
    registered_event_ids = set(
        r["eventId"] for r in registrations_collection.find({"userId": user_oid})
    )
    # No comparison with datetime — rely only on 'status'
    candidate_events = list(events_collection.find({
        "status": {"$in": ["live", "upcoming"]},
        "_id": {"$nin": list(registered_event_ids)}
    }))
    return candidate_events

def cluster(events_list, n_clusters=5):
    descriptions = [e.get("description", "") for e in events_list]
    tfidf = TfidfVectorizer(max_features=50, stop_words="english")
    tfidf_features = tfidf.fit_transform(descriptions).toarray()

    event_types = [e.get("eventType", "unknown") for e in events_list]
    organizers = [str(e.get("organizerId", "unknown")) for e in events_list]
    encoder = OneHotEncoder(handle_unknown='ignore')
    categorical_features = encoder.fit_transform(np.column_stack([event_types, organizers])).toarray()

    features = np.hstack([tfidf_features, categorical_features])
    kmeans = KMeans(n_clusters=min(n_clusters, len(events_list)), random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features)

    for i, event in enumerate(events_list):
        event["cluster"] = int(clusters[i])
    return events_list

def recommend_events(user_id, top_n=3):
    user_oid = ObjectId(user_id)
    past_events = get_user_past_events(user_oid)
    live_events = list(events_collection.find({"status": {"$in": ["live", "upcoming"]}}))

    if not live_events:
        return []

    clustered_events = cluster(live_events)
    user_clusters = list(set([e.get("cluster") for e in past_events if "cluster" in e])) if past_events else []

    candidate_events = filter_candidate_events(user_oid)
    if not candidate_events:
        # Fallback: most recent or popular events
        return list(events_collection.find().sort([("createdAt", -1)]).limit(top_n))

    event_descriptions = [e.get("description", "") for e in candidate_events]
    past_descriptions = [e.get("description", "") for e in past_events] if past_events else [""]

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(event_descriptions + past_descriptions)
    candidate_vectors = tfidf_matrix[:len(candidate_events)]
    past_vectors = tfidf_matrix[len(candidate_events):]

    similarity_scores = cosine_similarity(past_vectors, candidate_vectors).mean(axis=0) if past_events else [0.5] * len(candidate_events)

    scored_events = []
    for idx, event in enumerate(candidate_events):
        cluster_score = 1 if user_clusters and event.get("cluster") in user_clusters else 0.5
        type_score = 1 if past_events and event.get("eventType", "") in [e.get("eventType", "") for e in past_events] else 0.5
        score = (0.5 * similarity_scores[idx]) + (0.3 * cluster_score) + (0.2 * type_score)
        scored_events.append((event, score))

    scored_events.sort(key=lambda x: x[1], reverse=True)
    return [event for event, _ in scored_events[:top_n]]

# Example usage:
# if __name__ == "__main__":
# user_id = "681e52223ab5f5946dcacec0"
# recommended = recommend_events(user_id)
# print(recommended)