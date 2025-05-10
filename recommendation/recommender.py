import json
from datetime import datetime, timezone
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Load data
with open('data/events.json', 'r') as f:
    events = json.load(f)

with open('data/registrations.json', 'r') as f:
    registrations = json.load(f)

with open('data/users.json', 'r') as f:
    users = json.load(f)

def get_user_past_events(user_id):
    past_event_ids = [r['eventId'] for r in registrations if r['userId'] == user_id]
    return [e for e in events if e['_id'] in past_event_ids]

def filter_candidate_events(user_id, include_attended_events=True):
    current_time = datetime.now(timezone.utc)
    candidates = []
    registered_event_ids = {r['eventId'] for r in registrations if r['userId'] == user_id}
    for event in events:
        end_time = datetime.fromisoformat(event['endTime'].replace('+00:00', '+00:00')).replace(tzinfo=timezone.utc)
        is_past = end_time <= current_time
        is_registered = event['_id'] in registered_event_ids
        include_event = (include_attended_events and is_past) if is_registered else (event['status'] == 'live' and (not is_past or include_attended_events))
        if include_event:
            event_copy = event.copy()
            event_copy.update({
                'is_past': is_past,
                'is_registered': is_registered,
                'user_attended': is_registered and is_past
            })
            candidates.append(event_copy)
    return candidates

def cluster(events, n_clusters=5):
    descriptions = [e["description"] for e in events]
    tfidf = TfidfVectorizer(max_features=50, stop_words="english")
    tfidf_features = tfidf.fit_transform(descriptions).toarray()
    event_types = [e["eventType"] for e in events]
    organizers = [e["organizerId"] for e in events]
    encoder = OneHotEncoder(handle_unknown='ignore')
    categorical_features = encoder.fit_transform(np.column_stack([event_types, organizers])).toarray()
    features = np.hstack([tfidf_features, categorical_features])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)
    for i, event in enumerate(events):
        event["cluster"] = int(clusters[i])
    return events

def recommend_events(user_id, top_n=3):
    clustered_events = cluster(events)
    past_events = get_user_past_events(user_id)
    user_clusters = list(set([e["cluster"] for e in past_events])) if past_events else []
    candidate_events = filter_candidate_events(user_id)

    if candidate_events:
        event_descriptions = [e["description"] for e in candidate_events]
        past_descriptions = [e["description"] for e in past_events] if past_events else [""]
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(event_descriptions + past_descriptions)
        candidate_vectors = tfidf_matrix[:len(candidate_events)]
        past_vectors = tfidf_matrix[len(candidate_events):]
        similarity_scores = cosine_similarity(past_vectors, candidate_vectors).mean(axis=0) if past_events else [0.5]*len(candidate_events)
        scored_events = []
        for idx, event in enumerate(candidate_events):
            cluster_score = 1 if user_clusters and event["cluster"] in user_clusters else 0.5
            type_score = 1 if past_events and event["eventType"] in [e["eventType"] for e in past_events] else 0.5
            start_time = datetime.fromisoformat(event["startTime"].replace("+00:00", "+00:00")).replace(tzinfo=timezone.utc)
            days_until_start = (start_time - datetime.now(timezone.utc)).days
            time_score = 1 / (1 + abs(days_until_start))
            score = (0.4 * similarity_scores[idx] + 0.3 * cluster_score + 0.2 * type_score + 0.1 * time_score)
            scored_events.append((event, score))
        scored_events.sort(key=lambda x: x[1], reverse=True)
        return [event for event, _ in scored_events[:top_n]]

    event_popularity = defaultdict(int)
    for reg in registrations:
        event_popularity[reg["eventId"]] += 1
    user_registered_ids = {r["eventId"] for r in registrations if r["userId"] == user_id}
    if user_clusters:
        cluster_events = [e for e in clustered_events if e["cluster"] in user_clusters and e["_id"] not in user_registered_ids]
        cluster_events.sort(key=lambda x: -event_popularity[x["_id"]])
        return cluster_events[:top_n] if cluster_events else []
    all_events = [e for e in clustered_events if e["_id"] not in user_registered_ids]
    all_events.sort(key=lambda x: -event_popularity[x["_id"]])
    return all_events[:top_n]
