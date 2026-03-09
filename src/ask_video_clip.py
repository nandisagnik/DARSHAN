import chromadb
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ---------- CHROMADB ----------
db = chromadb.PersistentClient(path="chroma_db")
collection = db.get_collection("video_segments_multi")

print("Collection count:", collection.count())


# ---------- EMBED QUERY ----------
def embed_query(text):

    # enrich query slightly so CLIP matches scene/event text better
    enriched = f"""
User query: {text}

Possible elements to match:
actors, actions, objects, location, sounds, speech, events
"""

    inputs = processor(
        text=[enriched],
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    with torch.no_grad():
        features = model.get_text_features(**inputs)

    emb = features[0].cpu().numpy()
    emb = emb / np.linalg.norm(emb)

    return emb.tolist()


# ---------- RETRIEVE SEGMENTS ----------
def retrieve_segments(question, video_id, top_k=10):

    q_emb = embed_query(question)

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        where={"video_id": video_id}
    )

    matches = results["metadatas"][0]

    # ---------- TEMPORAL EXPANSION ----------
    windows = []

    for m in matches:
        start = m["start"]
        end = m["end"]

        windows.append((start - 20, end + 20))

    # ---------- GET ALL SEGMENTS FROM VIDEO ----------
    all_segments = collection.get(
        where={"video_id": video_id}
    )["metadatas"]

    expanded = []

    for seg in all_segments:

        s = seg["start"]
        e = seg["end"]

        for w_start, w_end in windows:

            if e >= w_start and s <= w_end:
                expanded.append(seg)
                break

    # ---------- DEDUPLICATE ----------
    unique = {}
    for seg in expanded:
        key = (seg["start"], seg["end"])
        unique[key] = seg

    expanded = list(unique.values())

    # ---------- SORT CHRONOLOGICALLY ----------
    expanded.sort(key=lambda x: x["start"])

    return expanded


# ---------- BUILD CONTEXT ----------
def build_context(matches):

    context = ""

    for m in matches:

        event = m.get("event_summary", "")

        context += (
            f"Time {m['start']}–{m['end']} sec\n"
            f"Visual: {m['visual']}\n"
            f"Speech: {m['speech']}\n"
            f"Sounds: {m['sounds']}\n"
            f"Event: {event}\n\n"
        )

    return context


# ---------- ASK QUESTION ----------
def ask_question(question, video_id):

    matches = retrieve_segments(question, video_id)

    if not matches:
        return "Not observed in selected video."

    context = build_context(matches)

    prompt = f"""
You are analyzing a timeline of a video.

User Question:
{question}

Video ID:
{video_id}

Timeline segments:

{context}

Instructions:

1. Use ONLY the provided segments.
2. Identify events relevant to the question.
3. Track actors/actions across time if needed.
4. Provide timestamps.
5. If multiple events exist, list them.

Response format:

Event 1:
Time: <start–end sec>
Explanation:

Event 2:
Time: <start–end sec>
Explanation:

Final reasoning:
Explain how the events relate to the question.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3000
    )

    return response.choices[0].message.content


# ---------- INTERACTIVE LOOP ----------
if __name__ == "__main__":

    while True:

        video_id = input("Select video (day1/day2): ")
        q = input("Question: ")

        if q.lower() == "exit":
            break

        print("\nAnswer:\n")
        print(ask_question(q, video_id))
        print("\n")
