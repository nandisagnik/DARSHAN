import json
import chromadb
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("video_segments_multi")


# ---------- TEXT EMBEDDING ----------
def embed_text(text):

    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    with torch.no_grad():
        features = model.get_text_features(**inputs)

    emb = features[0].cpu().numpy()
    emb = emb / np.linalg.norm(emb)

    return emb.tolist()


# ---------- FRAME EMBEDDING ----------
def embed_frame(video_path):

    cap = cv2.VideoCapture(video_path)

    embeddings = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            features = model.get_image_features(**inputs)

        emb = features[0].cpu().numpy()
        emb = emb / np.linalg.norm(emb)

        embeddings.append(emb)

    cap.release()

    if len(embeddings) == 0:
        return None

    avg_emb = np.mean(embeddings, axis=0)
    avg_emb = avg_emb / np.linalg.norm(avg_emb)

    return avg_emb.tolist()


# ---------- LOAD TIMELINE ----------
with open("timeline.json", "r") as f:
    timeline = json.load(f)


video_id = input("Enter video ID (example: day1 or day2): ")

count = 0


for i, seg in enumerate(timeline):

    # ----- structured scene info -----
    actors = ", ".join(seg.get("scene", {}).get("actors", []))
    actions = ", ".join(seg.get("scene", {}).get("actions", []))
    objects = ", ".join(seg.get("scene", {}).get("objects", []))
    location = seg.get("scene", {}).get("location", "")

    event_summary = seg.get("event_summary", "")

    # ----- build strong semantic text -----
    text = f"""
Visual description: {seg['visual']}

Speech: {seg['speech']}

Sounds: {seg['sounds']}

Actors present: {actors}

Actions happening: {actions}

Objects involved: {objects}

Location: {location}

Event summary: {event_summary}
"""

    text_embedding = embed_text(text)

    video_path = f"segments/seg_{seg['start']}_{seg['end']}.mp4"
    frame_embedding = embed_frame(video_path)

    metadata = {
        "video_id": video_id,
        "start": seg["start"],
        "end": seg["end"],
        "visual": seg["visual"],
        "speech": seg["speech"],
        "sounds": seg["sounds"],
        "event_summary": event_summary
    }

    # ----- store text embedding -----
    collection.add(
        ids=[f"{video_id}_text_{i}"],
        embeddings=[text_embedding],
        metadatas=[metadata]
    )

    count += 1

    # ----- store frame embedding -----
    if frame_embedding is not None:

        collection.add(
            ids=[f"{video_id}_frame_{i}"],
            embeddings=[frame_embedding],
            metadatas=[metadata]
        )

        count += 1


print("Stored embeddings:", count)
print("Video ID:", video_id)
