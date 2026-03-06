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


def embed_query(text):

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


def retrieve_segments(question, video_id, top_k=10):

    q_emb = embed_query(question)

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        where={"video_id": video_id}
    )

    return results["metadatas"][0]


def ask_question(question, video_id):

    matches = retrieve_segments(question, video_id)

    if not matches:
        return "Not observed in selected video."

    context = ""

    for m in matches:
        context += (
            f"Time {m['start']}–{m['end']} sec:\n"
            f"Visual: {m['visual']}\n"
            f"Speech: {m['speech']}\n"
            f"Sounds: {m['sounds']}\n\n"
        )

    prompt = f"""
User asked: {question}

Video: {video_id}

Relevant observed segments:

{context}

Answer clearly:
- Tell WHEN the event happened (exact timestamp)
- Explain briefly
- If multiple → list all times
- Do NOT guess outside given segments

Format the response like this:

Event 1: Time <start–end sec>
<your response>

Event 2: Time <start–end sec>
<your response>
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3000
    )

    return response.choices[0].message.content


if __name__ == "__main__":

    while True:

        video_id = input("Select video (day1/day2): ")
        q = input("Question: ")

        if q.lower() == "exit":
            break

        print("\nAnswer:", ask_question(q, video_id), "\n")