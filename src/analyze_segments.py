import base64
import cv2
from openai import OpenAI
from panns_infer import detect_sound_events
from dotenv import load_dotenv
import os
import json
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -------- Frame extraction --------
def frames_to_b64(video_path, max_frames=5):

    cap = cv2.VideoCapture(video_path)
    frames = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    indices = [int(total * i / max_frames) for i in range(max_frames)]

    current = 0
    idx_pointer = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        if idx_pointer < len(indices) and current >= indices[idx_pointer]:

            _, buf = cv2.imencode(".jpg", frame)
            frames.append(base64.b64encode(buf).decode("utf-8"))
            idx_pointer += 1

        current += 1

    cap.release()
    return frames


# -------- Vision analysis --------
def analyze_visual(frames_b64):

    content = [{
        "type": "text",
        "text": """
Analyze these frames and produce a concise scene description.

Rules:
- Keep description under 30 words
- Focus only on observable actions
- Avoid speculation
"""
    }]

    for f in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{f}"}
        })

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        max_tokens=120
    )

    return resp.choices[0].message.content.strip()


# -------- Structured scene extraction --------
def extract_scene_structure(description):

    prompt = f"""
Convert the scene description into structured information.

Description:
{description}

Return JSON only:

{{
"actors": [],
"actions": [],
"objects": [],
"location": ""
}}

Rules:
- actors = people present
- actions = verbs describing activities
- objects = important physical items
- location = scene setting
"""

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user","content":prompt}],
        max_tokens=150
    )

    text = resp.choices[0].message.content

    try:
        json_text = re.search(r"\{.*\}", text, re.S).group()
        return json.loads(json_text)
    except:
        return {
            "actors": [],
            "actions": [],
            "objects": [],
            "location": ""
        }


# -------- Event summarization --------
def build_event_summary(scene):

    actors = ", ".join(scene["actors"])
    actions = ", ".join(scene["actions"])
    objects = ", ".join(scene["objects"])
    location = scene["location"]

    summary = f"""
Actors: {actors}
Actions: {actions}
Objects: {objects}
Location: {location}
"""

    return summary.strip()


# -------- Analyze one segment --------
def analyze_segment(seg):

    # ----- visual analysis -----
    frames_b64 = frames_to_b64(seg["video"])

    visual_description = analyze_visual(frames_b64)

    scene = extract_scene_structure(visual_description)

    event_summary = build_event_summary(scene)

    # ----- audio -----
    if seg["audio"] and os.path.exists(seg["audio"]):

        with open(seg["audio"], "rb") as f:
            audio_bytes = f.read()

        speech = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=("seg.wav", audio_bytes)
        ).text

        events = detect_sound_events(seg["audio"], top_k=3)
        sounds = ", ".join([e[0] for e in events])

    else:

        speech = "No speech"
        sounds = "No sound"

    return {
        "start": seg["start"],
        "end": seg["end"],
        "visual": visual_description,
        "speech": speech,
        "sounds": sounds,
        "scene": scene,
        "event_summary": event_summary
    }


# -------- Main execution --------
if __name__ == "__main__":

    from temporal_segments import segment_video

    segs = segment_video("video.mp4")

    timeline = []

    total = len(segs)

    print(f"\nProcessing {total} segments...\n")

    for i, s in enumerate(segs, 1):

        print(f"[{i}/{total}] analyzing {s['start']}–{s['end']} sec")

        result = analyze_segment(s)

        timeline.append(result)

    print("\nTIMELINE COMPLETE\n")

    print(json.dumps(timeline, indent=2))

    with open("timeline.json", "w") as f:
        json.dump(timeline, f, indent=2)
