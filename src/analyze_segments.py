import base64  # used to convert image bytes → text (so model can read images)
import cv2  # OpenCV for reading video frames
from openai import OpenAI  # OpenAI client for vision + speech reasoning
from panns_infer import detect_sound_events  # PANNs model for environmental sound detection
from dotenv import load_dotenv  # load API key from .env file
import os  # file handling
import json  # for saving timeline data

# Load API key from environment
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -------- Extract a few key frames from a video segment --------
def frames_to_b64(video_path, max_frames=4):
    cap = cv2.VideoCapture(video_path)  # open video segment
    frames = []

    # Total number of frames in this small video
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    # Decide step so only a few frames are sampled (not entire video)
    step = max(1, total // max_frames)

    i = 0
    while True:
        ret, frame = cap.read()  # read next frame
        if not ret:
            break

        # Sample only some frames (reduces cost + keeps important visuals)
        if i % step == 0:
            _, buf = cv2.imencode(".jpg", frame)  # convert frame → JPEG bytes
            frames.append(base64.b64encode(buf).decode("utf-8"))  # encode → text

        i += 1

    cap.release()  # release video file
    return frames


# -------- Analyze one 2-second segment --------
def analyze_segment(seg):

    # ----- Visual understanding -----
    # Extract few frames from this segment
    frames_b64 = frames_to_b64(seg["video"])

    # Build prompt for vision model
    content = [{"type": "text", "text": "Describe briefly what is happening in these frames."}]
    for f in frames_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}})

    # Send frames → vision model → get scene description
    vis = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        max_tokens=200,
    ).choices[0].message.content


    # ----- Speech + sound understanding -----
    # If audio exists in this segment
    if seg["audio"] and os.path.exists(seg["audio"]):

        # Read audio bytes
        with open(seg["audio"], "rb") as f:
            audio_bytes = f.read()

        # Speech recognition (what was spoken)
        speech = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=("seg.wav", audio_bytes),
        ).text

        # Detect environmental sounds (dog, footsteps, crowd, etc.)
        events = detect_sound_events(seg["audio"], top_k=3)
        sounds = ", ".join([e[0] for e in events])

    else:
        # If no audio in this segment
        speech = "No speech"
        sounds = "No sound"


    # Return structured observation for this segment
    return {
        "start": seg["start"],   # segment start time
        "end": seg["end"],       # segment end time
        "visual": vis,           # what was seen
        "speech": speech,        # what was spoken
        "sounds": sounds,        # background sounds detected
    }


# -------- Main execution --------
if __name__ == "__main__":
    from temporal_segments import segment_video  # import segmentation function

    # Cut video into small time segments
    segs = segment_video("video.mp4")

    timeline = []  # stores full video understanding over time

    total = len(segs)
    print(f"\nProcessing {total} segments...\n")

    # Analyze each segment one by one
    for i, s in enumerate(segs, 1):
        print(f"[{i}/{total}] Analyzing segment {s['start']}–{s['end']} sec...")
        result = analyze_segment(s)
        timeline.append(result)

    print("\nTIMELINE COMPLETE\n")
    print(json.dumps(timeline, indent=2))

    # Save timeline to file for later querying
    with open("timeline.json", "w") as f:
        json.dump(timeline, f, indent=2)

'''This script performs multimodal understanding of a video over time.

Step 1 — The video is split into small time segments (2 seconds each).

Step 2 — For each segment:
    • A few frames are extracted → Vision model describes what is happening.
    • Audio is analyzed:
        - Speech is transcribed (what was said).
        - Environmental sounds are detected using PANNs (real audio classification).

Step 3 — The system creates a time-aligned timeline:
    For every moment → it knows:
        - What was seen
        - What was spoken
        - What sounds were present

Step 4 — The full timeline is saved as timeline.json,
which can later be queried (e.g., "When did horses appear?").

In simple terms:
Video + Speech + Environmental Sound → Structured timeline of events over time.'''
