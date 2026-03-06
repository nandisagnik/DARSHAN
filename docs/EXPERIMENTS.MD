# Experiments

## Overview

This document describes how to reproduce the experimental pipeline used in **DARSHAN**.

The system converts raw video recordings into structured semantic timelines and enables natural language querying of events occurring within the video.

The experimental pipeline consists of the following stages:

1. Video segmentation  
2. Multimodal analysis  
3. Timeline generation  
4. Semantic embedding generation  
5. Vector database indexing  
6. Natural language querying  

---

## Hardware Requirements

The system can run on standard desktop hardware.

Recommended configuration:

| Component | Requirement |
|----------|-------------|
| CPU | 4+ cores |
| RAM | 8 GB minimum |
| GPU | Optional (recommended for faster inference) |
| Storage | 2–5 GB |

Future versions aim to support edge devices such as:

- Raspberry Pi  
- NVIDIA Jetson  

---

## Software Requirements

The system requires the following software components:

- Python 3.10+
- PyTorch
- FastAPI
- ChromaDB
- Transformers
- OpenCV
- MoviePy
- Librosa

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Experimental Pipeline

### Step 1 — Video Segmentation

The first stage divides the input video into short temporal segments.

Run:

```bash
python temporal_segments.py
```

Output directory:

```
segments/
 ├ seg_0_2.mp4
 ├ seg_2_4.mp4
 ├ seg_4_6.mp4
```

Each segment contains:

- video frames
- audio track
- timestamp metadata

---

### Step 2 — Multimodal Analysis

Each segment is analyzed using vision, speech, and environmental sound models.

Run:

```bash
python analyze_segments.py
```

Output:

```
timeline.json
```

Example timeline entry:

```json
{
  "start": 0,
  "end": 2,
  "visual": "three people gathered around a computer",
  "speech": "hello",
  "sounds": "tick-tock, music"
}
```

---

### Step 3 — Contextual Understanding

The timeline is refined using contextual reasoning across segments.

Run:

```bash
python total_understanding.py
```

Output:

```
timeline_human.json
```

This step generates more natural and coherent event descriptions.

---

### Step 4 — Build Vector Database

Semantic embeddings are generated for each segment and stored in a vector database.

Run:

```bash
python build_vector_db_clip.py
```

The embeddings are stored in **ChromaDB** together with segment metadata.

---

### Step 5 — Query the System

Users can ask natural language questions about the video.

Run:

```bash
python ask_video_clip.py
```

Example query:

```
When did someone open the door?
```

Example response:

```
A person opened the door around 22–24 seconds.
```

---

## Running the API

DARSHAN also provides a REST API using FastAPI.

Start the server:

```bash
uvicorn api:app --reload
```

Open the interactive API documentation:

```
http://127.0.0.1:8000/docs
```

---

## Evaluation Metrics

Possible evaluation metrics include:

| Metric | Description |
|------|-------------|
| Segment Retrieval Accuracy | Correct segments retrieved for a query |
| Event Localization | Accuracy of detected timestamps |
| Speech Recognition Accuracy | Quality of transcript generation |
| Sound Detection Accuracy | Environmental sound classification accuracy |

---

## Experimental Limitations

Current experimental limitations include:

- latency when running local models
- limited dataset diversity
- high computational cost for multimodal processing
- absence of labeled ground truth event annotations

---

## Future Experimental Directions

Future experiments may focus on:

- benchmarking on large-scale video datasets
- evaluating real-time inference performance
- testing edge-device deployments
- evaluating multimodal reasoning accuracy
- supporting multi-video knowledge retrieval