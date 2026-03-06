# Model Architecture

## Overview

DARSHAN is a multimodal video intelligence system designed to understand events occurring in video streams by integrating information from multiple modalities including vision, speech, and environmental sound.

Instead of relying on a single monolithic model, DARSHAN uses a modular architecture where specialized models process different modalities and their outputs are combined into a unified reasoning pipeline.

This design allows the system to remain flexible, interpretable, and extensible.

---

## System Pipeline

The DARSHAN pipeline transforms raw video into structured knowledge using the following stages:

```
Video Input
     ↓
Temporal Segmentation
     ↓
Multimodal Analysis
 ├─ Vision Understanding
 ├─ Speech Recognition
 └─ Environmental Sound Detection
     ↓
Timeline Construction
     ↓
Contextual Understanding
     ↓
Semantic Embeddings
     ↓
Vector Database
     ↓
Natural Language Query
     ↓
Language Model Reasoning
     ↓
Final Answer
```

---

## Temporal Segmentation

The first stage divides the input video into short temporal segments.

Typical segment duration:

```
2 seconds
```

Example segmentation output:

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

This segmentation allows the system to analyze events at fine temporal resolution.

---

## Vision Understanding

Representative frames from each segment are analyzed using a vision-language model.

The model generates semantic descriptions of the scene.

Example output:

```
Three people are gathered around a computer.
```

This step captures visual context and activities occurring in the video.

---

## Speech Recognition

Speech present in the audio track is transcribed using automatic speech recognition.

Example transcript:

```
Hello everyone
```

Speech information provides additional context for interpreting events in the video.

---

## Environmental Sound Detection

Environmental sounds are detected using **PANNs (Pretrained Audio Neural Networks)**.

Detected sounds may include:

- footsteps
- door opening
- keyboard typing
- background music

Example output:

```
["footsteps", "door", "typing"]
```

Audio cues help detect events that may not be visually obvious.

---

## Timeline Construction

Outputs from the vision, speech, and sound models are merged into a structured timeline.

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

This timeline forms the semantic representation of the video.

---

## Contextual Understanding

DARSHAN introduces contextual reasoning across segments to better understand temporal relationships.

Example:

Segment 1: person walks toward the door  
Segment 2: door opens  

Inference:

A person exited the room.

This step improves event interpretation by considering temporal continuity.

---

## Semantic Embeddings

Segment descriptions are converted into vector embeddings using **CLIP**.

Model used:

```
openai/clip-vit-base-patch32
```

These embeddings enable semantic similarity comparisons between video segments and user queries.

---

## Vector Database

Embeddings are stored in **ChromaDB** along with segment metadata.

Stored information includes:

- start timestamp
- end timestamp
- visual description
- speech transcript
- detected sound events

This enables efficient retrieval of relevant video segments.

---

## Query Processing

User queries are embedded using the same text encoder.

Example query:

```
When did someone leave the room?
```

Relevant segments are retrieved from the vector database.

---

## Language Model Reasoning

Retrieved segments are passed to a language model which generates the final answer.

Example output:

```
A person exited the room around 22–24 seconds.
```

---

## Design Principles

DARSHAN follows several architectural principles:

- modular multimodal perception
- interpretable timeline representation
- semantic video retrieval
- extensible AI pipeline

---

## Future Architecture Improvements

Planned improvements include:

- unified audio-visual language models
- streaming real-time inference
- edge-optimized model deployment
- multi-video knowledge reasoning