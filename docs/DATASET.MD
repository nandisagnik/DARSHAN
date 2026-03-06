# Dataset

## Overview

DARSHAN processes video data containing visual scenes, speech, and environmental audio to construct a structured understanding of events occurring over time.

Rather than relying on a single fixed dataset, DARSHAN is designed to work with general video recordings and automatically convert them into semantic event timelines.

This allows the system to be applied to many environments including:

- indoor monitoring
- office environments
- collaborative workspaces
- smart surveillance systems

---

## Input Data

The primary input to DARSHAN is a video file.

Example dataset structure:

```
data/
 ├ video.mp4
```

The video may contain:

- human activities
- spoken dialogue
- environmental sounds
- background events

---

## Modalities

DARSHAN processes three modalities extracted from the video.

| Modality | Description |
|--------|-------------|
| Vision | scene and activity description |
| Speech | transcription of spoken dialogue |
| Audio | detection of environmental sound events |

These modalities are combined to generate semantic event descriptions.

---

## Temporal Segmentation

The input video is automatically divided into short temporal segments.

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
 ├ seg_6_8.mp4
```

Each segment contains:

- video frames
- extracted audio
- timestamp metadata

---

## Timeline Representation

After processing each segment, DARSHAN generates a structured timeline describing events.

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

Two timeline files are typically generated:

```
timeline.json
timeline_human.json
```

Where:

- `timeline.json` contains raw multimodal outputs
- `timeline_human.json` contains refined descriptions generated through contextual reasoning

---

## Dataset Characteristics

The current experimental dataset includes:

- short recorded video clips
- indoor environments
- human activities and conversations
- environmental sounds

Key characteristics:

| Property | Description |
|--------|-------------|
| Video Duration | short clips (< 1 minute) |
| Environment | indoor workspace |
| Modalities | vision, speech, environmental audio |
| Output Format | structured timeline |

---

## Dataset Limitations

Current dataset limitations include:

- small number of video samples
- limited environmental diversity
- absence of manually labeled ground truth events
- limited number of detectable sound classes

Future work will involve expanding the dataset and evaluating the system on larger public datasets.

---

## Potential Benchmark Datasets

Future experiments may evaluate DARSHAN using public datasets such as:

- AudioSet
- ActivityNet
- AVA Dataset
- Ego4D

These datasets provide large-scale multimodal video data suitable for benchmarking multimodal reasoning systems.

---

## Privacy Considerations

DARSHAN is designed with privacy-preserving deployment in mind.

Future edge deployments aim to support:

- local video processing
- user-controlled storage
- on-device inference

This enables the development of intelligent edge-based camera systems that maintain user privacy.