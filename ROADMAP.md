# Roadmap

This roadmap outlines the planned evolution of DARSHAN. The goal is to progressively improve multimodal perception, retrieval accuracy, and real-world usability.

---

## v1.1 — Audio Understanding Enhancements

* Redesign and retrain the audio classification pipeline from scratch to support a significantly larger set of environmental sound classes.
* Expand the audio event taxonomy to better represent real-world acoustic environments.
* Improve robustness and detection accuracy for overlapping or noisy audio signals.

---

## v1.2 — Local Model Deployment

* Enable the entire framework to run with locally hosted models, reducing reliance on external APIs.
* Optimize inference pipelines to maintain strong performance with minimal trade-offs in accuracy and latency.
* Support lightweight and quantized models for resource-constrained environments.

---

## v1.3 — Multimodal Reasoning Improvements

* Integrate advanced temporal reasoning models to better understand event sequences and cause–effect relationships within videos.
* Improve retrieval ranking through hybrid search and reranking strategies.

---

## v2.0 — Real-Time Video Intelligence

* Implement real-time video question answering capable of processing live video streams.
* Introduce person tracking and entity-aware event detection to follow individuals across frames.
* Enable multi-video memory, allowing the system to reason across multiple videos and maintain contextual continuity.

---

## v2.1 — User Experience & Accessibility

* Develop a graphical user interface (GUI) for interactive video querying and visualization of detected events.
* Provide timeline exploration tools that allow users to navigate and inspect multimodal video understanding results.

---

## Long-Term Vision

### Intelligent Edge-Based CCTV System

A key long-term objective of DARSHAN is the development of an **intelligent smart CCTV system** that integrates multimodal AI directly into the camera infrastructure.

The vision is to design a **privacy-preserving, edge-powered surveillance device** built around low-cost embedded hardware such as **ESP32-based camera modules** and **Raspberry Pi / NVIDIA Jetson processing units**.

Instead of traditional CCTV systems that rely on passive video recording or proprietary cloud processing, DARSHAN aims to enable **on-device intelligent perception and reasoning**.

Key capabilities of the proposed system include:

* Real-time scene understanding from video streams
* Detection of human activities and environmental audio events
* Natural language querying of recorded events
* Edge-based inference to reduce reliance on cloud infrastructure
* Secure, user-controlled storage and processing of video data

In the long term, this approach could lead to the development of a **commercial-grade intelligent CCTV product** that combines:

* camera hardware
* embedded edge computing
* multimodal AI reasoning

Such systems could be deployed in **homes, offices, research labs, and smart environments**, providing intelligent monitoring while maintaining **data privacy and local control**.

DARSHAN serves as the **core intelligence layer** that powers this future generation of smart surveillance devices.


# Limitations and Current Challenges

While the current system demonstrates the feasibility of multimodal video understanding and question answering, several technical limitations remain. These limitations motivate the future improvements outlined in the project roadmap.

---

# 1. Edge Hardware Constraints

The system is intended to eventually run on **low-cost edge devices** such as Raspberry Pi or similar embedded platforms. However, several hardware-related challenges currently exist.

### Low-Cost Camera Integration Issues

Initial experiments with low-cost (~₹1000) camera modules integrated with **ESP32-based systems** revealed thermal stability issues.

Observed behavior:

* Camera modules begin to **overheat after approximately 30 minutes of continuous operation**
* Sustained video streaming significantly increases thermal load
* Hardware throttling may reduce capture stability

These limitations must be addressed before deploying the system in long-running edge environments.

---

# 2. Latency in Local Model Inference

Running the pipeline entirely using **local models** introduces significant latency.

Challenges include:

* slower inference compared to cloud-hosted models
* limited compute resources on edge hardware
* increased processing time for multimodal analysis

As a result, the current system performs best when processing **pre-recorded video segments** rather than live video streams.

---

# 3. End-to-End Processing Time

The complete processing pipeline currently includes:

* video segmentation
* multimodal frame analysis
* speech transcription
* audio event detection
* vector database retrieval
* language model reasoning

The **end-to-end processing time** for this pipeline remains relatively high.

Reducing this latency is a key challenge for enabling:

* real-time video analysis
* streaming video question answering
* edge-based deployment

Optimizing the pipeline for faster inference is therefore a major focus for future development.

---

# 4. Limited Access to Advanced Multimodal Models

Recent research has introduced advanced **Audio-Visual Language Models (AVLMs)** capable of deeper multimodal reasoning.

For example, several models proposed in recent research publications (2025) demonstrate strong performance on multimodal reasoning tasks. However, some of these models have not yet released public implementations or weights.

As a result:

* access to cutting-edge multimodal architectures remains limited
* replication and experimentation are constrained
* existing systems must rely on modular pipelines instead of unified multimodal models

Developing **domain-specific multimodal models from scratch** could significantly improve system capabilities.

---

# 5. Hardware Integration Challenges

The long-term vision includes deploying the system as a **dedicated hardware unit** consisting of:

* camera
* microphone
* embedded computing device (e.g., Raspberry Pi Zero W)
* wireless connectivity

However, integrating the full multimodal pipeline within such a constrained hardware environment introduces challenges related to:

* computational resources
* thermal stability
* memory constraints
* power efficiency

Addressing these constraints will require optimized models and efficient system architecture.

---

# 6. Emerging Interaction Paradigms

The system currently relies on **text-based query interfaces**.

Future deployments may benefit from integrating with emerging interaction platforms such as:

* voice-based interfaces
* wearable devices (e.g., smart glasses)
* ambient AI environments

These applications introduce additional challenges in latency, energy consumption, and multimodal perception accuracy.

---

# Summary

The primary technical challenges currently faced by the system include:

* thermal limitations in low-cost edge camera systems
* latency when running local models
* high end-to-end processing time
* limited access to state-of-the-art multimodal models
* hardware integration constraints for edge deployment

Addressing these limitations is the primary motivation behind the **future roadmap and system evolution**.

