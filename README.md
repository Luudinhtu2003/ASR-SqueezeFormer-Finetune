# VN-ASR Mobile Editor: Automated Vietnamese Speech-to-Text on Smartphone
This project focuses on designing and implementing an automated Vietnamese speech-to-text application on smartphones. By leveraging the SqueezeFormer-XS architecture and advanced optimization techniques, the system provides high-accuracy, real-time transcription directly on mobile devices.
## Project Overview
The core of this project is a lightweight Automatic Speech Recognition (ASR) model tailored for Vietnamese. The model was trained on large-scale datasets including VLSP2020, FPT, and VIVOS, along with custom-recorded data to enhance real-world robustness.
Hardware: Trained on NVIDIA RTX 3050 (6GB VRAM).
Training Duration: Approximately 200 epochs.
Model Architecture: SqueezeFormer-XS (~9M parameters).
Core Theories: CTC Loss, Greedy Search, Beam Search, and N-gram Language Modeling.
## Streaming ASR & Local Agreement
To handle real-time transcription (Streaming ASR), the project implements the Local Agreement algorithm. This ensures that the displayed text remains stable and updates accurately as the audio stream is processed, providing a seamless user experience.
![alt text](images/streaming_log.png)

Figure 1: Log showing the Local Agreement algorithm stabilizing the streaming text output.
## Mobile Deployment & Optimization
To ensure high performance on mobile hardware, the trained model was converted to TensorFlow Lite (TFLite). We applied Quantization and Optimization techniques to reduce model size and latency without significantly sacrificing accuracy.
Platform: Android OS.
Format: TFLite with Integer/Float16 Quantization.
## Performance Evaluation
The system was evaluated based on Word Error Rate (WER), Latency, and Real-Time Factor (RTF).
### 1. Word Error Rate (WER) with 4-gram Language Model
Integrating a 4-gram Language Model significantly improved recognition accuracy:
Dataset	Type	WER no LM (%)	WER with 4-gram LM (%)
Speaker + VIVOS	Test	24.50	5.89
Database	Test	28.12	19.02
FPT	Test	40.72	7.71
Custom Recorded	Test	29.80	7.54
### 2. Latency and RTF on Smartphones (Offline Mode)
Evaluation performed on various mobile devices (5s audio input):
Device	Latency (s)	RTF
Samsung M10	13.73	1.75
OPPO A16	12.80	1.56
Vivo X100s Pro	6.10	0.22
## Application Interface
The Android application features a minimalist UI designed for efficient text editing via voice.
<p align="center">
<img src="images/app_interface_1.png" width="45%" alt="App Interface Start">
<img src="images/app_interface_2.png" width="45%" alt="App Interface Prediction">
</p>
Figure 2: Mobile UI displaying the "Start Speaking" trigger and the real-time transcription result.
## Source Code & Documentation
Android App Source Code: View Repository Here
Full Technical Report: ASR_thesis.pdf
## References
SqueezeFormer: Kim, S., et al. "SqueezeFormer: An Efficient Transformer for Automatic Speech Recognition."
CTC: Graves, A., et al. "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data."
Local Agreement: Senior, A., et al. "Local Agreement for Streaming ASR."
Language Modeling: Heafield, K. "KenLM: Faster and Smaller Language Model Queries."
Author: [Your Name]
Student ID: 20213016
Project: Graduation Thesis / Research Project 2024
