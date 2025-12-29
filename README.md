# VN-ASR Mobile Editor: Automated Vietnamese Speech-to-Text on Smartphone

This project develops an automated Vietnamese speech-to-text application for smartphones. It features a highly optimized **SqueezeFormer-XS** model, designed to balance high accuracy and low latency for real-time mobile environments.

## ## Project Specifications

| Feature | Requirement / Context | Achievement |
| :--- | :--- | :--- |
| **Model Architecture** | Lightweight for Mobile | SqueezeFormer-XS (~9M parameters) |
| **Training Data** | Vietnamese Corpora | VLSP2020, FPT, VIVOS |
| **Hardware** | GPU Training | NVIDIA RTX 3050 (6GB VRAM) |
| **Epochs** | Model Convergence | 200 Epochs |
| **Optimization** | Edge Deployment | TFLite (Quantization & Optimization) |
| **Streaming Logic** | Real-time Stability | Local Agreement Algorithm |
| **Language Model** | Accuracy Boost | 4-gram LM Integration |

## ## Performance Evaluation

The integration of a **4-gram Language Model** combined with **Beam Search** significantly reduces the Word Error Rate (WER) across various test sets.

### ### 1. Word Error Rate (WER %)
| Dataset | WER no LM (%) | WER with 4-gram LM (%) |
| :--- | :---: | :---: |
| **Speaker + VIVOS** | 24.50 | **5.89** |
| **Database** | 28.12 | **19.02** |
| **FPT** | 40.72 | **7.71** |
| **Self-recorded** | 29.80 | **7.54** |

### ### 2. Mobile Inference Benchmarks (Offline ASR)
Evaluated on a 5-second audio sample:

| Device | Latency (s) | Real-Time Factor (RTF) |
| :--- | :---: | :---: |
| Samsung M10 | 13.73 | 1.75 |
| OPPO A16 | 12.80 | 1.56 |
| **Vivo X100s Pro** | **6.10** | **0.22** |

## ## Application Interface

The Android application provides a seamless user experience for voice-based text editing. It supports both online and offline recognition modes.

<p align="center">
  <img src="images/app_screen_1.png" width="45%" alt="Application UI 1">
  <img src="images/app_screen_2.png" width="45%" alt="Application UI 2">
</p>

*Figure 1: Mobile application interface displaying speech recognition in progress.*

## ## Streaming ASR & Management Logs

By utilizing the **Local Agreement** algorithm, the system ensures that the transcription remains consistent during the streaming process.

![Gateway Logs](images/streaming_logs.png)
*Figure 2: Real-time logs demonstrating the Local Agreement logic during voice input.*

## ## Source Code & Documentation

*   **Android Source Code:** [Access Repository Here](https://github.com/Luudinhtu2003/android-app-asr-tensorRT)
*   **Full Thesis Report:** [ASR_thesis.pdf](https://github.com/Luudinhtu2003/ASR-SqueezeFormer-Finetune/blob/main/ASR_thesis.pdf)

## ## References

1.  **SqueezeFormer:** Kim, S., et al. *"SqueezeFormer: An Efficient Transformer for Automatic Speech Recognition."* (2022).
2.  **CTC:** Graves, A., et al. *"Connectionist Temporal Classification: Labelling Unsegmented Sequence Data."*
3.  **Beam Search:** Heafield, K. *"KenLM: Faster and Smaller Language Model Queries."*
4.  **Local Agreement:** Senior, A., et al. *"Local Agreement for Streaming ASR."*

---
**Author:** [Dinh-Tu]  
**Student ID:** 20213016
