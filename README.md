# 🚀 Efficient Korean Voice-Phishing Detection using QLoRA-tuned Small Language Models (SLMs)

## 💡 Project Overview

Responding to the increasing sophistication of voice-phishing crimes, this project establishes an **efficient and accurate** automated detection system. We overcome the limitations of conventional **LLM-based approaches**, such as prohibitive computational costs, by proposing an optimized training framework based on **Small Language Models (SLMs)**.

### Three Core Contributions
* **1. Maximizing Memory Efficiency:** Application of **QLoRA** to models like Synatra, Kanana, and Bllossom for memory-efficient 4-bit fine-tuning.
* **2. Mitigating Data Imbalance:** Enhanced generalization through **Multilingual Back-Translation (BT-ALL)**, improving minority class detection.
* **3. Long Context Processing:** Implementation of an **Overlapping Sliding Window** strategy to minimize context loss in long conversational transcripts.

---

## 🏗️ Architecture Pipeline

<img src="image/model_architecture.png" alt="Pipeline" width="750"/>

The following diagram illustrates the complete workflow from data preparation to training and evaluation.

### 1. 💾 Data Preparation & Augmentation
* **Dataset:** Transcribed transcripts from **KorCCVi** and **Financial Supervisory Service (FSS)** using **Whisper-large-v3**.
* **Back-Translation (BT):** Applied English, Chinese, and Japanese BT to alleviate class imbalance.
    > **Result:** The **BT-ALL** strategy significantly improved performance across all models. For instance, the Qwen model's F1-Score increased from **0.2581 to 0.8283** through this augmentation.

### 2. ✂️ Text Segmentation Strategy
Processing long conversational texts using different segmentation methods (based on Synatra):

| Strategy | Description | F1-Score |
| :--- | :--- | :--- |
| **Baseline** | Default tokenizer truncation. | 0.9745 |
| **Head & Tail** | Keeping only the start and end of the transcript. | 0.9734 |
| **Sliding Window** | Chunks of 512 tokens with **25% overlap**. | **0.9875** |

---

## 🧠 Model Training & Efficiency Analysis

We evaluated a diverse range of models, from sub-3B SLMs to larger 10B+ models, using **QLoRA** on a single **NVIDIA A100 GPU**.

### 1. Quantitative Performance Comparison
The integrated framework (BT-ALL + Sliding Window) outperformed traditional ML and PLM baselines.

| Category | Model | Params | F1-Score |
| :--- | :--- | :--- | :--- |
| **Proposed SLM** | **Synatra** | **0.9938** |
| **Proposed SLM** | **Qwen** | **0.8283** |
| **Proposed SLM** | **Kanana** | **0.9938** |
| **Proposed SLM** | **Bllossom** | 0.9812 |
| **Proposed SLM** | **Solar** | **0.9969** |
| PLM | KoBERT | 0.6433 |
| ML Models | Random Forest | 0.9835 |

### 2. Qualitative Interpretability

We verified the **Interpretability**, a primary motivation for employing SLMs.
* **Synatra/Kanana:** Consistent with their high F1-Scores, these models accurately identified core phishing patterns, such as **'impersonation of authority'** and **'induction of urgency'**, and provided reasonable rationales for their classification decisions. Kanana, in particular, offered interpretations in a structured format, including an analysis overview, decision rationale, and conclusion.
* **Qwen:** Qwen, which recorded the lowest F1-Score, failed to comprehend the phishing context and exhibited **'Hallucination'**, generating inaccurate results and rationales.

---

## 🧑‍💻 Getting Started

### 1. Repository Clone
```bash
git clone [https://github.com/junhoeKu/Voice-Phishing-Detection](https://github.com/junhoeKu/Voice-Phishing-Detection)
cd Voice-Phishing-Detection
