# 🚀 Efficient Korean Voice-Phishing Detection using QLORA-tuned Small Language Models

## 💡 프로젝트 개요 (Project Overview)

보이스피싱 범죄의 지능화와 피해 규모 증가에 대응하여, 본 프로젝트는 **효율적이고 정확한** 자동 탐지 시스템 구축을 목표로 합니다. 기존 **LLM 기반 방법론**의 한계점인 막대한 연산 비용과 실제 데이터셋의 심각한 **클래스 불균형** 문제를 해결하기 위해, **경량 언어 모델(SLM)** 기반의 최적화된 학습 프레임워크를 제안합니다.

### 핵심 기여 3가지
* **1. 메모리 효율성 극대화:** Synatra, Kanana, Qwen 등 SLM에 **QLoRA** 기법을 적용하여 메모리 효율적인 정밀 튜닝을 수행합니다.
* **2. 데이터 불균형 해소:** **다국어 Back-Translation**을 통한 데이터 증강으로 불균형 문제를 완화하고 모델의 일반화 능력을 향상시킵니다.
* **3. 긴 문맥 처리:** SLM의 제한된 컨텍스트 길이를 극복하기 위해 **Overlapping Sliding Window** 기반의 텍스트 분할 전략을 도입합니다.

---

## 🏗️ 아키텍처 파이프라인 (Architecture Pipeline)

<img src="image/model_architecture.png" alt="파이프라인" width="750"/>

다음은 데이터 준비부터 학습 및 평가에 이르는 전체 워크플로우를 보여주는 다이어그램입니다.

### 1. 💾 데이터 준비 및 증강 (Data Preparation & Augmentation)

* **데이터셋 구축:** **KorCCVi** [cite: 104]와 **금융감독원(FSS)** 공개 오디오 파일을 **Whisper-large-v3**로 전사하여 한국어 보이스피싱 텍스트 데이터셋을 구축했습니다.
* **Back-Translation (BT):** 정상 대화 대비 현저히 부족한 피싱 데이터를 보강하기 위해, 한국어 텍스트를 **영어, 중국어, 일본어** 등으로 번역했다가 다시 한국어로 재번역하는 **다국어 BT**를 적용했습니다.
    > **결과:** **BT-ALL** 전략을 사용했을 때 모든 모델의 성능이 일관되게 향상되었으며, 특히 Qwen 모델의 F1-Score가 **0.2581에서 0.6213**으로 대폭 상승했습니다.

### 2. ✂️ 문장 분할 전략 (Text Segmentation Strategy)

SLM의 최대 입력 길이를 초과하는 긴 대화 텍스트 처리를 위해 세 가지 분할 전략을 비교했습니다.

| 전략 (Strategy) | 설명 (Description) | F1-Score (Synatra) |
| :--- | :--- | :--- |
| **Baseline** | 토크나이저의 기본 Truncation 기능 사용. | 0.9745 |
| **Head & Tail** | 피싱 핵심 정보가 앞/뒤에 집중된다는 가설 기반으로 분할. | 0.9734 |
| **Sliding Window (SW-512)** | **25% 중첩**을 가지는 고정 길이 청크로 분할하여 컨텍스트 손실 최소화. | **0.9875** |

* **최적 결과:** **Sliding Window (SW-512)** 전략이 가장 우수한 성능을 보였습니다. 이는 대화 중간 부분에 포함된 컨텍스트가 탐지에 중요한 역할을 하며, 단순한 시작/끝 부분 집중 가설과는 상반됨을 시사합니다.

---

## 🧠 sLLM 학습 및 최적화 (sLLM Training & Optimization)

### 모델 선정 및 QLoRA 적용
* **모델:** 디코더-온리 아키텍처, 파라미터 수 3B 미만, 한국어 이해 능력을 기준으로 **Synatra(1.3B)**, **Kanana(2.1B)**, **Qwen (0.5B)** 모델을 선정했습니다.
* **QLoRA:** **4-bit 정밀도**로 사전 학습된 모델 가중치를 양자화하고, **LoRA** 모듈을 주요 어텐션 모듈에 적용하여 학습 메모리 사용량을 대폭 절감했습니다.

---

## 📈 주요 정량적/정성적 결과 (Key Results)

### 1. 정량적 성능 비교 (Quantitative Performance)

제안된 통합 프레임워크 (**BT-ALL** 및 **SW-512** 적용)는 기존 머신러닝 및 PLM(KOBERT) 기반 모델들을 크게 능가하는 성능을 입증했습니다.

| Category | Model | F1-Score |
| :--- | :--- | :--- |
| Proposed SLLM | **Synatra** | **0.9938** |
| Proposed SLLM | **Kanana** | **0.9938** |
| PLM (Baseline) | KOBERT | 0.6433 |
| ML Models | Random Forest | 0.9835 |

### 2. 정성적 해석 가능성 (Qualitative Interpretability)

SLM 채택의 주요 동기인 **해석 가능성(Interpretability)**을 검증했습니다.
* **Synatra/Kanana:** 높은 F1-Score에 부합하게 **'기관 사칭'**, **'긴급 유도'** 등 피싱의 핵심 패턴을 정확하게 식별하고, 분류 결정에 대한 합리적인 근거를 제공했습니다. 특히 Kanana는 분석 개요, 결정 근거, 결론을 포함하는 구조화된 형식으로 해석을 제공했습니다.
* **Qwen:** 가장 낮은 F1-Score를 기록한 Qwen은 피싱 컨텍스트를 이해하지 못하고 **'Hallucination'** 현상을 보이며 부정확한 결과와 근거를 생성했습니다.


---

## 🧑‍💻 시작하기 (Getting Started)

### 1. 저장소 클론
'''bash
git clone [https://github.com/junhoeKu/Voice-Phishing-Detection](https://github.com/junhoeKu/Voice-Phishing-Detection) # 공식 저장소 URL
cd Voice-Phishing-Detection
'''