# 🚀 고성능 STT 기반 데이터 활용 경량 언어 모델 (sLLM) 구축 프로젝트

이 프로젝트는 **Speech-to-Text (STT)** 데이터를 정교하게 처리하고 증강하여, **QLoRA 기반**의 **경량 언어 모델(sLLM)**을 효율적으로 정밀 튜닝하는 엔드투엔드(End-to-End) 파이프라인을 제시합니다. STT 데이터의 특성에 최적화된 데이터 구조화 및 증강 전략을 통해, 제한된 리소스 환경에서도 높은 성능을 발휘하는 **분류 레이어**를 갖춘 sLLM을 구현하는 것을 목표로 합니다.

---

## 🏗️ 아키텍처 파이프라인 (Architecture Pipeline)

다음 다이어그램은 본 프로젝트의 전체 워크플로우를 보여줍니다. 데이터 준비부터 학습 및 평가에 이르는 4단계 핵심 프로세스를 확인하세요.


### 1. Data Preparation (데이터 준비)
KorCCV1, FSS 데이터셋과 **Whisper-large-v3**를 활용하여 대규모 STT 데이터를 수집합니다.
* **전처리:** `data cleaning`, `stop word processing`, `missing value processing`을 통해 데이터의 품질을 극대화합니다.

### 2. Data Augmentation (데이터 증강)
준비된 학습 데이터셋에 **Back-Translation (BT)** 기법을 적용하여 모델의 일반화 성능을 높입니다.
* **BT-En, BT-Ia, BT-all** 등 다양한 백 트랜슬레이션 전략을 실험합니다.

### 3. Sentence Segmentation (문장 분할)
STT 데이터의 긴 문맥을 효과적으로 처리하고 문장 내 관계를 학습할 수 있도록 데이터를 구조화합니다.
* **Head & Tail 구조:** 문장의 시작과 끝 정보를 명시적으로 표시하여 컨텍스트를 확보합니다.
* **Sliding Window 기법:** 긴 텍스트를 중첩된 윈도우로 분할하여 중요한 정보를 놓치지 않도록 처리합니다.

### 4. Training & Evaluation (학습 및 평가)
구조화된 데이터를 바탕으로 sLLM을 정밀 튜닝하고, 정량적인 지표로 성능을 검증합니다.

---

## 🧠 sLLM 학습 및 최적화 (sLLM Training & Optimization)

본 프로젝트는 **경량화**와 **고성능**을 동시에 달성하기 위해 최신 LLM 튜닝 기법을 적용했습니다.

### QLoRA Fine-Tuning
* **기법:** **QLoRA (Quantized Low-Rank Adaptation)** 기법을 활용하여 메모리 사용량을 획기적으로 줄이면서도 sLLM을 효율적으로 학습시킵니다.
* **sLLM 모델:** Synatra, Kanaria, Owen 등 한국어 특화 및 경량화 모델을 기반으로 실험을 진행했습니다.
* **구조:** 모델의 최종 출력에 **Classification Layer**를 추가하여 STT 데이터 분류 목적에 맞게 모델을 전문화합니다.

## 📊 정량적 평가 (Quantitative Evaluation)

모델의 최종 성능은 다음의 지표들을 통해 엄격하게 평가됩니다.
* **핵심 지표:** **F1-Score** (정확도와 재현율의 균형)
* **세부 지표:** Accuracy, Precision, Recall

---

## 🧑‍💻 시작하기 (Getting Started)

프로젝트를 로컬 환경에서 실행하고 실험을 재현하는 방법입니다.

### 1. 저장소 클론
```bash
git clone [프로젝트_Github_URL]
cd [프로젝트_디렉토리]