"""
모델 평가 모듈

학습된 모델을 사용하여 테스트 데이터셋을 평가하고 성능 지표를 계산합니다.
정확도, 정밀도, 재현율, F1 점수를 계산하여 저장합니다.

사용 예시:
    python eval/eval.py --test_data dataset/test.csv --adapter_path model/my_adapter
"""

import os
import torch
import random
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import load_dataset


def set_seed(seed: int = 42):
    """
    난수 시드 고정
    
    Args:
        seed: 시드 값 (기본값: 42)
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_environment(cache_dir: str = "cache", hf_cache_dir: str = None):
    """
    환경 변수 설정
    
    Args:
        cache_dir: 캐시 디렉토리 경로 (상대경로)
        hf_cache_dir: Hugging Face 캐시 디렉토리 경로 (상대경로, None이면 cache_dir 사용)
    """
    base_path = Path(__file__).parent.parent
    cache_path = base_path / cache_dir
    cache_path.mkdir(parents=True, exist_ok=True)
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_CACHE"] = str(cache_path)
    os.environ["TRITON_CACHE_DIR"] = str(cache_path / "triton")
    
    if hf_cache_dir:
        hf_cache_path = base_path / hf_cache_dir
        hf_cache_path.mkdir(parents=True, exist_ok=True)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache_path)
    else:
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_path / "hf_cache")


def preprocess_input(text: str) -> str:
    """
    입력 텍스트 전처리
    
    Args:
        text: 입력 텍스트
        
    Returns:
        전처리된 텍스트
    """
    return " ".join(text.strip().split())


def load_model_and_tokenizer(
    base_model_path: str,
    adapter_path: str,
    tokenizer_path: str = None,
    cache_dir: str = "cache",
    device: str = None,
    merge_adapter: bool = True
):
    """
    모델과 토크나이저 로드
    
    Args:
        base_model_path: 기본 모델 경로
        adapter_path: 어댑터 경로 (상대경로)
        tokenizer_path: 토크나이저 경로 (상대경로, None이면 base_model_path 사용)
        cache_dir: 캐시 디렉토리 경로 (상대경로)
        device: 디바이스 (None이면 자동 선택)
        merge_adapter: 어댑터 병합 여부
        
    Returns:
        (model, tokenizer) 튜플
    """
    base_path = Path(__file__).parent.parent
    adapter_full_path = base_path / adapter_path
    cache_full_path = base_path / cache_dir
    
    # 디바이스 설정
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 토크나이저 경로 설정
    if tokenizer_path is None:
        tokenizer_path = base_model_path
    else:
        tokenizer_full_path = base_path / tokenizer_path
        if tokenizer_full_path.exists():
            tokenizer_path = str(tokenizer_full_path)
    
    print(f"📥 토크나이저 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        cache_dir=str(cache_full_path)
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ 토크나이저 로드 완료")
    
    # 모델 로드
    print(f"📥 기본 모델 로딩 중... ({base_model_path})")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
    
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=2,
        torch_dtype=torch_dtype,
        cache_dir=str(cache_full_path),
        low_cpu_mem_usage=True,
        device_map=device if device.startswith("cuda") else None
    )
    
    # PEFT 어댑터 로드
    if not adapter_full_path.exists():
        raise FileNotFoundError(f"어댑터 경로를 찾을 수 없습니다: {adapter_full_path}")
    
    print(f"📥 어댑터 로딩 중... ({adapter_path})")
    model = PeftModel.from_pretrained(base_model, str(adapter_full_path))
    
    # 어댑터 병합
    if merge_adapter:
        print("🔗 어댑터 병합 중...")
        model = model.merge_and_unload()
    
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    print("✅ 모델 로드 완료")
    
    return model, tokenizer


def evaluate_model(
    model,
    tokenizer,
    test_data_path: str,
    max_length: int = 1024,
    label_map: dict = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    모델 평가 수행
    
    Args:
        model: 평가할 모델
        tokenizer: 토크나이저
        test_data_path: 테스트 데이터 경로 (상대경로)
        max_length: 최대 입력 길이 (기본값: 1024)
        label_map: 라벨 매핑 딕셔너리 (기본값: {0: "정상", 1: "보이스피싱"})
        
    Returns:
        (예측_결과_DataFrame, 평가_지표_DataFrame) 튜플
    """
    base_path = Path(__file__).parent.parent
    test_data_full_path = base_path / test_data_path
    
    if not test_data_full_path.exists():
        raise FileNotFoundError(f"테스트 데이터를 찾을 수 없습니다: {test_data_full_path}")
    
    # 라벨 매핑 설정
    if label_map is None:
        label_map = {0: "정상", 1: "보이스피싱"}
    
    inv_label_map = {v: k for k, v in label_map.items()}
    
    # 테스트 데이터셋 로드
    print(f"📂 테스트 데이터 로딩 중... ({test_data_path})")
    test_dataset = load_dataset("csv", data_files=str(test_data_full_path))["train"]
    print(f"📊 테스트 데이터: {len(test_dataset)}개")
    
    records = []
    y_true = []
    y_pred = []
    
    # 평가 수행
    print("🧪 평가 수행 중...")
    for sample in tqdm(test_dataset, desc="평가 진행"):
        input_text = sample["text"]
        
        # 라벨 컬럼명 확인 (Label 또는 label)
        label_key = "Label" if "Label" in sample else "label"
        target_label_idx = sample[label_key]
        target_label = label_map[target_label_idx]
        
        # 전처리 및 토크나이징
        preprocessed_text = preprocess_input(input_text)
        inputs = tokenizer(
            preprocessed_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_token_type_ids=False
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 예측
        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_label_idx = torch.argmax(logits, dim=-1).item()
        
        pred_text = label_map[predicted_label_idx]
        is_correct = (pred_text == target_label)
        
        y_true.append(target_label_idx)
        y_pred.append(predicted_label_idx)
        
        records.append({
            "Input": input_text,
            "Prediction": pred_text,
            "Label": target_label,
            "Correct": int(is_correct)
        })
    
    # 지표 계산
    print("📐 평가 지표 계산 중...")
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # 결과 DataFrame 생성
    results_df = pd.DataFrame(records)
    scores_df = pd.DataFrame([{
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }])
    
    return results_df, scores_df


def save_results(
    results_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    results_path: str,
    scores_path: str
):
    """
    평가 결과 저장
    
    Args:
        results_df: 예측 결과 DataFrame
        scores_df: 평가 지표 DataFrame
        results_path: 예측 결과 저장 경로 (상대경로)
        scores_path: 평가 지표 저장 경로 (상대경로)
    """
    base_path = Path(__file__).parent.parent
    results_full_path = base_path / results_path
    scores_full_path = base_path / scores_path
    
    # 디렉토리 생성
    results_full_path.parent.mkdir(parents=True, exist_ok=True)
    scores_full_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 저장
    results_df.to_csv(results_full_path, index=False, encoding="utf-8-sig")
    scores_df.to_csv(scores_full_path, index=False, encoding="utf-8-sig")
    
    print(f"💾 예측 결과 저장: {results_full_path}")
    print(f"💾 평가 지표 저장: {scores_full_path}")


def evaluate_and_save(
    base_model_path: str,
    adapter_path: str,
    test_data_path: str,
    tokenizer_path: str = None,
    results_path: str = "results/eval_results.csv",
    scores_path: str = "results/eval_metrics.csv",
    cache_dir: str = "cache",
    device: str = None,
    merge_adapter: bool = True,
    max_length: int = 1024
):
    """
    모델 평가 및 결과 저장
    
    Args:
        base_model_path: 기본 모델 경로
        adapter_path: 어댑터 경로 (상대경로)
        test_data_path: 테스트 데이터 경로 (상대경로)
        tokenizer_path: 토크나이저 경로 (상대경로, None이면 base_model_path 사용)
        results_path: 예측 결과 저장 경로 (상대경로)
        scores_path: 평가 지표 저장 경로 (상대경로)
        cache_dir: 캐시 디렉토리 경로 (상대경로)
        device: 디바이스 (None이면 자동 선택)
        merge_adapter: 어댑터 병합 여부
        max_length: 최대 입력 길이
    """
    # 환경 설정
    setup_environment(cache_dir)
    set_seed(42)
    
    # 모델 및 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer(
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        tokenizer_path=tokenizer_path,
        cache_dir=cache_dir,
        device=device,
        merge_adapter=merge_adapter
    )
    
    # 평가 수행
    results_df, scores_df = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_data_path=test_data_path,
        max_length=max_length
    )
    
    # 결과 출력
    print("\n✅ 평가 지표:")
    print(scores_df.to_string(index=False, float_format="%.4f"))
    
    # 결과 저장
    save_results(results_df, scores_df, results_path, scores_path)
    
    print("\n✅ 평가 완료!")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="모델 평가")
    parser.add_argument("--base_model", type=str, default="maywell/Synatra-42dot-1.3B", help="기본 모델 경로")
    parser.add_argument("--adapter_path", type=str, required=True, help="어댑터 경로 (상대경로)")
    parser.add_argument("--test_data", type=str, required=True, help="테스트 데이터 경로 (상대경로)")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="토크나이저 경로 (상대경로, None이면 base_model 사용)")
    parser.add_argument("--results_path", type=str, default="results/eval_results.csv", help="예측 결과 저장 경로 (상대경로)")
    parser.add_argument("--scores_path", type=str, default="results/eval_metrics.csv", help="평가 지표 저장 경로 (상대경로)")
    parser.add_argument("--cache_dir", type=str, default="cache", help="캐시 디렉토리 경로 (상대경로)")
    parser.add_argument("--device", type=str, default=None, help="디바이스 (cuda/cpu, None이면 자동 선택)")
    parser.add_argument("--no_merge", action="store_true", help="어댑터 병합하지 않기")
    parser.add_argument("--max_length", type=int, default=1024, help="최대 입력 길이")
    args = parser.parse_args()
    
    evaluate_and_save(
        base_model_path=args.base_model, adapter_path=args.adapter_path, test_data_path=args.test_data,
        tokenizer_path=args.tokenizer_path, results_path=args.results_path, scores_path=args.scores_path,
        cache_dir=args.cache_dir, device=args.device, merge_adapter=not args.no_merge, max_length=args.max_length
    )


if __name__ == "__main__":
    main()

