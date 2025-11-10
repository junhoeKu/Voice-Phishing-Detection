"""
KoBERT 모델 학습 및 평가

KoBERT를 사용하여 보이스피싱 탐지 모델을 학습합니다.
5회 실행의 평균 성능을 계산하고 평가합니다.

사용 예시:
    python train/kobert_train.py --data dataset/total_dataset.csv
"""

import os
import random
import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback
)
from datasets import Dataset
import wandb


def set_seed(seed_value: int = 42):
    """난수 시드 고정"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_data(data_path: str, text_column: str = "text", label_column: str = "label"):
    """
    데이터 로드
    
    Args:
        data_path: 데이터 파일 경로 (상대경로)
        text_column: 텍스트 컬럼명
        label_column: 라벨 컬럼명
        
    Returns:
        (X, y) 튜플
    """
    base_path = Path(__file__).parent.parent
    full_path = base_path / data_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {full_path}")
    
    df = pd.read_csv(full_path)
    df = df.dropna(subset=[text_column, label_column])
    
    X = df[text_column].astype(str).tolist()
    y = df[label_column].astype(int).tolist()
    
    print(f"📊 전체 데이터: {len(df)}개")
    return X, y


def compute_metrics(eval_pred):
    """평가 지표 계산"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


def train_and_evaluate(
    X_all,
    y_all,
    num_runs: int = 5,
    learning_rate: float = 2e-5,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    epochs: int = 5,
    max_length: int = 256,
    early_stopping: int = 3,
    seed: int = 42,
    cache_dir: str = "cache",
    output_dir: str = "results/kobert",
    log_dir: str = "logs/kobert"
):
    """
    모델 학습 및 평가
    
    Args:
        X_all: 전체 텍스트 데이터
        y_all: 전체 라벨 데이터
        num_runs: 실행 횟수
        learning_rate: 학습률
        train_batch_size: 학습 배치 크기
        eval_batch_size: 평가 배치 크기
        epochs: 에포크 수
        max_length: 최대 길이
        early_stopping: 조기 종료 patience
        seed: 시드 값
        cache_dir: 캐시 디렉토리 (상대경로)
        output_dir: 출력 디렉토리 (상대경로)
        log_dir: 로그 디렉토리 (상대경로)
        
    Returns:
        (모델, 토크나이저, 평가_결과) 튜플
    """
    base_path = Path(__file__).parent.parent
    cache_full_path = base_path / cache_dir
    output_full_path = base_path / output_dir
    log_full_path = base_path / log_dir
    
    cache_full_path.mkdir(parents=True, exist_ok=True)
    output_full_path.mkdir(parents=True, exist_ok=True)
    log_full_path.mkdir(parents=True, exist_ok=True)
    
    val_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    test_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    
    last_model = None
    last_tokenizer = None
    
    for run_idx in range(num_runs):
        run_seed = seed + run_idx
        set_seed(run_seed)
        
        # 데이터 분할: 70% train, 30% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_all,
            y_all,
            test_size=0.30,
            random_state=run_seed,
            stratify=y_all
        )
        
        # temp을 15% val, 15% test로 분할
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=run_seed,
            stratify=y_temp
        )
        
        print(f"[Run {run_idx+1}] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # 데이터셋 준비
        train_ds = [{"text": x, "label": int(y)} for x, y in zip(X_train, y_train)]
        val_ds = [{"text": x, "label": int(y)} for x, y in zip(X_val, y_val)]
        test_ds = [{"text": x, "label": int(y)} for x, y in zip(X_test, y_test)]
        
        # 토크나이저 및 모델 로드 (매 run마다 새로 로드)
        tokenizer = BertTokenizer.from_pretrained("monologg/kobert", cache_dir=str(cache_full_path))
        model = BertForSequenceClassification.from_pretrained("monologg/kobert", num_labels=2)
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # 토큰화
        def tokenize_fn(batch):
            return tokenizer(
                batch["text"],
                max_length=max_length,
                truncation=True,
                padding="max_length"
            )
        
        train_dataset = Dataset.from_list(train_ds).map(tokenize_fn, batched=True)
        val_dataset = Dataset.from_list(val_ds).map(tokenize_fn, batched=True)
        test_dataset = Dataset.from_list(test_ds).map(tokenize_fn, batched=True)
        
        train_dataset = train_dataset.rename_column("label", "labels")
        val_dataset = val_dataset.rename_column("label", "labels")
        test_dataset = test_dataset.rename_column("label", "labels")
        
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_full_path),
            overwrite_output_dir=True,
            learning_rate=learning_rate,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            num_train_epochs=epochs,
            seed=run_seed,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_dir=str(log_full_path),
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to="wandb",
            disable_tqdm=True
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping)]
        )
        
        # 학습
        print(f"[Run {run_idx+1}] 학습 중...")
        trainer.train()
        
        # Validation 평가
        val_metrics_run = trainer.evaluate(val_dataset)
        print(f"[Run {run_idx+1}] Validation metrics: {val_metrics_run}")
        wandb.log({
            f"run_idx": run_idx + 1,
            **{f"val_{k}": v for k, v in val_metrics_run.items() if isinstance(v, float)}
        })
        
        for k in val_metrics.keys():
            val_metrics[k].append(val_metrics_run.get(f"eval_{k}", np.nan))
        
        # Test 평가
        test_metrics_run = trainer.evaluate(test_dataset)
        print(f"[Run {run_idx+1}] Test metrics: {test_metrics_run}")
        wandb.log({
            f"run_idx": run_idx + 1,
            **{f"test_{k}": v for k, v in test_metrics_run.items() if isinstance(v, float)}
        })
        
        for k in test_metrics.keys():
            test_metrics[k].append(test_metrics_run.get(f"eval_{k}", np.nan))
        
        # 마지막 run의 모델/토크나이저만 저장
        if run_idx == num_runs - 1:
            last_model = model
            last_tokenizer = tokenizer
    
    # 평균 결과 계산
    def _mean(v):
        return float(np.mean([x for x in v if x is not None and not np.isnan(x)])) if len(v) > 0 else float("nan")
    
    val_avg = {k: _mean(val_metrics[k]) for k in val_metrics}
    test_avg = {k: _mean(test_metrics[k]) for k in test_metrics}
    
    print("\n=== Validation Averages over 5 runs ===")
    print(f"Val Accuracy: {val_avg['accuracy']:.4f}, Val Precision: {val_avg['precision']:.4f}, Val Recall: {val_avg['recall']:.4f}, Val F1: {val_avg['f1']:.4f}")
    
    print("\n=== Test Averages over 5 runs ===")
    print(f"Test Accuracy: {test_avg['accuracy']:.4f}, Test Precision: {test_avg['precision']:.4f}, Test Recall: {test_avg['recall']:.4f}, Test F1: {test_avg['f1']:.4f}")
    
    # wandb summary에 평균 기록
    wandb.summary["val_accuracy_mean"] = val_avg["accuracy"]
    wandb.summary["val_precision_mean"] = val_avg["precision"]
    wandb.summary["val_recall_mean"] = val_avg["recall"]
    wandb.summary["val_f1_mean"] = val_avg["f1"]
    
    wandb.summary["test_accuracy_mean"] = test_avg["accuracy"]
    wandb.summary["test_precision_mean"] = test_avg["precision"]
    wandb.summary["test_recall_mean"] = test_avg["recall"]
    wandb.summary["test_f1_mean"] = test_avg["f1"]
    
    return last_model, last_tokenizer, {"val": val_avg, "test": test_avg}


def save_model(model, tokenizer, model_path: str, tokenizer_path: str = None):
    """
    모델 및 토크나이저 저장
    
    Args:
        model: 학습된 모델
        tokenizer: 토크나이저
        model_path: 모델 저장 경로 (상대경로)
        tokenizer_path: 토크나이저 저장 경로 (상대경로, None이면 자동 생성)
    """
    base_path = Path(__file__).parent.parent
    model_full_path = base_path / model_path
    model_full_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(str(model_full_path))
    print(f"💾 모델 저장: {model_full_path}")
    
    if tokenizer_path:
        tokenizer_full_path = base_path / tokenizer_path
        tokenizer_full_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(tokenizer_full_path))
        print(f"💾 토크나이저 저장: {tokenizer_full_path}")
    else:
        # 토크나이저도 같은 경로에 저장
        tokenizer.save_pretrained(str(model_full_path))
        print(f"💾 토크나이저 저장: {model_full_path}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="KoBERT 모델 학습 및 평가")
    parser.add_argument("--data", type=str, default="dataset/total_dataset.csv", help="데이터 파일 경로 (상대경로)")
    parser.add_argument("--model_path", type=str, default="model/model_kobert", help="모델 저장 경로 (상대경로)")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="토크나이저 저장 경로 (상대경로, None이면 model_path 사용)")
    parser.add_argument("--num_runs", type=int, default=5, help="실행 횟수")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="학습률")
    parser.add_argument("--train_batch_size", type=int, default=16, help="학습 배치 크기")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="평가 배치 크기")
    parser.add_argument("--epochs", type=int, default=5, help="에포크 수")
    parser.add_argument("--max_length", type=int, default=256, help="최대 길이")
    parser.add_argument("--early_stopping", type=int, default=3, help="조기 종료 patience")
    parser.add_argument("--seed", type=int, default=42, help="시드 값")
    parser.add_argument("--cache_dir", type=str, default="cache", help="캐시 디렉토리 (상대경로)")
    parser.add_argument("--output_dir", type=str, default="results/kobert", help="출력 디렉토리 (상대경로)")
    parser.add_argument("--log_dir", type=str, default="logs/kobert", help="로그 디렉토리 (상대경로)")
    parser.add_argument("--wandb_project", type=str, default="Voicephishing", help="WandB 프로젝트명")
    parser.add_argument("--wandb_name", type=str, default=None, help="WandB 실행명 (None이면 자동 생성)")
    args = parser.parse_args()
    
    wandb.init(project=args.wandb_project, name=args.wandb_name or f"kobert_{args.num_runs}runs", config={
        "model": "KoBERT", "learning_rate": args.learning_rate, "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size, "epochs": args.epochs, "max_length": args.max_length,
        "early_stopping": args.early_stopping, "num_runs": args.num_runs, "split_ratio": "train 0.70, val 0.15, test 0.15"
    })
    
    try:
        X_all, y_all = load_data(args.data)
        model, tokenizer, metrics = train_and_evaluate(
            X_all=X_all, y_all=y_all, num_runs=args.num_runs, learning_rate=args.learning_rate,
            train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size, epochs=args.epochs,
            max_length=args.max_length, early_stopping=args.early_stopping, seed=args.seed,
            cache_dir=args.cache_dir, output_dir=args.output_dir, log_dir=args.log_dir
        )
        if model is not None:
            save_model(model, tokenizer, args.model_path, args.tokenizer_path)
            try:
                wandb.save(str(Path(__file__).parent.parent / args.model_path))
            except Exception:
                pass
        print("\n✅ 학습 및 평가 완료!")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        raise
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()

