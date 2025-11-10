"""
모델 학습 스크립트

이 스크립트는 Synatra-42dot-1.3B 모델을 LoRA를 사용하여 파인튜닝합니다.
보이스피싱 탐지 태스크를 위한 시퀀스 분류 모델을 학습합니다.
"""

import os
import torch
import random
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    BitsAndBytesConfig, EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import HfFolder
import mlflow
from datasets import load_dataset
from pathlib import Path


def set_seed(seed_value=42):
    """난수 시드 고정"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True


def setup_environment(
    cache_dir: str = "cache",
    mlruns_dir: str = "mlruns",
    hf_token: str = ""
):
    """
    환경 변수 및 경로 설정
    
    Args:
        cache_dir: 캐시 디렉토리 경로 (상대경로)
        mlruns_dir: MLflow runs 디렉토리 경로 (상대경로)
        hf_token: Hugging Face 토큰
    """
    base_path = Path(__file__).parent.parent
    
    # 경로 설정
    cache_path = base_path / cache_dir
    mlruns_path = base_path / mlruns_dir
    
    # 디렉토리 생성
    cache_path.mkdir(parents=True, exist_ok=True)
    mlruns_path.mkdir(parents=True, exist_ok=True)
    
    # 환경 변수 설정
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_CACHE"] = str(cache_path)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_path)
    os.environ["TRITON_CACHE_DIR"] = str(cache_path / "triton")
    
    # GPU 설정
    gpu_count = torch.cuda.device_count()
    if gpu_count > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpu_count))
        print(f'Device: {torch.cuda.current_device()}')
        print(f'Using {torch.cuda.device_count()} GPUs')
    else:
        print('Using CPU')
    
    # Hugging Face 인증
    if hf_token:
        os.environ["HUGGINGFACE_TOKEN"] = hf_token
        HfFolder.save_token(hf_token)
    
    # MLflow 설정
    mlflow.set_tracking_uri(f"file:{mlruns_path}")
    mlflow.set_experiment("llm_experiment")
    
    return str(cache_path)


def load_model_and_tokenizer(
    model_name: str = "maywell/Synatra-42dot-1.3B",
    cache_dir: str = "cache",
    num_labels: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.2,
    target_modules: list = None
):
    """
    모델과 토크나이저 로드 및 LoRA 설정
    
    Args:
        model_name: 모델 이름
        cache_dir: 캐시 디렉토리 경로
        num_labels: 분류 레이블 수
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        target_modules: LoRA를 적용할 모듈 리스트
        
    Returns:
        (model, tokenizer) 튜플
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    
    base_path = Path(__file__).parent.parent
    cache_path = base_path / cache_dir
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(cache_path),
        use_fast=False,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # BitsAndBytes 설정 (4-bit 양자화)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # 모델 로드
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        num_labels=num_labels,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        cache_dir=str(cache_path),
    )
    
    # LoRA 설정
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules
    )
    
    # LoRA 적용
    model = get_peft_model(model, lora_config)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    
    return model, tokenizer


def tokenize_voicephishing_data(dataset, tokenizer, max_length=1024):
    """
    데이터셋 토큰화
    
    Args:
        dataset: 데이터셋
        tokenizer: 토크나이저
        max_length: 최대 시퀀스 길이
        
    Returns:
        토큰화된 데이터셋
    """
    def tokenize_fn(batch):
        tokenized = tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            add_special_tokens=True
        )
        tokenized["labels"] = batch["label"]
        return tokenized

    return dataset.map(tokenize_fn, batched=True, num_proc=4)


def fine_tune_model(
    train_data_path: str,
    val_data_path: str,
    model_save_path: str,
    output_dir: str = "results/voicephishing",
    log_dir: str = "logs/voicephishing",
    model_name: str = "maywell/Synatra-42dot-1.3B",
    cache_dir: str = "cache",
    num_labels: int = 2,
    learning_rate: float = 2e-5,
    num_train_epochs: int = 5,
    per_device_train_batch_size: int = 16,
    max_length: int = 1024,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.2,
    hf_token: str = ""
):
    """
    모델 파인튜닝 수행
    
    Args:
        train_data_path: 학습 데이터 CSV 파일 경로 (상대경로)
        val_data_path: 검증 데이터 CSV 파일 경로 (상대경로)
        model_save_path: 모델 저장 경로 (상대경로)
        output_dir: 출력 디렉토리 경로 (상대경로)
        log_dir: 로그 디렉토리 경로 (상대경로)
        model_name: 모델 이름
        cache_dir: 캐시 디렉토리 경로 (상대경로)
        num_labels: 분류 레이블 수
        learning_rate: 학습률
        num_train_epochs: 학습 에포크 수
        per_device_train_batch_size: 디바이스당 배치 크기
        max_length: 최대 시퀀스 길이
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        hf_token: Hugging Face 토큰
    """
    # 시드 고정
    set_seed(42)
    
    # 환경 설정
    cache_path = setup_environment(cache_dir=cache_dir, hf_token=hf_token)
    
    # 경로 설정 (상대경로)
    base_path = Path(__file__).parent.parent
    train_path = base_path / train_data_path
    val_path = base_path / val_data_path
    model_path = base_path / model_save_path
    output_path = base_path / output_dir
    log_path = base_path / log_dir
    
    # 디렉토리 생성
    model_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ 학습 데이터 로드: {train_path}")
    print(f"✅ 검증 데이터 로드: {val_path}")
    
    # 데이터셋 로드
    train_dataset = load_dataset("csv", data_files=str(train_path))["train"]
    val_dataset = load_dataset("csv", data_files=str(val_path))["train"]
    
    # 모델 및 토크나이저 로드
    print("📦 모델 및 토크나이저 로드 중...")
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        cache_dir=cache_dir,
        num_labels=num_labels,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    )
    
    # 데이터 토큰화
    print("🔄 데이터 토큰화 중...")
    tokenized_train = tokenize_voicephishing_data(train_dataset, tokenizer, max_length)
    tokenized_val = tokenize_voicephishing_data(val_dataset, tokenizer, max_length)
    
    # 학습 인자 설정
    training_args = TrainingArguments(
        output_dir=str(output_path),
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.2,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        save_steps=100,
        save_total_limit=3,
        logging_dir=str(log_path),
        logging_steps=100,
        optim="paged_adamw_32bit",
        max_grad_norm=5,
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False,
        bf16=True
    )
    
    # Trainer 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorWithPadding(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # 학습 시작
    print("🚀 학습 시작...")
    trainer.train()
    
    # 모델 저장
    print(f"💾 모델 저장: {model_path}")
    model.save_pretrained(str(model_path))
    
    print("✅ 학습 완료!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="보이스피싱 탐지 모델 학습")
    parser.add_argument("--train_data", type=str, default="dataset/bt_all_512_25.csv",
                        help="학습 데이터 CSV 파일 경로 (상대경로)")
    parser.add_argument("--val_data", type=str, default="dataset/total_dataset_val.csv",
                        help="검증 데이터 CSV 파일 경로 (상대경로)")
    parser.add_argument("--model_save", type=str, default="model/model_synatra_bt_all_512_25",
                        help="모델 저장 경로 (상대경로)")
    parser.add_argument("--output_dir", type=str, default="results/voicephishing",
                        help="출력 디렉토리 경로 (상대경로)")
    parser.add_argument("--log_dir", type=str, default="logs/voicephishing",
                        help="로그 디렉토리 경로 (상대경로)")
    parser.add_argument("--model_name", type=str, default="maywell/Synatra-42dot-1.3B",
                        help="모델 이름")
    parser.add_argument("--cache_dir", type=str, default="cache",
                        help="캐시 디렉토리 경로 (상대경로)")
    parser.add_argument("--num_labels", type=int, default=2,
                        help="분류 레이블 수")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="학습률")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="학습 에포크 수")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="디바이스당 배치 크기")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="최대 시퀀스 길이")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.2,
                        help="LoRA dropout")
    parser.add_argument("--hf_token", type=str, default="",
                        help="Hugging Face 토큰 (환경변수 HUGGINGFACE_TOKEN 사용 가능)")
    
    args = parser.parse_args()
    
    # 환경변수에서 토큰 가져오기
    hf_token = args.hf_token or os.getenv("HUGGINGFACE_TOKEN", "")
    
    fine_tune_model(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        model_save_path=args.model_save,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        num_labels=args.num_labels,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        hf_token=hf_token
    )

