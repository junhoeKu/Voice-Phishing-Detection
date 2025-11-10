"""
LLM 추론 시간 측정 모듈

여러 LLM 모델의 추론 시간을 측정하고 벤치마크를 수행합니다.
메모리 효율을 위해 각 모델 측정 후 메모리를 정리합니다.

사용 예시:
    python inference/inference_time.py --prompt "안녕하세요" --max_new_tokens 100
    python inference/inference_time.py --models Qwen/Qwen2.5-0.5B-Instruct maywell/Synatra-42dot-1.3B
"""

import os
import gc
import time
import torch
import argparse
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_environment(cache_dir: str = "cache"):
    """
    환경 변수 설정
    
    Args:
        cache_dir: 캐시 디렉토리 경로 (상대경로)
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    base_path = Path(__file__).parent.parent
    cache_path = base_path / cache_dir
    cache_path.mkdir(parents=True, exist_ok=True)
    
    os.environ["TRANSFORMERS_CACHE"] = str(cache_path)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_path)


def measure_inference_time(
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    cache_dir: str = "cache",
    device: str = None,
    warmup_tokens: int = 10
) -> float:
    """
    지정된 모델 ID로 모델과 토크나이저를 로드하고, 워밍업 후 순수 추론 시간을 측정합니다.
    
    Args:
        model_id: 모델 ID (Hugging Face 모델명)
        prompt: 입력 프롬프트
        max_new_tokens: 최대 생성 토큰 수
        cache_dir: 캐시 디렉토리 경로 (상대경로)
        device: 디바이스 (None이면 자동 선택)
        warmup_tokens: 워밍업 시 생성할 토큰 수
        
    Returns:
        추론 시간 (초), 실패 시 -1.0
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_path = Path(__file__).parent.parent
    cache_path = base_path / cache_dir
    
    print(f"\n[모델 로딩]: {model_id}")
    model = None
    tokenizer = None
    
    try:
        # 1. 모델 및 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=str(cache_path))
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            cache_dir=str(cache_path)
        ).to(device)
        model.eval()
        
        # 2. 패딩 토큰 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 3. 입력 준비
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # 4. 워밍업 (Warm-up)
        print("... 워밍업 1회 실행 중 ...")
        _ = model.generate(
            **inputs,
            max_new_tokens=warmup_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False
        )
        
        # 5. 실제 추론 시간 측정
        if device == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False
        )
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # 6. 결과 디코딩
        output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        print(f"  [생성 결과]: ...{output_text.strip()[:50]}")
        print(f"  [추론 시간]: {duration:.4f} 초")
        
        return duration
        
    except Exception as e:
        print(f"  [오류 발생]: {model_id} 처리 중 오류 - {e}")
        # VRAM 부족(OOM) 오류가 가장 흔한 원인입니다.
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            print("  [원인]: 🔴 GPU VRAM(메모리) 부족 (OOM) 🔴")
        return -1.0
        
    finally:
        # 메모리 확보를 위해 모델과 토크나이저를 메모리에서 명시적으로 해제
        del model, tokenizer
        gc.collect()  # 가비지 컬렉션 강제 실행
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"[메모리 해제]: {model_id} 완료")


def run_benchmark(
    model_ids: list,
    prompt: str,
    max_new_tokens: int,
    cache_dir: str = "cache",
    device: str = None,
    warmup_tokens: int = 10
) -> dict:
    """
    여러 모델에 대해 벤치마크를 실행합니다.
    
    Args:
        model_ids: 테스트할 모델 ID 리스트
        prompt: 입력 프롬프트
        max_new_tokens: 최대 생성 토큰 수
        cache_dir: 캐시 디렉토리 경로 (상대경로)
        device: 디바이스 (None이면 자동 선택)
        warmup_tokens: 워밍업 시 생성할 토큰 수
        
    Returns:
        {모델_ID: 추론시간} 딕셔너리
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"🚀 디바이스: {device}")
    print(f"⚡ 데이터 타입: {torch_dtype}")
    print("-------------------------------------------------")
    
    results = {}
    print("======== LLM 추론 속도 벤치마크 시작 ========")
    
    for model_id in model_ids:
        duration = measure_inference_time(
            model_id=model_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            cache_dir=cache_dir,
            device=device,
            warmup_tokens=warmup_tokens
        )
        results[model_id] = duration
        print("-" * 40)
    
    return results


def print_results(results: dict, prompt: str, max_new_tokens: int, device: str):
    """
    결과를 출력합니다.
    
    Args:
        results: {모델_ID: 추론시간} 딕셔너리
        prompt: 입력 프롬프트
        max_new_tokens: 최대 생성 토큰 수
        device: 디바이스
    """
    print("\n" + "="*40)
    print("          📊 최종 비교 결과 📊")
    print("="*40)
    print(f"(Prompt: '{prompt}')")
    print(f"(Max New Tokens: {max_new_tokens}, Device: {device.upper()})\n")
    
    for model_id, duration in results.items():
        if duration == -1.0:
            print(f"모델: {model_id}\n  결과: ❌ 로드 또는 추론 실패 (OOM 가능성)\n")
        else:
            print(f"모델: {model_id}\n  시간: {duration:.4f} 초\n")
    
    print("="*40)
    print("⚠️ 7B 이상 모델은 T4 환경에서 VRAM 부족(OOM)으로")
    print("   실패했을 가능성이 높습니다. (T4 VRAM: 약 15GB)")
    print("   더 큰 GPU(A100/H100)가 있는 환경이 필요할 수 있습니다.")


def save_results(results: dict, prompt: str, max_new_tokens: int, device: str, output_path: str):
    """
    결과를 CSV 파일로 저장합니다.
    
    Args:
        results: {모델_ID: 추론시간} 딕셔너리
        prompt: 입력 프롬프트
        max_new_tokens: 최대 생성 토큰 수
        device: 디바이스
        output_path: 출력 파일 경로 (상대경로)
    """
    base_path = Path(__file__).parent.parent
    output_full_path = base_path / output_path
    output_full_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame([
        {
            "model_id": model_id,
            "inference_time": duration if duration != -1.0 else None,
            "status": "success" if duration != -1.0 else "failed",
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "device": device
        }
        for model_id, duration in results.items()
    ])
    
    df.to_csv(output_full_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 결과 저장: {output_full_path}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="LLM 추론 시간 측정 벤치마크")
    parser.add_argument("--prompt", type=str, default="안녕하세요, 저는 보이스피싱 탐지 AI 모델", help="입력 프롬프트")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="최대 생성 토큰 수")
    parser.add_argument("--models", type=str, nargs="+", default=[
        "Qwen/Qwen2.5-0.5B-Instruct",
        "maywell/Synatra-42dot-1.3B",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct"
    ], help="테스트할 모델 ID 리스트")
    parser.add_argument("--cache_dir", type=str, default="cache", help="캐시 디렉토리 경로 (상대경로)")
    parser.add_argument("--device", type=str, default=None, help="디바이스 (cuda/cpu, None이면 자동 선택)")
    parser.add_argument("--warmup_tokens", type=int, default=10, help="워밍업 시 생성할 토큰 수")
    parser.add_argument("--output", type=str, default="results/inference_time_results.csv", help="결과 저장 경로 (상대경로)")
    args = parser.parse_args()
    
    setup_environment(args.cache_dir)
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    
    results = run_benchmark(
        model_ids=args.models,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        cache_dir=args.cache_dir,
        device=device,
        warmup_tokens=args.warmup_tokens
    )
    
    print_results(results, args.prompt, args.max_new_tokens, device)
    
    if args.output:
        save_results(results, args.prompt, args.max_new_tokens, device, args.output)


if __name__ == "__main__":
    main()

