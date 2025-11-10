"""
보이스피싱 탐지 추론 모듈

학습된 분류기 모델과 생성기 모델을 사용하여 보이스피싱을 탐지하고 분석합니다.
Gradio 인터페이스를 통해 웹 UI를 제공합니다.

사용 예시:
    python inference/inference.py --adapter_path model/model_qwen_cls_bt_all_512_25
"""

import os
import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel
from huggingface_hub import HfFolder
import gradio as gr


def setup_environment(cache_dir: str = "cache", hf_token: str = ""):
    """
    환경 변수 설정
    
    Args:
        cache_dir: 캐시 디렉토리 경로 (상대경로)
        hf_token: Hugging Face 토큰
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    base_path = Path(__file__).parent.parent
    cache_path = base_path / cache_dir
    cache_path.mkdir(parents=True, exist_ok=True)
    
    os.environ["TRANSFORMERS_CACHE"] = str(cache_path)
    
    if hf_token:
        os.environ["HUGGINGFACE_TOKEN"] = hf_token
        HfFolder.save_token(hf_token)


def load_models(
    base_model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
    adapter_path: str = "model/model_qwen_cls_bt_all_512_25",
    cache_dir: str = "cache",
    device: str = None,
    merge_adapter: bool = True
):
    """
    분류기와 생성기 모델 로드
    
    Args:
        base_model_path: 기본 모델 경로
        adapter_path: 어댑터 경로 (상대경로)
        cache_dir: 캐시 디렉토리 경로 (상대경로)
        device: 디바이스 (None이면 자동 선택)
        merge_adapter: 어댑터 병합 여부
        
    Returns:
        (classifier_model, generator_model, tokenizer) 튜플
    """
    base_path = Path(__file__).parent.parent
    adapter_full_path = base_path / adapter_path
    cache_full_path = base_path / cache_dir
    
    # 디바이스 설정
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    print(f"📱 디바이스: {device}")
    print(f"💾 캐시 디렉토리: {cache_full_path}")
    
    # 1. 분류기 모델 로드
    print("📥 분류기 모델 로딩 중...")
    classifier_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=2,
        torch_dtype=torch_dtype,
        cache_dir=str(cache_full_path),
        low_cpu_mem_usage=True
    )
    
    # PEFT 어댑터 로드
    if not adapter_full_path.exists():
        raise FileNotFoundError(f"어댑터 경로를 찾을 수 없습니다: {adapter_full_path}")
    
    classifier_model = PeftModel.from_pretrained(classifier_model, str(adapter_full_path))
    
    # 어댑터 병합 (선택적)
    if merge_adapter:
        print("🔗 어댑터 병합 중...")
        classifier_model = classifier_model.merge_and_unload()
    
    classifier_model.eval()
    classifier_model.to(device)
    print("✅ 분류기 모델 로드 완료")
    
    # 2. 생성기 모델 로드
    print("📥 생성기 모델 로딩 중...")
    generator_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        cache_dir=str(cache_full_path),
        low_cpu_mem_usage=True
    )
    generator_model.eval()
    generator_model.to(device)
    print("✅ 생성기 모델 로드 완료")
    
    # 3. 토크나이저 로드
    print("📥 토크나이저 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        cache_dir=str(cache_full_path),
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # generate를 위해 padding side를 left로 설정
    print("✅ 토크나이저 로드 완료")
    
    return classifier_model, generator_model, tokenizer


def preprocess_input(text: str) -> str:
    """
    입력 텍스트 전처리
    
    Args:
        text: 입력 텍스트
        
    Returns:
        전처리된 텍스트
    """
    return " ".join(text.strip().split())


def generate_response(
    prompt: str,
    classifier_model,
    generator_model,
    tokenizer,
    device: str,
    max_length: int = 1024,
    max_new_tokens: int = 200,
    min_new_tokens: int = 30
) -> str:
    """
    보이스피싱 탐지 및 분석 수행
    
    Args:
        prompt: 입력 텍스트
        classifier_model: 분류기 모델
        generator_model: 생성기 모델
        tokenizer: 토크나이저
        device: 디바이스
        max_length: 최대 입력 길이
        max_new_tokens: 최대 생성 토큰 수
        min_new_tokens: 최소 생성 토큰 수
        
    Returns:
        분석 결과 (Markdown 형식)
    """
    # 1단계: 분류 수행
    processed_prompt = preprocess_input(prompt)
    inputs = tokenizer(
        processed_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = classifier_model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)
        prediction_idx = torch.argmax(probabilities, dim=1).item()
    
    if prediction_idx == 1:
        classification_result = "보이스피싱"
        confidence = probabilities[0][1].item()
    else:
        classification_result = "정상 대화"
        confidence = probabilities[0][0].item()
    
    # 2단계: 분석 보고서 생성
    role = "당신은 보이스피싱 텍스트를 분석하는 AI 수사관입니다."
    context = f"다음 텍스트는 1차 분석 결과 '{classification_result}'로 판별되었습니다."
    task = f"이 텍스트를 정밀 분석하여, '{classification_result}' 판별이 타당한지 검증하고, 그 핵심 근거를 '보이스피싱 패턴'(기관 사칭, 긴급성, 금전/정보 요구 등)에 기반하여 설명하세요."
    output_format = "반드시 [판단 근거] 3개와 [결론]으로 구성된 보고서 형식으로 답변하고, 정상 대화라면 정상 대화인 이유를 분석해서 작성하세요."
    
    reasoning_prompt = f"""{role}
{context}

[임무]
{task}
{output_format}

[분석 대상 텍스트]
{processed_prompt}

[분석 보고서]
"""
    
    # 생성기 토크나이징
    gen_inputs = tokenizer(reasoning_prompt, return_tensors="pt").to(device)
    
    # 텍스트 생성
    gen_outputs = generator_model.generate(
        **gen_inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        num_beams=3,
        do_sample=True,
        top_p=0.8,
        temperature=0.7,
        no_repeat_ngram_size=3,
        repetition_penalty=1.5,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # 생성된 텍스트 디코딩
    generated_text = tokenizer.decode(
        gen_outputs[0][gen_inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()
    
    # 결과 포맷팅
    final_output = f"## 1. 분석 결과:\n"
    final_output += f"**{classification_result}** (신뢰도: {confidence:.2%})\n\n"
    final_output += f"## 2. 분석 보고서 (AI 수사관):\n"
    final_output += f"{generated_text}"
    
    return final_output


def create_gradio_interface(
    classifier_model,
    generator_model,
    tokenizer,
    device: str
):
    """
    Gradio 인터페이스 생성
    
    Args:
        classifier_model: 분류기 모델
        generator_model: 생성기 모델
        tokenizer: 토크나이저
        device: 디바이스
        
    Returns:
        Gradio 인터페이스
    """
    def inference_fn(prompt: str) -> str:
        """Gradio용 추론 함수"""
        if not prompt or not prompt.strip():
            return "⚠️ 텍스트를 입력해주세요."
        
        try:
            return generate_response(
                prompt,
                classifier_model,
                generator_model,
                tokenizer,
                device
            )
        except Exception as e:
            return f"❌ 오류 발생: {str(e)}"
    
    interface = gr.Interface(
        fn=inference_fn,
        inputs=[gr.Textbox(
            lines=10,
            placeholder="분석할 대화를 입력하세요...",
            label="대화 내용"
        )],
        outputs=gr.Markdown(label="분석 결과"),
        title="보이스피싱 탐지 AI (분류 및 분석)",
        description="입력한 대화를 분석하여 보이스피싱 여부를 '분류'하고, '이유'를 생성합니다.",
        examples=[
            "저는 서울중앙지검 첨단범죄수사 1팀 김상수 수사관입니다. 본인 맞으십니까?",
            "이제 알바를 구하려다가 그 메가스터디 러셀학원이라고 있는데, 안양에요."
        ]
    )
    
    return interface


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="보이스피싱 탐지 추론")
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="기본 모델 경로"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="model/model_qwen_cls_bt_all_512_25",
        help="어댑터 경로 (상대경로)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache",
        help="캐시 디렉토리 경로 (상대경로)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="디바이스 (cuda/cpu, None이면 자동 선택)"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default="",
        help="Hugging Face 토큰 (환경변수 HUGGINGFACE_TOKEN에서도 읽을 수 있음)"
    )
    parser.add_argument(
        "--no_merge",
        action="store_true",
        help="어댑터 병합하지 않기"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Gradio 공유 링크 생성"
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default="127.0.0.1",
        help="서버 주소"
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=7860,
        help="서버 포트"
    )
    
    args = parser.parse_args()
    
    # Hugging Face 토큰 설정 (환경변수 우선)
    hf_token = os.getenv("HUGGINGFACE_TOKEN", args.hf_token)
    
    # 환경 설정
    setup_environment(args.cache_dir, hf_token)
    
    # 모델 로드
    classifier_model, generator_model, tokenizer = load_models(
        base_model_path=args.base_model,
        adapter_path=args.adapter_path,
        cache_dir=args.cache_dir,
        device=args.device,
        merge_adapter=not args.no_merge
    )
    
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Gradio 인터페이스 생성 및 실행
    print("🚀 Gradio 인터페이스 시작 중...")
    interface = create_gradio_interface(
        classifier_model,
        generator_model,
        tokenizer,
        device
    )
    
    interface.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port
    )


if __name__ == "__main__":
    main()

