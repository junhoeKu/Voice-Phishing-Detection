import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from pydub import AudioSegment
from transformers import WhisperProcessor, WhisperForConditionalGeneration


# 상수 설정
MAX_DURATION_SEC = 30  # Whisper는 30초(=480000 samples) 기준
SAMPLE_RATE = 16000
CHUNK_SIZE = MAX_DURATION_SEC * SAMPLE_RATE


def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    오디오 파일 로드
    
    Args:
        path: 오디오 파일 경로
        target_sr: 목표 샘플링 레이트 (기본값: 16000)
        
    Returns:
        오디오 웨이브폼 텐서
    """
    ext = os.path.splitext(path)[1][1:]
    audio = AudioSegment.from_file(path, format=ext)
    audio = audio.set_channels(1).set_frame_rate(target_sr)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    waveform = torch.tensor(samples)
    return waveform


def transcribe_long_audio(
    path: str,
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    device: str,
    sample_rate: int = 16000,
    chunk_size: int = CHUNK_SIZE
) -> str:
    """
    긴 오디오 파일을 청크로 나누어 STT 수행
    
    Args:
        path: 오디오 파일 경로
        processor: Whisper 프로세서
        model: Whisper 모델
        device: 디바이스 (cuda/cpu)
        sample_rate: 샘플링 레이트 (기본값: 16000)
        chunk_size: 청크 크기 (기본값: 30초)
        
    Returns:
        전체 전사 텍스트
    """
    audio = load_audio(path, sample_rate)
    total_len = len(audio)
    results = []
    
    num_chunks = (total_len + chunk_size - 1) // chunk_size
    
    for i in tqdm(range(0, total_len, chunk_size), total=num_chunks, desc=f"🧠 STT: {os.path.basename(path)}"):
        chunk = audio[i:i+chunk_size]
        input_features = processor(chunk, sampling_rate=sample_rate, return_tensors="pt").input_features.to(device)
        
        outputs = model.generate(
            input_features,
            num_beams=1,
            num_return_sequences=1,
            output_scores=False,
            return_dict_in_generate=True,
            early_stopping=False,  # num_beams=1일 때 early_stopping은 의미 없음
        )
        
        decoded = processor.batch_decode(outputs.sequences, skip_special_tokens=True)
        results.append(decoded[0].strip())
    
    return " ".join(results)


def transcribe_folder(
    folder_path: str,
    output_path: str = "transcriptions.csv",
    model_name: str = "openai/whisper-large-v3",
    device: str = None,
    file_extensions: list = None
) -> pd.DataFrame:
    """
    폴더 내 음성 파일들을 STT 처리하여 CSV 파일로 저장
    
    Args:
        folder_path: 입력 폴더 경로 (상대경로)
        output_path: 출력 CSV 파일 경로 (상대경로)
        model_name: Whisper 모델명 (기본값: "openai/whisper-large-v3")
        device: 디바이스 (None이면 자동 선택)
        file_extensions: 처리할 파일 확장자 리스트 (기본값: [".mp4", ".wav", ".mp3", ".m4a"])
        
    Returns:
        전사 결과 DataFrame
    """
    base_path = Path(__file__).parent.parent
    folder_full_path = base_path / folder_path
    output_full_path = base_path / output_path
    
    if not folder_full_path.exists():
        raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {folder_full_path}")
    
    # 디바이스 설정
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 파일 확장자 설정
    if file_extensions is None:
        file_extensions = [".mp4", ".wav", ".mp3", ".m4a", ".flac", ".ogg"]
    
    print(f"📂 폴더: {folder_full_path}")
    print(f"📱 디바이스: {device}")
    print(f"🤖 모델: {model_name}")
    
    # 모델 로드
    print("📥 Whisper 모델 로딩 중...")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    print("✅ 모델 로드 완료")
    
    # 파일 목록 생성
    file_list = sorted([
        f for f in os.listdir(folder_full_path)
        if any(f.lower().endswith(ext.lower()) for ext in file_extensions)
    ])
    
    if not file_list:
        print(f"⚠️ 경고: {folder_full_path}에 처리할 파일이 없습니다.")
        return pd.DataFrame(columns=["index", "filename", "transcription", "text_length"])
    
    print(f"📊 처리할 파일: {len(file_list)}개")
    
    # STT 처리
    data = []
    for idx, file in enumerate(file_list):
        full_path = folder_full_path / file
        try:
            transcription = transcribe_long_audio(
                str(full_path),
                processor,
                model,
                device
            )
            data.append({
                "index": idx,
                "filename": file,
                "transcription": transcription,
                "text_length": len(transcription)
            })
        except Exception as e:
            print(f"❌ 오류 발생 - {file}: {e}")
            data.append({
                "index": idx,
                "filename": file,
                "transcription": "",
                "text_length": 0
            })
    
    # DataFrame 생성
    df = pd.DataFrame(data)
    
    # CSV 저장
    output_full_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_full_path, index=False, encoding='utf-8-sig')
    print(f"💾 결과 저장: {output_full_path} ({len(df)}개 파일)")
    
    return df


def transcribe_single_file(
    file_path: str,
    model_name: str = "openai/whisper-large-v3",
    device: str = None
) -> str:
    """
    단일 음성 파일을 STT 처리
    
    Args:
        file_path: 오디오 파일 경로 (상대경로)
        model_name: Whisper 모델명 (기본값: "openai/whisper-large-v3")
        device: 디바이스 (None이면 자동 선택)
        
    Returns:
        전사 텍스트
    """
    base_path = Path(__file__).parent.parent
    file_full_path = base_path / file_path
    
    if not file_full_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_full_path}")
    
    # 디바이스 설정
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 모델 로드
    print(f"📥 Whisper 모델 로딩 중... ({model_name})")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    print("✅ 모델 로드 완료")
    
    # STT 처리
    print(f"🎤 전사 중: {file_full_path}")
    transcription = transcribe_long_audio(
        str(file_full_path),
        processor,
        model,
        device
    )
    
    print(f"✅ 전사 완료: {len(transcription)}자")
    return transcription


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="음성 파일을 텍스트로 변환 (STT)")
    parser.add_argument("--folder", type=str, default=None, help="처리할 폴더 경로 (상대경로)")
    parser.add_argument("--file", type=str, default=None, help="처리할 단일 파일 경로 (상대경로)")
    parser.add_argument("--output", type=str, default="fss_dataset.csv", help="출력 CSV 파일 경로 (상대경로, 폴더 처리 시만 사용)")
    parser.add_argument("--model", type=str, default="openai/whisper-large-v3", help="Whisper 모델명")
    parser.add_argument("--device", type=str, default=None, help="디바이스 (cuda/cpu, None이면 자동 선택)")
    parser.add_argument("--extensions", type=str, nargs="+", default=[".mp4", ".wav", ".mp3", ".m4a"], help="처리할 파일 확장자")
    
    args = parser.parse_args()
    
    # 단일 파일 처리
    if args.file:
        transcription = transcribe_single_file(
            file_path=args.file,
            model_name=args.model,
            device=args.device
        )
        print("\n" + "="*50)
        print("전사 결과:")
        print("="*50)
        print(transcription)
        return
    
    # 폴더 처리
    if args.folder:
        df = transcribe_folder(
            folder_path=args.folder,
            output_path=args.output,
            model_name=args.model,
            device=args.device,
            file_extensions=args.extensions
        )
        print(f"\n✅ 처리 완료: {len(df)}개 파일")
        print(f"📊 전사 결과 미리보기:")
        print(df.head())
        return
    
    # 둘 다 없으면 에러
    parser.error("--folder 또는 --file 중 하나를 지정해야 합니다.")


if __name__ == "__main__":
    main()

