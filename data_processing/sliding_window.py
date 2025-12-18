"""
Sliding Window (슬라이딩 윈도우) 모듈

긴 텍스트를 지정된 윈도우 크기와 오버랩 비율로 분할하여 데이터셋을 생성합니다.
Ablation Study를 위해 다양한 윈도우 크기와 오버랩 비율 조합을 지원합니다.

사용 예시:
    from data_processing.sliding_window import apply_sliding_window
    
    result = apply_sliding_window(
        input_csv_path="dataset/train.csv",
        output_csv_path="dataset/train_512_25.csv",
        window_size=512,
        overlap_ratio=0.25
    )
"""

import pandas as pd
import os
from pathlib import Path
from typing import List, Dict, Optional


class SlidingWindowProcessor:
    """
    슬라이딩 윈도우 텍스트 분할 클래스
    
    긴 텍스트를 지정된 크기의 윈도우로 분할하며, 오버랩을 통해 정보 손실을 최소화합니다.
    """
    
    def __init__(self, window_size: int, overlap_ratio: float):
        """
        Args:
            window_size: 윈도우 크기 (문자 수)
            overlap_ratio: 오버랩 비율 (0.0 ~ 1.0)
        """
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.overlap = int(window_size * overlap_ratio)
        self.stride = window_size - self.overlap
    
    def split_text(self, text: str, label: int) -> List[Dict[str, any]]:
        """
        텍스트를 슬라이딩 윈도우로 분할
        
        Args:
            text: 분할할 텍스트
            label: 라벨 값
            
        Returns:
            분할된 청크 리스트 [{'text': str, 'Label': int}, ...]
        """
        text = str(text)
        text_len = len(text)
        chunks = []
        
        # 텍스트가 윈도우 크기 이하면 원본 반환
        if text_len <= self.window_size:
            return [{'text': text, 'Label': label}]
        
        # 슬라이딩 윈도우로 분할
        start = 0
        while True:
            end = start + self.window_size
            chunk = text[start:end]
            
            # 마지막 청크 처리 (정보 손실 방지)
            if len(chunk) < self.window_size:
                last_start = max(0, text_len - self.window_size)
                chunks.append({'text': text[last_start:], 'Label': label})
                break
            
            chunks.append({'text': chunk, 'Label': label})
            
            # 종료 조건 (stride=0인 경우 무한 루프 방지)
            if self.stride == 0 or start + self.stride >= text_len:
                break
            
            start += self.stride
        
        return chunks
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'text', label_column: str = 'label') -> pd.DataFrame:
        """
        데이터프레임의 모든 행에 슬라이딩 윈도우 적용
        
        Args:
            df: 입력 데이터프레임
            text_column: 텍스트 컬럼명 (기본값: 'text')
            label_column: 라벨 컬럼명 (기본값: 'label')
            
        Returns:
            분할된 데이터프레임
        """
        if text_column not in df.columns:
            raise ValueError(f"'{text_column}' 컬럼을 찾을 수 없습니다.")
        if label_column not in df.columns:
            raise ValueError(f"'{label_column}' 컬럼을 찾을 수 없습니다.")
        
        chunks = []
        for _, row in df.iterrows():
            text = row[text_column]
            label = row[label_column]
            
            if pd.isna(text):
                continue
            
            chunks.extend(self.split_text(text, label))
        
        return pd.DataFrame(chunks)


def apply_sliding_window(
    input_csv_path: str,
    output_csv_path: str,
    window_size: int,
    overlap_ratio: float,
    text_column: str = 'text',
    label_column: str = 'label'
) -> pd.DataFrame:
    """
    슬라이딩 윈도우를 적용하여 데이터셋 생성
    
    Args:
        input_csv_path: 입력 CSV 파일 경로 (상대경로)
        output_csv_path: 출력 CSV 파일 경로 (상대경로)
        window_size: 윈도우 크기 (문자 수)
        overlap_ratio: 오버랩 비율 (0.0 ~ 1.0)
        text_column: 텍스트 컬럼명 (기본값: 'text')
        label_column: 라벨 컬럼명 (기본값: 'label')
        
    Returns:
        분할된 데이터프레임
    """
    base_path = Path(__file__).parent.parent
    input_path = base_path / input_csv_path
    output_path = base_path / output_csv_path
    
    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")
    
    # 데이터 로드
    df = pd.read_csv(input_path)
    print(f"📂 데이터 로드: {input_path} ({len(df)}개 행)")
    
    # 출력 디렉토리 생성
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 슬라이딩 윈도우 처리
    processor = SlidingWindowProcessor(window_size, overlap_ratio)
    overlap = int(window_size * overlap_ratio)
    stride = window_size - overlap
    overlap_percent = int(overlap_ratio * 100)
    
    print(f"⚙️  처리 중: window_size={window_size}, overlap={overlap_percent}% (stride={stride})")
    
    result_df = processor.process_dataframe(df, text_column, label_column)
    
    # 결과 저장
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 완료: {output_path} ({len(result_df)}개 청크)")
    
    return result_df


def generate_sliding_window_datasets(
    input_csv_path: str,
    output_dir: str,
    window_sizes: List[int],
    overlap_ratios: List[float],
    text_column: str = 'text',
    label_column: str = 'label',
    output_filename_prefix: str = 'spam_bt_all'
) -> Dict[str, pd.DataFrame]:
    """
    여러 윈도우 크기와 오버랩 비율 조합으로 데이터셋 생성 (Ablation Study용)
    
    Args:
        input_csv_path: 입력 CSV 파일 경로 (상대경로)
        output_dir: 출력 디렉토리 경로 (상대경로)
        window_sizes: 윈도우 크기 리스트
        overlap_ratios: 오버랩 비율 리스트
        text_column: 텍스트 컬럼명 (기본값: 'text')
        label_column: 라벨 컬럼명 (기본값: 'label')
        output_filename_prefix: 출력 파일명 접두사 (기본값: 'spam_bt_all')
        
    Returns:
        생성된 데이터프레임 딕셔너리 {파일명: 데이터프레임}
    """
    base_path = Path(__file__).parent.parent
    input_path = base_path / input_csv_path
    output_dir_path = base_path / output_dir
    
    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")
    
    # 데이터 로드
    df = pd.read_csv(input_path)
    print(f"📂 데이터 로드: {input_path} ({len(df)}개 행)")
    
    # 출력 디렉토리 생성
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    total_combinations = len(window_sizes) * len(overlap_ratios)
    current = 0
    
    # 모든 조합에 대해 데이터셋 생성
    for window_size in window_sizes:
        for overlap_ratio in overlap_ratios:
            current += 1
            processor = SlidingWindowProcessor(window_size, overlap_ratio)
            overlap_percent = int(overlap_ratio * 100)
            
            print(f"[{current}/{total_combinations}] window_size={window_size}, overlap={overlap_percent}%")
            
            result_df = processor.process_dataframe(df, text_column, label_column)
            
            # 파일명 생성 및 저장
            filename = f'{output_filename_prefix}_{window_size}_{overlap_percent}.csv'
            output_path = output_dir_path / filename
            result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            print(f"  ✅ 저장: {filename} ({len(result_df)}개 청크)")
            results[filename] = result_df
    
    print(f"\n🎉 모든 데이터셋 생성 완료 ({total_combinations}개)")
    return results


if __name__ == "__main__":
    # 예시 1: 단일 데이터셋 생성
    result = apply_sliding_window(
        input_csv_path="dataset/total_dataset.csv",
        output_csv_path="dataset/sliding_window_512_25.csv",
        window_size=512,
        overlap_ratio=0.25
    )
    
    print(f"\n✅ 생성 완료: {result.shape}")
    
    # 예시 2: Ablation Study용 여러 데이터셋 생성
    # window_sizes = [256, 512, 768, 1024]
    # overlap_ratios = [0.0, 0.25, 0.5]
    # 
    # results = generate_sliding_window_datasets(
    #     input_csv_path="dataset/spam_bt_all.csv",
    #     output_dir="dataset",
    #     window_sizes=window_sizes,
    #     overlap_ratios=overlap_ratios,
    #     output_filename_prefix="spam_bt_all"
    # )

