"""
Head & Tail (앞뒤 자르기) 모듈

긴 텍스트의 앞부분(head)과 뒷부분(tail)을 잘라서 재구성하여 데이터셋을 생성합니다.
중간 부분을 제거하여 중요한 시작과 끝 부분만 유지합니다.

사용 예시:
    from data_processing.head_tail import apply_head_tail
    
    result = apply_head_tail(
        input_csv_path="dataset/train.csv",
        output_csv_path="dataset/train_head_tail.csv",
        head_size=512,
        tail_size=512
    )
"""

import pandas as pd
from pathlib import Path
from typing import Optional


class HeadTailProcessor:
    """
    Head & Tail 텍스트 처리 클래스
    
    텍스트의 앞부분과 뒷부분만 추출하여 재구성합니다.
    """
    
    def __init__(self, head_size: int, tail_size: int):
        """
        Args:
            head_size: 앞부분 크기 (문자 수)
            tail_size: 뒷부분 크기 (문자 수)
        """
        self.head_size = head_size
        self.tail_size = tail_size
        self.total_size = head_size + tail_size
    
    def create_head_tail_text(self, text: str) -> str:
        """
        텍스트의 앞/뒤를 잘라 재구성
        
        Args:
            text: 처리할 텍스트
            
        Returns:
            앞부분 + 뒷부분으로 구성된 텍스트
        """
        text = str(text)
        
        # 텍스트가 전체 크기 이하면 원본 반환
        if len(text) <= self.total_size:
            return text
        
        # 앞부분 + 뒷부분
        return text[:self.head_size] + text[-self.tail_size:]
    
    def process_dataframe(
        self, 
        df: pd.DataFrame, 
        text_column: str = 'text',
        label_column: str = 'label',
        output_text_column: str = 'text'
    ) -> pd.DataFrame:
        """
        데이터프레임의 모든 행에 Head & Tail 처리 적용
        
        Args:
            df: 입력 데이터프레임
            text_column: 입력 텍스트 컬럼명 (기본값: 'text')
            label_column: 라벨 컬럼명 (기본값: 'label')
            output_text_column: 출력 텍스트 컬럼명 (기본값: 'text')
            
        Returns:
            처리된 데이터프레임
        """
        if text_column not in df.columns:
            raise ValueError(f"'{text_column}' 컬럼을 찾을 수 없습니다.")
        if label_column not in df.columns:
            raise ValueError(f"'{label_column}' 컬럼을 찾을 수 없습니다.")
        
        result_df = df.copy()
        result_df[output_text_column] = result_df[text_column].apply(self.create_head_tail_text)
        
        return result_df[[output_text_column, label_column]].copy()


def apply_head_tail(
    input_csv_path: str,
    output_csv_path: str,
    head_size: int = 512,
    tail_size: int = 512,
    text_column: str = 'text',
    label_column: str = 'label',
    output_text_column: str = 'text'
) -> pd.DataFrame:
    """
    Head & Tail을 적용하여 데이터셋 생성
    
    Args:
        input_csv_path: 입력 CSV 파일 경로 (상대경로)
        output_csv_path: 출력 CSV 파일 경로 (상대경로)
        head_size: 앞부분 크기 (기본값: 512)
        tail_size: 뒷부분 크기 (기본값: 512)
        text_column: 입력 텍스트 컬럼명 (기본값: 'text')
        label_column: 라벨 컬럼명 (기본값: 'label')
        output_text_column: 출력 텍스트 컬럼명 (기본값: 'text')
        
    Returns:
        처리된 데이터프레임
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
    
    # Head & Tail 처리
    processor = HeadTailProcessor(head_size, tail_size)
    print(f"⚙️  처리 중: head_size={head_size}, tail_size={tail_size} (total={head_size + tail_size})")
    
    result_df = processor.process_dataframe(df, text_column, label_column, output_text_column)
    
    # 결과 저장
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 완료: {output_path} ({len(result_df)}개 행)")
    
    return result_df


if __name__ == "__main__":
    # 사용 예시
    result = apply_head_tail(
        input_csv_path="dataset/total_dataset_train.csv",
        output_csv_path="dataset/head_tail_train.csv",
        head_size=512,
        tail_size=512,
        text_column="text",
        label_column="label"
    )
    
    print(f"\n✅ 생성 완료: {result.shape}")

