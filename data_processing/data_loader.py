"""
데이터 로더 및 분할 모듈

CSV 파일을 로드하고 train, validation, test 데이터셋으로 분할합니다.
재현성을 위해 시드를 고정하고 분할된 데이터셋을 CSV 파일로 저장합니다.

사용 예시:
    from data_processing.data_loader import split_dataset
    
    train, val, test = split_dataset(
        input_csv_path="dataset/total_dataset.csv",
        output_dir="dataset"
    )
"""

import os
import pandas as pd
from datasets import Dataset
from pathlib import Path
from typing import Tuple, Optional


def load_voicephishing_data(
    file_path: str,
    text_column: str = "text",
    label_column: str = "label",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    output_dir: Optional[str] = None,
    train_filename: str = "total_dataset_train.csv",
    val_filename: str = "total_dataset_val.csv",
    test_filename: str = "total_dataset_test.csv"
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    보이스피싱 데이터를 로드하고 train, validation, test로 분할
    
    Args:
        file_path: 입력 CSV 파일 경로 (상대경로)
        text_column: 텍스트 컬럼명 (기본값: 'text')
        label_column: 라벨 컬럼명 (기본값: 'label')
        train_ratio: 학습 데이터 비율 (기본값: 0.7)
        val_ratio: 검증 데이터 비율 (기본값: 0.15)
        test_ratio: 테스트 데이터 비율 (기본값: 0.15)
        seed: 난수 시드 (기본값: 42)
        output_dir: 출력 디렉토리 경로 (상대경로, None이면 입력 파일과 같은 디렉토리)
        train_filename: 학습 데이터 파일명 (기본값: 'total_dataset_train.csv')
        val_filename: 검증 데이터 파일명 (기본값: 'total_dataset_val.csv')
        test_filename: 테스트 데이터 파일명 (기본값: 'total_dataset_test.csv')
        
    Returns:
        (train_dataset, val_dataset, test_dataset) 튜플
        
    Raises:
        ValueError: CSV 파일이 아닌 경우
        FileNotFoundError: 입력 파일을 찾을 수 없는 경우
    """
    # 경로를 상대경로로 처리
    base_path = Path(__file__).parent.parent
    input_path = base_path / file_path
    
    # 입력 파일 확인
    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")
    
    # CSV 파일 확인
    if not file_path.endswith(".csv"):
        raise ValueError("CSV 파일만 지원됩니다.")
    
    # 데이터 로드
    df = pd.read_csv(input_path)
    print(f"📂 데이터 로드: {input_path} ({len(df)}개 행)")
    
    # 컬럼 확인
    if text_column not in df.columns:
        raise ValueError(f"'{text_column}' 컬럼을 찾을 수 없습니다.")
    if label_column not in df.columns:
        raise ValueError(f"'{label_column}' 컬럼을 찾을 수 없습니다.")
    
    # 데이터 정리
    df[text_column] = df[text_column].astype(str)
    df = df[[text_column, label_column]].copy()
    
    # Dataset으로 변환
    dataset = Dataset.from_pandas(df)
    print(f"📊 전체 데이터: {len(dataset)}개")
    
    # 1단계: train vs temp 분할
    test_size_temp = 1.0 - train_ratio
    train_val_split = dataset.train_test_split(test_size=test_size_temp, shuffle=True, seed=seed)
    
    # 2단계: temp를 val과 test로 분할
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    val_test_split = train_val_split["test"].train_test_split(
        test_size=(1.0 - val_test_ratio), shuffle=True, seed=seed
    )
    
    train_dataset = train_val_split['train']
    val_dataset = val_test_split['train']
    test_dataset = val_test_split['test']
    
    print(f"📊 학습: {len(train_dataset)}개 ({len(train_dataset)/len(dataset)*100:.1f}%)")
    print(f"📊 검증: {len(val_dataset)}개 ({len(val_dataset)/len(dataset)*100:.1f}%)")
    print(f"📊 테스트: {len(test_dataset)}개 ({len(test_dataset)/len(dataset)*100:.1f}%)")
    
    # 출력 디렉토리 설정
    if output_dir is None:
        save_dir = input_path.parent
    else:
        save_dir = base_path / output_dir
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터셋 저장
    train_path = save_dir / train_filename
    val_path = save_dir / val_filename
    test_path = save_dir / test_filename
    
    print(f"💾 저장 중...")
    train_dataset.to_csv(str(train_path), index=False)
    val_dataset.to_csv(str(val_path), index=False)
    test_dataset.to_csv(str(test_path), index=False)
    
    print(f"✅ 학습: {train_path}")
    print(f"✅ 검증: {val_path}")
    print(f"✅ 테스트: {test_path}")
    
    return train_dataset, val_dataset, test_dataset


def split_dataset(
    input_csv_path: str,
    output_dir: str = "dataset",
    text_column: str = "text",
    label_column: str = "label",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    train_filename: str = "total_dataset_train.csv",
    val_filename: str = "total_dataset_val.csv",
    test_filename: str = "total_dataset_test.csv"
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    데이터셋을 train, validation, test로 분할하는 간편 함수
    
    Args:
        input_csv_path: 입력 CSV 파일 경로 (상대경로)
        output_dir: 출력 디렉토리 경로 (상대경로)
        text_column: 텍스트 컬럼명 (기본값: 'text')
        label_column: 라벨 컬럼명 (기본값: 'label')
        train_ratio: 학습 데이터 비율 (기본값: 0.7)
        val_ratio: 검증 데이터 비율 (기본값: 0.15)
        test_ratio: 테스트 데이터 비율 (기본값: 0.15)
        seed: 난수 시드 (기본값: 42)
        train_filename: 학습 데이터 파일명
        val_filename: 검증 데이터 파일명
        test_filename: 테스트 데이터 파일명
        
    Returns:
        (train_dataset, val_dataset, test_dataset) 튜플
    """
    return load_voicephishing_data(
        file_path=input_csv_path,
        text_column=text_column,
        label_column=label_column,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        output_dir=output_dir,
        train_filename=train_filename,
        val_filename=val_filename,
        test_filename=test_filename
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="데이터셋을 train, validation, test로 분할")
    parser.add_argument("--input", type=str, default="dataset/total_dataset.csv",
                        help="입력 CSV 파일 경로 (상대경로)")
    parser.add_argument("--output_dir", type=str, default="dataset",
                        help="출력 디렉토리 경로 (상대경로)")
    parser.add_argument("--text_column", type=str, default="text",
                        help="텍스트 컬럼명")
    parser.add_argument("--label_column", type=str, default="label",
                        help="라벨 컬럼명")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                        help="학습 데이터 비율")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="검증 데이터 비율")
    parser.add_argument("--test_ratio", type=float, default=0.15,
                        help="테스트 데이터 비율")
    parser.add_argument("--seed", type=int, default=42,
                        help="난수 시드")
    parser.add_argument("--train_filename", type=str, default="total_dataset_train.csv",
                        help="학습 데이터 파일명")
    parser.add_argument("--val_filename", type=str, default="total_dataset_val.csv",
                        help="검증 데이터 파일명")
    parser.add_argument("--test_filename", type=str, default="total_dataset_test.csv",
                        help="테스트 데이터 파일명")
    
    args = parser.parse_args()
    
    # 비율 합계 확인
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"⚠️ 경고: 비율의 합이 1.0이 아닙니다 ({total_ratio}). 정규화합니다.")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio
    
    try:
        train_dataset, val_dataset, test_dataset = split_dataset(
            input_csv_path=args.input,
            output_dir=args.output_dir,
            text_column=args.text_column,
            label_column=args.label_column,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            train_filename=args.train_filename,
            val_filename=args.val_filename,
            test_filename=args.test_filename
        )
        
        print("\n✅ 데이터셋 분할 완료! 이제 train.py를 실행할 수 있습니다.")
        
    except FileNotFoundError as e:
        print(f"❌ 오류: {e}")
    except ValueError as e:
        print(f"❌ 오류: {e}")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")

