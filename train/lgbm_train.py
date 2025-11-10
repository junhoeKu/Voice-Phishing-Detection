"""
LightGBM 모델 학습 및 평가

TF-IDF 벡터화와 LightGBM을 사용하여 보이스피싱 탐지 모델을 학습합니다.
5회 실행의 평균 성능을 계산하고 평가합니다.

사용 예시:
    python train/lgbm_train.py --data dataset/total_dataset.csv
"""

import os
import random
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score
import lightgbm as lgb
import wandb


def set_seed(seed: int = 42):
    """난수 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)


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
    
    X = df[text_column].astype(str)
    y = df[label_column]
    
    print(f"📊 전체 데이터: {len(df)}개")
    return X, y


def train_and_evaluate(
    X_all,
    y_all,
    num_runs: int = 5,
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
    learning_rate: float = 0.05,
    num_boost_round: int = 200,
    max_depth: int = 6,
    early_stopping_rounds: int = 20,
    seed: int = 42
):
    """
    모델 학습 및 평가
    
    Args:
        X_all: 전체 텍스트 데이터
        y_all: 전체 라벨 데이터
        num_runs: 실행 횟수
        max_features: TF-IDF 최대 특성 수
        ngram_range: N-gram 범위
        learning_rate: 학습률
        num_boost_round: 부스팅 라운드 수
        max_depth: 최대 깊이
        early_stopping_rounds: 조기 종료 라운드
        seed: 시드 값
        
    Returns:
        (모델, 벡터라이저, 평가_결과) 튜플
    """
    val_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    test_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    
    last_model = None
    last_vectorizer = None
    
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
        
        # TF-IDF 벡터화
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.9
        )
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_val_tfidf = vectorizer.transform(X_val)
        X_test_tfidf = vectorizer.transform(X_test)
        
        print(f"[Run {run_idx+1}] TF-IDF Train shape: {X_train_tfidf.shape}")
        
        # LightGBM Dataset 생성
        lgb_train = lgb.Dataset(X_train_tfidf, label=y_train)
        lgb_val = lgb.Dataset(X_val_tfidf, label=y_val, reference=lgb_train)
        
        # 하이퍼파라미터
        pos_count = float(np.sum(y_train))
        neg_count = float(len(y_train) - pos_count)
        scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0
        
        params = {
            "objective": "binary",
            "metric": ["binary_logloss"],
            "learning_rate": learning_rate,
            "num_leaves": 31,
            "max_depth": max_depth if max_depth is not None else -1,
            "scale_pos_weight": scale_pos_weight,
            "seed": run_seed,
            "verbosity": -1
        }
        
        # 학습
        print(f"[Run {run_idx+1}] 학습 중...")
        bst = lgb.train(
            params,
            lgb_train,
            num_boost_round=num_boost_round,
            valid_sets=[lgb_train, lgb_val],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=50)
            ]
        )
        
        # 검증 (Validation)
        y_val_pred_prob = bst.predict(X_val_tfidf, num_iteration=bst.best_iteration)
        y_val_pred = (y_val_pred_prob > 0.5).astype(int)
        
        print(f"--- [Run {run_idx+1}] Validation Set Results ---")
        print(classification_report(y_val, y_val_pred))
        
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred, zero_division=0)
        val_recall = recall_score(y_val, y_val_pred, zero_division=0)
        val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
        
        print(f"[Run {run_idx+1}] Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
        
        wandb.log({
            "run_idx": run_idx + 1,
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1_score": val_f1
        })
        
        val_metrics["accuracy"].append(val_accuracy)
        val_metrics["precision"].append(val_precision)
        val_metrics["recall"].append(val_recall)
        val_metrics["f1"].append(val_f1)
        
        # 최종 평가 (Test)
        y_test_pred_prob = bst.predict(X_test_tfidf, num_iteration=bst.best_iteration)
        y_test_pred = (y_test_pred_prob > 0.5).astype(int)
        
        print(f"\n--- [Run {run_idx+1}] Test Set Results (Final) ---")
        print(classification_report(y_test, y_test_pred))
        
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        
        print(f"[Run {run_idx+1}] Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}")
        
        wandb.log({
            "run_idx": run_idx + 1,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1_score": test_f1
        })
        
        test_metrics["accuracy"].append(test_accuracy)
        test_metrics["precision"].append(test_precision)
        test_metrics["recall"].append(test_recall)
        test_metrics["f1"].append(test_f1)
        
        # 마지막 모델과 벡터라이저 저장
        last_model = bst
        last_vectorizer = vectorizer
    
    # 평균 결과 계산
    def _mean(v):
        return float(np.mean(v)) if len(v) > 0 else float("nan")
    
    val_avg = {k: _mean(v) for k, v in val_metrics.items()}
    test_avg = {k: _mean(v) for k, v in test_metrics.items()}
    
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
    
    return last_model, last_vectorizer, {"val": val_avg, "test": test_avg}


def save_model(model, vectorizer, model_path: str, vectorizer_path: str = None):
    """
    모델 및 벡터라이저 저장
    
    Args:
        model: 학습된 모델
        vectorizer: 학습된 벡터라이저
        model_path: 모델 저장 경로 (상대경로)
        vectorizer_path: 벡터라이저 저장 경로 (상대경로, None이면 자동 생성)
    """
    base_path = Path(__file__).parent.parent
    model_full_path = base_path / model_path
    model_full_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.save_model(str(model_full_path))
    print(f"💾 모델 저장: {model_full_path}")
    
    if vectorizer_path:
        vectorizer_full_path = base_path / vectorizer_path
        vectorizer_full_path.parent.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(vectorizer, vectorizer_full_path)
        print(f"💾 벡터라이저 저장: {vectorizer_full_path}")
    else:
        # 벡터라이저도 같은 경로에 저장 (확장자만 다름)
        import joblib
        vectorizer_path = str(model_path).replace(".txt", "_vectorizer.pkl")
        vectorizer_full_path = base_path / vectorizer_path
        joblib.dump(vectorizer, vectorizer_full_path)
        print(f"💾 벡터라이저 저장: {vectorizer_full_path}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="LightGBM 모델 학습 및 평가")
    parser.add_argument("--data", type=str, default="dataset/total_dataset.csv", help="데이터 파일 경로 (상대경로)")
    parser.add_argument("--model_path", type=str, default="model/lgbm_voicephishing.txt", help="모델 저장 경로 (상대경로)")
    parser.add_argument("--vectorizer_path", type=str, default=None, help="벡터라이저 저장 경로 (상대경로, None이면 자동 생성)")
    parser.add_argument("--num_runs", type=int, default=5, help="실행 횟수")
    parser.add_argument("--max_features", type=int, default=5000, help="TF-IDF 최대 특성 수")
    parser.add_argument("--learning_rate", type=float, default=0.05, help="학습률")
    parser.add_argument("--num_boost_round", type=int, default=200, help="부스팅 라운드 수")
    parser.add_argument("--max_depth", type=int, default=6, help="최대 깊이")
    parser.add_argument("--early_stopping_rounds", type=int, default=20, help="조기 종료 라운드")
    parser.add_argument("--seed", type=int, default=42, help="시드 값")
    parser.add_argument("--wandb_project", type=str, default="Voicephishing", help="WandB 프로젝트명")
    parser.add_argument("--wandb_name", type=str, default=None, help="WandB 실행명 (None이면 자동 생성)")
    args = parser.parse_args()
    
    wandb.init(project=args.wandb_project, name=args.wandb_name or f"tfidf_lgbm_{args.num_runs}runs", config={
        "model": "LightGBM", "max_features": args.max_features, "ngram_range": (1, 2), "random_state": args.seed,
        "learning_rate": args.learning_rate, "num_boost_round": args.num_boost_round, "max_depth": args.max_depth,
        "early_stopping_rounds": args.early_stopping_rounds, "num_runs": args.num_runs,
        "split_ratio": "train 0.70, val 0.15, test 0.15"
    })
    
    try:
        X_all, y_all = load_data(args.data)
        model, vectorizer, metrics = train_and_evaluate(
            X_all=X_all, y_all=y_all, num_runs=args.num_runs, max_features=args.max_features, ngram_range=(1, 2),
            learning_rate=args.learning_rate, num_boost_round=args.num_boost_round, max_depth=args.max_depth,
            early_stopping_rounds=args.early_stopping_rounds, seed=args.seed
        )
        if model is not None:
            save_model(model, vectorizer, args.model_path, args.vectorizer_path)
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

