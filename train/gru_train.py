"""
GRU 모델 학습 및 평가

BERT 토크나이저와 GRU를 사용하여 보이스피싱 탐지 모델을 학습합니다.
5회 실행의 평균 성능을 계산하고 평가합니다.

사용 예시:
    python train/gru_train.py --data dataset/total_dataset.csv
"""

import os
import random
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm
from transformers import BertTokenizer
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
    
    X = df[text_column].astype(str)
    y = df[label_column].astype(int)
    
    print(f"📊 전체 데이터: {len(df)}개")
    return X, y


class GRUDataset(Dataset):
    """GRU용 데이터셋"""
    def __init__(self, texts, labels, tokenizer, max_length=1024):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }


class GRUBinaryClassifier(nn.Module):
    """GRU 분류기 모델"""
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers=1, bidirectional=True, dropout=0.3):
        super(GRUBinaryClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        gru_out, _ = self.gru(x)
        out = gru_out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        return out.squeeze()


def evaluate(model, dataloader, criterion, device):
    """모델 평가"""
    model.eval()
    preds = []
    targets = []
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].float().to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * input_ids.size(0)
            preds.extend((outputs > 0.5).long().cpu().numpy())
            targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, zero_division=0)
    precision = precision_score(targets, preds, zero_division=0)
    recall = recall_score(targets, preds, zero_division=0)
    return avg_loss, acc, f1, precision, recall, targets, preds


def train_and_evaluate(
    X_all,
    y_all,
    num_runs: int = 5,
    embedding_dim: int = 256,
    hidden_dim: int = 128,
    num_layers: int = 1,
    bidirectional: bool = True,
    dropout: float = 0.3,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    epochs: int = 10,
    max_length: int = 1024,
    seed: int = 42,
    device: str = None
):
    """
    모델 학습 및 평가
    
    Args:
        X_all: 전체 텍스트 데이터
        y_all: 전체 라벨 데이터
        num_runs: 실행 횟수
        embedding_dim: 임베딩 차원
        hidden_dim: 히든 차원
        num_layers: GRU 레이어 수
        bidirectional: 양방향 여부
        dropout: 드롭아웃 비율
        batch_size: 배치 크기
        learning_rate: 학습률
        epochs: 에포크 수
        max_length: 최대 길이
        seed: 시드 값
        device: 디바이스
        
    Returns:
        (모델, 토크나이저, 평가_결과) 튜플
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"📱 디바이스: {device}")
    
    # 토크나이저 로드
    tokenizer = BertTokenizer.from_pretrained("monologg/kobert", cache_dir=str(Path(__file__).parent.parent / "cache"))
    vocab_size = tokenizer.vocab_size
    
    val_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    test_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    
    last_model = None
    
    for run_idx in range(num_runs):
        run_seed = seed + run_idx
        set_seed(run_seed)
        
        # 데이터 분할: 70% train, 30% temp
        X_train_texts, X_temp_texts, y_train, y_temp = train_test_split(
            X_all,
            y_all,
            test_size=0.30,
            random_state=run_seed,
            stratify=y_all
        )
        
        # temp을 15% val, 15% test로 분할
        X_val_texts, X_test_texts, y_val, y_test = train_test_split(
            X_temp_texts,
            y_temp,
            test_size=0.50,
            random_state=run_seed,
            stratify=y_temp
        )
        
        print(f"[Run {run_idx+1}] Train: {len(X_train_texts)}, Val: {len(X_val_texts)}, Test: {len(X_test_texts)}")
        
        # Dataset / DataLoader
        train_dataset = GRUDataset(X_train_texts.tolist(), y_train.tolist(), tokenizer, max_length)
        val_dataset = GRUDataset(X_val_texts.tolist(), y_val.tolist(), tokenizer, max_length)
        test_dataset = GRUDataset(X_test_texts.tolist(), y_test.tolist(), tokenizer, max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size * 2)
        
        # 모델 생성
        model = GRUBinaryClassifier(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout
        ).to(device)
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # 학습 루프
        best_f1 = 0.0
        best_model_state = None
        
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
            loop = tqdm(train_loader, desc=f"Run {run_idx+1} Epoch {epoch+1}", leave=False)
            
            for batch in loop:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].float().to(device)
                optimizer.zero_grad()
                outputs = model(input_ids)
                loss = criterion(outputs, labels)
                
                if torch.isnan(loss):
                    print(f"⚠️ Warning: Run {run_idx+1} Epoch {epoch+1} NaN loss detected. Skipping batch.")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_train_loss += loss.item() * input_ids.size(0)
                loop.set_postfix(train_loss=f"{loss.item():.4f}")
            
            avg_train_loss = total_train_loss / len(train_loader.dataset)
            
            # Validation 평가
            valid_loss, acc, f1, precision, recall, _, _ = evaluate(model, val_loader, criterion, device)
            
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": valid_loss,
                "val_accuracy": acc,
                "val_f1": f1,
                "val_precision": precision,
                "val_recall": recall
            })
            
            print(f"Run {run_idx+1} Epoch {epoch+1} -> Train Loss: {avg_train_loss:.4f} | Valid Loss: {valid_loss:.4f} | Val F1: {f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_model_state = model.state_dict().copy()
                print(f"Run {run_idx+1} Best model saved! (Val F1: {best_f1:.4f})")
                wandb.save(str(Path(__file__).parent.parent / f"model/gru_run_{run_idx+1}.pt"))
        
        # Best 모델 로드 후 평가
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            last_model = model
        
        # Validation 최종 평가
        val_loss, val_acc, val_f1, val_precision, val_recall, val_targets, val_preds = evaluate(model, val_loader, criterion, device)
        print(f"\n--- [Run {run_idx+1}] Validation Set Results ---")
        print(classification_report(val_targets, val_preds, zero_division=0))
        print(f"[Run {run_idx+1}] Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f}")
        
        wandb.log({
            "run_idx": run_idx + 1,
            "final_val_loss": val_loss,
            "final_val_accuracy": val_acc,
            "final_val_f1": val_f1,
            "final_val_precision": val_precision,
            "final_val_recall": val_recall
        })
        
        val_metrics["accuracy"].append(val_acc)
        val_metrics["precision"].append(val_precision)
        val_metrics["recall"].append(val_recall)
        val_metrics["f1"].append(val_f1)
        
        # Test 평가
        test_loss, test_acc, test_f1, test_precision, test_recall, test_targets, test_preds = evaluate(model, test_loader, criterion, device)
        print(f"\n--- [Run {run_idx+1}] Test Set Results (Final) ---")
        print(classification_report(test_targets, test_preds, zero_division=0))
        print(f"[Run {run_idx+1}] Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f} | Test Precision: {test_precision:.4f} | Test Recall: {test_recall:.4f}")
        
        wandb.log({
            "run_idx": run_idx + 1,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "test_precision": test_precision,
            "test_recall": test_recall
        })
        
        test_metrics["accuracy"].append(test_acc)
        test_metrics["precision"].append(test_precision)
        test_metrics["recall"].append(test_recall)
        test_metrics["f1"].append(test_f1)
    
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
    
    return last_model, tokenizer, {"val": val_avg, "test": test_avg}


def save_model(model, tokenizer, model_path: str):
    """
    모델 및 토크나이저 저장
    
    Args:
        model: 학습된 모델
        tokenizer: 토크나이저
        model_path: 모델 저장 경로 (상대경로)
    """
    base_path = Path(__file__).parent.parent
    model_full_path = base_path / model_path
    model_full_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), model_full_path)
    print(f"💾 모델 저장: {model_full_path}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="GRU 모델 학습 및 평가")
    parser.add_argument("--data", type=str, default="dataset/total_dataset.csv", help="데이터 파일 경로 (상대경로)")
    parser.add_argument("--model_path", type=str, default="model/gru_voicephishing.pt", help="모델 저장 경로 (상대경로)")
    parser.add_argument("--num_runs", type=int, default=5, help="실행 횟수")
    parser.add_argument("--embedding_dim", type=int, default=256, help="임베딩 차원")
    parser.add_argument("--hidden_dim", type=int, default=128, help="히든 차원")
    parser.add_argument("--num_layers", type=int, default=1, help="GRU 레이어 수")
    parser.add_argument("--bidirectional", action="store_true", default=True, help="양방향 GRU 사용")
    parser.add_argument("--dropout", type=float, default=0.3, help="드롭아웃 비율")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="학습률")
    parser.add_argument("--epochs", type=int, default=10, help="에포크 수")
    parser.add_argument("--max_length", type=int, default=1024, help="최대 길이")
    parser.add_argument("--seed", type=int, default=42, help="시드 값")
    parser.add_argument("--device", type=str, default=None, help="디바이스 (cuda/cpu, None이면 자동 선택)")
    parser.add_argument("--wandb_project", type=str, default="Voicephishing", help="WandB 프로젝트명")
    parser.add_argument("--wandb_name", type=str, default=None, help="WandB 실행명 (None이면 자동 생성)")
    args = parser.parse_args()
    
    device = args.device if args.device else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    wandb.init(project=args.wandb_project, name=args.wandb_name or f"gru_{args.num_runs}runs", config={
        "model": "GRU", "embedding_dim": args.embedding_dim, "hidden_dim": args.hidden_dim, "num_layers": args.num_layers,
        "bidirectional": args.bidirectional, "dropout": args.dropout, "batch_size": args.batch_size,
        "max_length": args.max_length, "learning_rate": args.learning_rate, "num_epochs": args.epochs,
        "num_runs": args.num_runs, "split_ratio": "train 0.70, val 0.15, test 0.15"
    })
    
    try:
        X_all, y_all = load_data(args.data)
        model, tokenizer, metrics = train_and_evaluate(
            X_all=X_all, y_all=y_all, num_runs=args.num_runs, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim,
            num_layers=args.num_layers, bidirectional=args.bidirectional, dropout=args.dropout, batch_size=args.batch_size,
            learning_rate=args.learning_rate, epochs=args.epochs, max_length=args.max_length, seed=args.seed, device=device
        )
        if model is not None:
            save_model(model, tokenizer, args.model_path)
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

