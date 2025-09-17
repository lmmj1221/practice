"""
제4장: BERT/KoBERT를 활용한 민원 분류기 구현
민원 텍스트를 카테고리별로 분류하는 모델 학습
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import json
from datetime import datetime
import os

# CPU 사용 설정 (GPU 없이도 실행 가능)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 재현성을 위한 시드 고정
np.random.seed(42)
torch.manual_seed(42)

class ComplaintDataset(Dataset):
    """민원 데이터셋 클래스"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 토큰화
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SimpleBERTClassifier(nn.Module):
    """간단한 BERT 기반 분류기 (transformers 없이 구현)"""
    
    def __init__(self, vocab_size, hidden_size=768, num_labels=10, max_length=128):
        super(SimpleBERTClassifier, self).__init__()
        
        # 간단한 임베딩 레이어 (실제 BERT 대신)
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_length, hidden_size)
        
        # Transformer 인코더 레이어 (간소화)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=3072,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        
        # 분류 헤드
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        # 임베딩
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        embeddings = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings
        
        # Transformer 인코딩
        # attention_mask를 transformer가 사용하는 형식으로 변환
        attention_mask = attention_mask.float()
        attention_mask = (1.0 - attention_mask) * -10000.0
        
        encoder_output = self.transformer_encoder(embeddings, src_key_padding_mask=attention_mask<0)
        
        # [CLS] 토큰의 표현 사용 (첫 번째 토큰)
        pooled_output = encoder_output[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        # 분류
        logits = self.classifier(pooled_output)
        
        return logits

class SimpleTokenizer:
    """간단한 토크나이저 (실제 BERT 토크나이저 대체)"""
    
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word_to_idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[UNK]': 3}
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
        self.current_idx = 4
        
    def __call__(self, text, truncation=True, padding='max_length', max_length=128, return_tensors='pt'):
        # 간단한 토큰화 (공백 기준)
        tokens = text.lower().split()[:max_length-2]  # [CLS], [SEP] 공간 확보
        
        # 토큰을 인덱스로 변환
        input_ids = [1]  # [CLS]
        for token in tokens:
            if token not in self.word_to_idx:
                if self.current_idx < self.vocab_size:
                    self.word_to_idx[token] = self.current_idx
                    self.idx_to_word[self.current_idx] = token
                    self.current_idx += 1
                    input_ids.append(self.word_to_idx[token])
                else:
                    input_ids.append(3)  # [UNK]
            else:
                input_ids.append(self.word_to_idx[token])
        
        input_ids.append(2)  # [SEP]
        
        # 패딩
        attention_mask = [1] * len(input_ids)
        while len(input_ids) < max_length:
            input_ids.append(0)  # [PAD]
            attention_mask.append(0)
        
        # 자르기
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        
        if return_tensors == 'pt':
            return {
                'input_ids': torch.tensor([input_ids]),
                'attention_mask': torch.tensor([attention_mask])
            }
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

def train_model(model, train_loader, val_loader, epochs=3, learning_rate=2e-5):
    """모델 학습 함수"""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # 학습 모드
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 순전파
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 정확도 계산
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(avg_loss)
        
        # 검증
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
    return train_losses, val_accuracies

def evaluate_model(model, test_loader, label_encoder):
    """모델 평가 함수"""
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 성능 메트릭 계산
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # 분류 리포트
    target_names = label_encoder.classes_
    report = classification_report(all_labels, all_predictions, 
                                  target_names=target_names, 
                                  output_dict=True)
    
    # 혼동 행렬
    cm = confusion_matrix(all_labels, all_predictions)
    
    return accuracy, report, cm

def main():
    """메인 실행 함수"""
    
    print("=" * 50)
    print("BERT 기반 민원 분류기 학습")
    print("=" * 50)
    
    # 데이터 로드
    data_path = 'C:/Dev/book-analysis/practice/chapter04/data/preprocessed_data.csv'
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    
    # 데이터 샘플링 (빠른 실행을 위해)
    sample_size = min(500, len(df))  # 최대 500개 샘플
    df_sample = df.sample(n=sample_size, random_state=42)
    
    print(f"\n데이터 로드 완료: {sample_size}개 샘플")
    print(f"카테고리 분포:\n{df_sample['category'].value_counts()}")
    
    # 레이블 인코딩
    label_encoder = LabelEncoder()
    df_sample['label'] = label_encoder.fit_transform(df_sample['category'])
    num_labels = len(label_encoder.classes_)
    
    # 데이터 분할
    X_train, X_temp, y_train, y_temp = train_test_split(
        df_sample['processed_text'], 
        df_sample['label'],
        test_size=0.3,
        random_state=42,
        stratify=df_sample['label']
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )
    
    print(f"\n데이터 분할:")
    print(f"  학습: {len(X_train)}, 검증: {len(X_val)}, 테스트: {len(X_test)}")
    
    # 토크나이저 초기화
    tokenizer = SimpleTokenizer(vocab_size=10000)
    
    # 데이터셋 생성
    train_dataset = ComplaintDataset(X_train.values, y_train.values, tokenizer)
    val_dataset = ComplaintDataset(X_val.values, y_val.values, tokenizer)
    test_dataset = ComplaintDataset(X_test.values, y_test.values, tokenizer)
    
    # 데이터 로더
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 모델 초기화
    model = SimpleBERTClassifier(
        vocab_size=10000,
        hidden_size=256,  # 작은 크기로 설정 (빠른 학습)
        num_labels=num_labels,
        max_length=128
    ).to(device)
    
    print(f"\n모델 초기화 완료")
    print(f"  카테고리 수: {num_labels}")
    print(f"  모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")
    
    # 모델 학습
    print("\n학습 시작...")
    start_time = datetime.now()
    
    train_losses, val_accuracies = train_model(
        model, train_loader, val_loader, 
        epochs=2,  # 빠른 실행을 위해 2 에폭만
        learning_rate=2e-5
    )
    
    training_time = datetime.now() - start_time
    print(f"\n학습 완료! 소요 시간: {training_time}")
    
    # 모델 평가
    print("\n모델 평가...")
    accuracy, report, cm = evaluate_model(model, test_loader, label_encoder)
    
    print(f"\n테스트 정확도: {accuracy:.4f}")
    print("\n카테고리별 성능:")
    for category in label_encoder.classes_[:5]:  # 상위 5개 카테고리만 출력
        if category in report:
            metrics = report[category]
            print(f"  {category}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    # 모델 저장
    output_dir = 'C:/Dev/book-analysis/practice/chapter04/output'
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, 'bert_classifier.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder_classes': label_encoder.classes_.tolist(),
        'num_labels': num_labels,
        'accuracy': accuracy,
        'training_time': str(training_time)
    }, model_path)
    
    print(f"\n모델 저장: {model_path}")
    
    # 결과 저장
    results = {
        'training_summary': {
            'total_samples': sample_size,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'num_categories': num_labels,
            'epochs': 2,
            'batch_size': batch_size,
            'training_time': str(training_time)
        },
        'performance': {
            'test_accuracy': float(accuracy),
            'train_losses': train_losses,
            'val_accuracies': val_accuracies
        },
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    results_path = os.path.join(output_dir, 'bert_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"결과 저장: {results_path}")
    
    # 예측 샘플
    print("\n" + "=" * 50)
    print("예측 샘플")
    print("=" * 50)
    
    sample_texts = [
        "버스 배차 간격이 너무 깁니다 개선이 필요합니다",
        "복지 지원금 신청 절차가 복잡합니다",
        "공원 관리가 제대로 되지 않고 있습니다"
    ]
    
    model.eval()
    for text in sample_texts:
        # 토큰화
        encoding = tokenizer(text, truncation=True, padding='max_length', 
                           max_length=128, return_tensors='pt')
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # 예측
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs.data, 1)
            predicted_label = label_encoder.inverse_transform([predicted.cpu().numpy()[0]])[0]
        
        print(f"텍스트: {text}")
        print(f"예측 카테고리: {predicted_label}\n")
    
    return model, results

if __name__ == "__main__":
    model, results = main()