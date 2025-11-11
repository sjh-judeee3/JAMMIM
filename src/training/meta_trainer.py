import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import json
import numpy as np
import os
import random
from model.backbone.fsl_model import FewShotClassifier # 구현된 모델 import

# --- 설정 ---
PROCESSED_DIR = 'data/processed'
CHECKPOINT_DIR = 'model/checkpoints'
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# --- 학습 하이퍼파라미터 ---
# N-Way K-Shot 설정 (Few-Shot 에피소드 정의)
N_WAY = 5           # 한 에피소드에 사용될 클래스 수 (N)
K_SHOT = 5          # 각 클래스당 서포트 샘플 수 (K)
N_QUERY = 10        # 각 클래스당 쿼리 샘플 수
N_EPISODE = 1000    # 총 에피소드 수 (훈련 반복 횟수)
LEARNING_RATE = 1e-3
Z_DIM = 64          # FSL 모델의 특징 공간 차원 (fsl_model.py와 일치)

# --- 1. Few-Shot 데이터셋 로더 ---

class FewShotDataset(Dataset):
    """
    메타 트레이닝용 데이터셋. 파일 경로와 레이블 메타데이터를 로드.
    """
    def __init__(self, meta_path):
        with open(meta_path, 'r') as f:
            self.data_meta = json.load(f)
        
        # 클래스별 데이터 인덱스를 딕셔너리로 구성
        self.class_data = {}
        for item in self.data_meta:
            label = item['label']
            if label not in self.class_data:
                self.class_data[label] = []
            self.class_data[label].append(item['path'])
            
        self.unique_classes = list(self.class_data.keys())
        self.n_classes = len(self.unique_classes)
        print(f"Loaded {len(self.data_meta)} total samples from {self.n_classes} classes.")

    def __len__(self):
        # 학습의 용이성을 위해 총 에피소드 수에 맞춰 길이를 반환할 수도 있으나,
        # 여기서는 메타 러닝의 특성상 DataLoader에서 반복을 제어
        return self.n_classes

class FewShotDataLoader:
    """
    매 호출 시마다 N-Way K+Q-Shot 에피소드를 생성하는 데이터 로더.
    """
    def __init__(self, dataset, n_way, k_shot, n_query):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query

    def __iter__(self):
        # 1. N-Way 클래스 랜덤 선택
        selected_classes = random.sample(self.dataset.unique_classes, self.n_way)
        
        episode_features = []
        episode_labels = []

        for i, class_label in enumerate(selected_classes):
            # 2. 각 클래스에서 K+Q개 샘플 선택
            class_samples = self.dataset.class_data[class_label]
            # 샘플 수가 부족하면 건너뛰거나 패딩 필요 (여기서는 간단히 처리)
            if len(class_samples) < (self.k_shot + self.n_query):
                continue

            selected_samples_paths = random.sample(class_samples, self.k_shot + self.n_query)
            
            # 3. 데이터 로드 및 시퀀스 구성 (Padding 또는 Resizing 필요)
            class_features = []
            for path in selected_samples_paths:
                # Keypoint 시퀀스 로드
                seq = np.load(path)
                # 시퀀스 길이를 맞춰주는 전처리 (실제 구현 시 필요)
                # 여기서는 간단히 패딩/자르기 없이 사용한다고 가정
                class_features.append(seq) 

            # 임시 처리: 시퀀스 길이가 다를 경우 가장 짧은 길이에 맞춤 (실제 구현 시 통일 필요)
            min_len = min([f.shape[0] for f in class_features])
            
            # 4. Support Set (K-Shot)과 Query Set (N_QUERY) 구성
            
            # 특징 벡터를 Tensor로 변환
            features_tensor = torch.stack([torch.tensor(f[:min_len], dtype=torch.float32) for f in class_features])
            
            # 특징 시퀀스 리스트에 추가
            episode_features.append(features_tensor)
            
            # 레이블 생성: Support + Query 
            num_samples = features_tensor.shape[0]
            labels = torch.full((num_samples,), i) # 0부터 N-1까지의 클래스 레이블
            episode_labels.append(labels)
        
        if episode_features:
            # 배치 구성: (Batch_Size, Sequence_Length, Feature_Dim)
            batch_features = torch.cat(episode_features, dim=0)
            batch_labels = torch.cat(episode_labels, dim=0)
            
            # FewShotClassifier에 맞게 반환
            yield batch_features, batch_labels
    
    # 이 클래스는 에피소드 반복을 위한 함수를 제공하며, len()이 없으므로 DataLoader로 직접 래핑하지 않음

# --- 3. 트레이너 실행 ---

def train_fsl_model(meta_train_path, n_episode):
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 모델 및 최적화 도구 초기화
    INPUT_DIM = 31 # (GestureGPT 특징 차원 가정)
    model = FewShotClassifier(INPUT_DIM, 128, Z_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 데이터 로더 준비
    dataset = FewShotDataset(meta_train_path)
    dataloader = FewShotDataLoader(dataset, N_WAY, K_SHOT, N_QUERY)

    model.train()
    
    total_loss = 0
    total_acc = 0

    for episode in range(1, n_episode + 1):
        # 에피소드 로드
        for batch_features, batch_labels in dataloader:
            
            # 데이터 디바이스로 전송
            features = batch_features.to(device)
            labels = batch_labels.to(device)

            optimizer.zero_grad()
            
            # 모델 포워드 패스 (FewShotClassifier 내에서 Prototypical Loss 계산)
            loss, acc, _ = model(features, labels, K_SHOT)
            
            # 백워드 및 최적화
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc.item()

        if episode % 50 == 0:
            avg_loss = total_loss / 50
            avg_acc = total_acc / 50
            print(f"Episode {episode}/{n_episode} | Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.4f}")
            total_loss = 0
            total_acc = 0

        # 모델 저장 (1000 에피소드마다)
        if episode % 1000 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'fsl_gesture_checkpoint_{episode}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

    print("Meta-training complete.")

if __name__ == '__main__':
    # 이 경로에 'seen_train_meta.json' 파일이 존재해야 학습이 시작됩니다.
    meta_train_path = os.path.join(PROCESSED_DIR, 'seen_train_meta.json')
    if os.path.exists(meta_train_path):
        train_fsl_model(meta_train_path, N_EPISODE)
    else:
        print(f"Error: Metadata file not found at {meta_train_path}. Please run rules_processor.py first.")