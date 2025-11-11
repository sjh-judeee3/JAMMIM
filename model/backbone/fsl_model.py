import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# --- 1. LSTM Encoder 정의: 제스처 시퀀스를 단일 특징 벡터로 압축 ---

class LSTMEncoder(nn.Module):
    """
    제스처의 시퀀스 특징을 입력받아 단일 특징 벡터(z)를 출력하는 LSTM 인코더.
    """
    def __init__(self, input_dim, hidden_dim, z_dim, num_layers=1):
        super(LSTMEncoder, self).__init__()
        
        # input_dim: 1단계에서 추출된 특징 벡터의 차원 (예: 31차원)
        # hidden_dim: LSTM 내부 은닉 상태의 차원
        # z_dim: 최종 출력 특징 벡터 z의 차원 (Prototypical Network의 특징 공간 차원)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Batch_first=True: 입력 텐서 순서가 (Batch_Size, Sequence_Length, Feature_Dim)이 됨
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # LSTM 출력(hidden_dim)을 최종 특징 공간(z_dim)으로 변환하는 FC 레이어
        self.fc = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        """
        x: (Batch_Size, Sequence_Length, Feature_Dim) 형태의 입력 특징 시퀀스
        """
        # LSTM 초기 상태 설정 (h0, c0) - 보통 0으로 초기화
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM 포워드 패스: output, (hn, cn)
        # hn: 최종 시점의 hidden state (여기서는 이것만 사용)
        _, (hn, _) = self.lstm(x, (h0, c0))
        
        # 최종 시점의 hidden state를 사용하여 특징 벡터 z 생성
        # hn[-1, :, :]는 마지막 레이어의 hidden state를 가져옴
        z = self.fc(hn[-1, :, :]) 
        
        return z

# --- 2. Prototypical Loss 함수 정의 (Few-Shot 학습 로직) ---

def prototypical_loss(input_features, target_labels, n_support):
    """
    Prototypical Network의 핵심 손실 함수.
    
    Args:
        input_features (Tensor): LSTM 인코더가 출력한 특징 벡터 z.
        target_labels (Tensor): 특징 벡터에 해당하는 클래스 레이블.
        n_support (int): 각 클래스당 사용된 서포트(Support) 샘플의 개수 (K-Shot의 K).
    """
    # 유클리디안 거리를 제곱한 값 (거리의 제곱)
    def euclidean_dist(x, y):
        x_sq = (x ** 2).sum(dim=1).unsqueeze(1)
        y_sq = (y ** 2).sum(dim=1).unsqueeze(0)
        # (x - y)^2 = x^2 - 2xy + y^2
        distances = x_sq + y_sq - 2 * torch.matmul(x, y.transpose(0, 1))
        # distances.clamp_(min=0)  # 음수 방지 (수치적 안정성)
        return distances

    # 클래스별 프로토타입 (p) 계산
    classes = torch.unique(target_labels)
    n_classes = len(classes)
    
    # 텐서를 분할하여 프로토타입 계산
    # features_per_class[i]에는 해당 클래스의 모든 서포트 및 쿼리 특징 벡터가 포함됨
    features_per_class = [input_features[torch.where(target_labels == c)[0]] for c in classes]
    
    # 프로토타입은 각 클래스의 서포트 샘플 특징 벡터들의 평균 (K-Shot의 K개)
    # n_support까지의 특징만 사용 (첫 K개가 서포트, 나머지는 쿼리라고 가정)
    prototypes = torch.stack([f[:n_support].mean(dim=0) for f in features_per_class])
    
    # 쿼리 샘플 (Query Samples) 추출: 각 클래스에서 서포트를 제외한 나머지
    query_features = torch.cat([f[n_support:] for f in features_per_class])
    
    # 쿼리 샘플에 대한 실제 레이블
    query_labels = torch.cat([torch.full((f.shape[0] - n_support,), i) for i, f in enumerate(features_per_class)])
    
    # 쿼리와 모든 프로토타입 간의 거리 계산
    # distance_matrix의 크기는 (N_query, N_classes)
    distance_matrix = euclidean_dist(query_features, prototypes)
    
    # 손실 계산: 거리가 가까울수록 확률이 높아지도록 음의 거리 사용 (-distance)
    # LogSoftmax를 사용하면 확률이 계산되고, NLLLoss로 손실을 구함 (Log-Probabilities)
    log_p_y = F.log_softmax(-distance_matrix, dim=1)
    
    # NLLLoss (Negative Log Likelihood Loss)를 사용하여 손실 계산
    loss = F.nll_loss(log_p_y, query_labels.long())
    
    # 정확도 계산 (추가)
    _, y_hat = log_p_y.max(1)
    accuracy = torch.eq(y_hat, query_labels.long()).float().mean()
    
    return loss, accuracy

# --- 3. FewShotClassifier 정의 (학습용 메타 모델) ---

class FewShotClassifier(nn.Module):
    """
    LSTM 인코더와 Prototypical Loss를 결합한 Few-Shot 학습 모델.
    """
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(FewShotClassifier, self).__init__()
        
        # LSTM 인코더를 모델의 특징 추출기로 사용
        self.encoder = LSTMEncoder(input_dim, hidden_dim, z_dim)

    def forward(self, input_sequence, labels, n_support):
        # 1. LSTM 인코더를 통해 입력 시퀀스를 특징 벡터로 변환
        z = self.encoder(input_sequence)
        
        # 2. Prototypical Loss 로직을 사용하여 손실 및 정확도 계산
        loss, accuracy = prototypical_loss(z, labels, n_support)
        
        return loss, accuracy, z

# --- 초기화 예시 (실제 학습 스크립트에서 사용) ---

if __name__ == '__main__':
    # 예시 설정
    INPUT_DIM = 31      # GestureGPT 규칙 기반 특징 차원 (rules_processor에서 사용)
    HIDDEN_DIM = 128    # LSTM hidden 차원
    Z_DIM = 64          # 최종 특징 벡터 z의 차원
    N_WAY = 5           # 한 에피소드에 사용될 클래스 수 (N)
    K_SHOT = 1          # 각 클래스당 서포트 샘플 수 (K)
    N_QUERY = 5         # 각 클래스당 쿼리 샘플 수
    SEQ_LEN = 30        # 제스처 시퀀스 길이 (프레임 수)
    BATCH_SIZE = N_WAY * (K_SHOT + N_QUERY)

    # 모델 인스턴스 생성
    model = FewShotClassifier(INPUT_DIM, HIDDEN_DIM, Z_DIM)

    # 가상 에피소드 데이터 생성: (5 클래스 * 6 샘플) = 30개 샘플
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)
    dummy_labels = torch.arange(N_WAY).repeat_interleave(K_SHOT + N_QUERY)

    # 모델 포워드 패스 (손실 계산)
    loss, acc, features = model(dummy_input, dummy_labels, K_SHOT)

    print(f"Few-Shot Classifier initialized.")
    print(f"Features shape (z): {features.shape}")
    print(f"Prototypical Loss: {loss.item():.4f}")
    print(f"Accuracy: {acc.item():.4f}")