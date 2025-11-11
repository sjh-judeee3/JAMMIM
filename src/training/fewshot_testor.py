import torch
import torch.nn as nn
import json
import numpy as np
import os
import random
from model.backbone.fsl_model import FewShotClassifier, prototypical_loss # 모델과 로스 함수 import

# --- 설정 ---
PROCESSED_DIR = 'data/processed'
CHECKPOINT_DIR = 'model/checkpoints'

# --- 테스트 하이퍼파라미터 ---
N_WAY_TEST = 5          # 테스트 에피소드에 사용할 클래스 수 (N)
K_SHOT_TEST = 1         # 테스트 시 서포트 샘플 수 (K=1 One-Shot 테스트가 일반적)
N_QUERY_TEST = 10       # 테스트 시 쿼리 샘플 수
N_TEST_EPISODE = 600    # 총 테스트 에피소드 수 (충분한 통계 확보를 위해 높게 설정)

# --- 모델 하이퍼파라미터 (Trainer와 일치해야 함) ---
INPUT_DIM = 31 
Z_DIM = 64
HIDDEN_DIM = 128

# --- 1. FewShotDataLoader (테스트용) ---

# meta_trainer.py에 정의된 FewShotDataset과 FewShotDataLoader 클래스를 그대로 사용

# (주의: FewShotDataset 및 FewShotDataLoader 클래스는 fsl_model.py, meta_trainer.py와 동일한 정의를 사용한다고 가정)

# --------------------------------------------------------------------------------

def test_fsl_model(meta_test_path, checkpoint_file):
    """
    Unseen Classes를 대상으로 Few-Shot 테스트를 수행합니다.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 모델 로드
    model = FewShotClassifier(INPUT_DIM, HIDDEN_DIM, Z_DIM).to(device)
    
    # 학습된 가중치 파일 로드
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_file)
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval() # 모델을 평가 모드로 전환 (Dropout 등 비활성화)

    # 2. 데이터 로더 준비
    dataset = FewShotDataset(meta_test_path) # Unseen Classes 데이터셋 로드
    
    # FewShotDataLoader는 매번 다른 에피소드를 생성해야 하므로, 반복을 수동으로 제어
    dataloader = FewShotDataLoader(dataset, N_WAY_TEST, K_SHOT_TEST, N_QUERY_TEST)

    test_accuracies = []
    
    with torch.no_grad(): # 테스트 시에는 기울기 계산 비활성화
        
        for episode in range(N_TEST_EPISODE):
            # 3. 테스트 에피소드 생성 및 실행
            for batch_features, batch_labels in dataloader:
                
                features = batch_features.to(device)
                labels = batch_labels.to(device)

                # Loss 함수를 사용하여 정확도만 계산
                # (forward 패스 대신, 인코더 출력 후 loss 로직 호출)
                z = model.encoder(features)
                _, acc = prototypical_loss(z, labels, K_SHOT_TEST)
                
                test_accuracies.append(acc.item())
                break # FewShotDataLoader는 한 에피소드만 생성하므로 break

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{N_TEST_EPISODE} completed.")

    # 4. 최종 결과 출력
    final_accuracy = np.mean(test_accuracies)
    print("\n--- Few-Shot Test Results ---")
    print(f"Total Test Episodes: {N_TEST_EPISODE}")
    print(f"Test Configuration: {N_WAY_TEST}-Way {K_SHOT_TEST}-Shot")
    print(f"Average Top-1 Accuracy: {final_accuracy * 100:.2f}%")


if __name__ == '__main__':
    # 'unseen_test_meta.json' 파일과 학습된 체크포인트 파일 이름을 설정해야 합니다.
    meta_test_path = os.path.join(PROCESSED_DIR, 'unseen_test_meta.json')
    
    # TODO: 학습이 완료된 후, meta_trainer.py가 저장한 체크포인트 파일명을 여기에 넣으세요.
    LATEST_CHECKPOINT = 'fsl_gesture_checkpoint_1000.pth' 
    
    if os.path.exists(meta_test_path):
        test_fsl_model(meta_test_path, LATEST_CHECKPOINT)
    else:
        print(f"Error: Metadata file not found at {meta_test_path}. Please run rules_processor.py first.")