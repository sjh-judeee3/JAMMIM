import numpy as np
import os
import random
from rules_transformer import transform_keypoints_to_rules # 규칙 변환 함수 import 가정

# --- 설정값 ---
KEYPOINTS_DIR = 'data/keypoints'
PROCESSED_DIR = 'data/processed'

# Few-Shot 학습을 위한 클래스 분리 설정 (AI Hub 데이터셋 클래스 수에 따라 조정)
NUM_TOTAL_CLASSES = 100  # 예시: 총 100개 제스처 클래스가 있다고 가정
NUM_SEEN_CLASSES = 70    # 학습에 사용할 클래스 수 (Seen Classes)
NUM_UNSEEN_CLASSES = 30  # 테스트에 사용할 클래스 수 (Unseen Classes, Few-Shot 대상)


def load_and_transform_data():
    """
    Keypoint 데이터를 로드하고 규칙 기반 특징으로 변환합니다.
    """
    all_data = []
    
    # keypoints 폴더의 모든 파일을 순회
    for filename in os.listdir(KEYPOINTS_DIR):
        if filename.endswith('_keypoints.npy'):
            path = os.path.join(KEYPOINTS_DIR, filename)
            
            # 1. Keypoint 로드
            keypoint_sequence = np.load(path)
            
            # 2. 규칙 기반 특징 변환 적용
            # transform_keypoints_to_rules 함수가 rules_transformer.py에 정의되어 있어야 함
            processed_features = transform_keypoints_to_rules(keypoint_sequence)
            
            # 파일명에서 클래스 레이블 추출 (AI Hub 파일명 규칙에 따라 수정 필요)
            # 예: filename = 'gesture_class_05_user_10_attempt_03_keypoints.npy'
            class_label = filename.split('_')[2] 
            
            all_data.append((processed_features, class_label, filename))
            
    return all_data

def split_and_save_data(all_data):
    """
    Few-Shot 학습을 위해 클래스를 Seen/Unseen으로 분리하고 저장합니다.
    """
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    # 모든 고유 클래스 레이블 추출
    unique_classes = list(set([item[1] for item in all_data]))
    
    # 클래스 개수 확인 및 분리 (랜덤 분리)
    random.shuffle(unique_classes)
    seen_classes = unique_classes[:NUM_SEEN_CLASSES]
    unseen_classes = unique_classes[NUM_SEEN_CLASSES:]

    print(f"Total unique classes found: {len(unique_classes)}")
    print(f"Seen classes for training: {len(seen_classes)}")
    print(f"Unseen classes for few-shot testing: {len(unseen_classes)}")

    # 데이터 저장 구조 초기화
    seen_train_data = []
    unseen_test_data = []
    
    # 데이터 분류 및 저장
    for features, label, filename in all_data:
        # NumPy 파일로 저장 (각 제스처 파일별로 저장)
        output_path = os.path.join(PROCESSED_DIR, filename.replace('_keypoints.npy', '_features.npy'))
        np.save(output_path, features)
        
        # 클래스 목록에 따라 메타 트레이닝 셋 또는 테스트 셋에 추가
        if label in seen_classes:
            seen_train_data.append({'path': output_path, 'label': label})
        elif label in unseen_classes:
            unseen_test_data.append({'path': output_path, 'label': label})

    # 최종적으로 Seen/Unseen 클래스 목록 및 데이터 경로를 JSON 파일로 저장 (LSTM 학습 스크립트가 사용)
    import json
    with open(os.path.join(PROCESSED_DIR, 'seen_train_meta.json'), 'w') as f:
        json.dump(seen_train_data, f)
    with open(os.path.join(PROCESSED_DIR, 'unseen_test_meta.json'), 'w') as f:
        json.dump(unseen_test_data, f)

    print("Data processed and metadata saved for Few-Shot training.")


if __name__ == '__main__':
    # 이 스크립트를 실행하기 전에 data/keypoints/ 폴더에 .npy 파일이 있어야 합니다.
    # 또한, 파일명에서 클래스 레이블을 추출하는 로직은 AI Hub 데이터셋의 파일명 규칙에 맞게 수정해야 합니다.
    data = load_and_transform_data()
    if data:
        split_and_save_data(data)