import numpy as np
import os
import random
import json
# rules_transformer.py 파일에서 transform_keypoints_to_rules 함수를 import 한다고 가정
from rules_transformer import transform_keypoints_to_rules 

# --- 설정값 ---
KEYPOINTS_DIR = 'data/keypoints'
PROCESSED_DIR = 'data/processed'

# Few-Shot 학습을 위한 클래스 분리 설정 (사용자 정의 제스처 클래스 수에 따라 조정)
NUM_TOTAL_CLASSES = 30   # 예시: 총 30개 제스처 클래스가 있다고 가정 (AI Hub 외 제스처 데이터 사용 시)
NUM_SEEN_CLASSES = 20    # 학습에 사용할 클래스 수 (Seen: 모델에게 학습 방법을 가르침)
NUM_UNSEEN_CLASSES = 10  # 테스트에 사용할 클래스 수 (Unseen: Few-Shot 능력 검증)


def load_and_transform_data():
    """
    Keypoint 데이터를 로드하고 규칙 기반 특징으로 변환합니다.
    """
    all_data = []
    
    # keypoints 폴더의 모든 파일을 순회
    for filename in os.listdir(KEYPOINTS_DIR):
        if filename.endswith('_keypoints.npy'):
            path = os.path.join(KEYPOINTS_DIR, filename)
            
            try:
                # 1. Keypoint 로드
                keypoint_sequence = np.load(path)
                
                # 2. 규칙 기반 특징 변환 적용
                # 이 함수는 GestureGPT 규칙에 따라 의미적 특징 벡터를 생성한다고 가정합니다.
                processed_features = transform_keypoints_to_rules(keypoint_sequence)
                
                # 파일명에서 클래스 레이블 추출 (파일명을 'class_XX_...' 형식으로 가정)
                # 이 로직은 실제 데이터셋 파일명 규칙에 맞게 수정해야 합니다.
                parts = filename.split('_')
                if len(parts) > 1 and parts[1].isdigit():
                    class_label = int(parts[1])  # 예: 'class_05_...' -> 5
                else:
                    print(f"Skipping {filename}: Cannot parse class label.")
                    continue
                
                all_data.append((processed_features, class_label, filename))
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
    return all_data

def split_and_save_data(all_data):
    """
    Few-Shot 학습을 위해 클래스를 Seen/Unseen으로 분리하고 저장합니다.
    """
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    # 모든 고유 클래스 레이블 추출
    unique_classes = list(set([item[1] for item in all_data]))
    
    # 클래스 개수가 충분한지 확인
    if len(unique_classes) < NUM_TOTAL_CLASSES:
        print(f"Warning: Found only {len(unique_classes)} classes. Expected {NUM_TOTAL_CLASSES}. Adjusting split.")
        NUM_SEEN_CLASSES = len(unique_classes) - 10 if len(unique_classes) > 10 else 0
        NUM_UNSEEN_CLASSES = len(unique_classes) - NUM_SEEN_CLASSES
        
    # 클래스 분리 (재현성을 위해 시드 고정 권장)
    random.seed(42)
    random.shuffle(unique_classes)
    seen_classes = unique_classes[:NUM_SEEN_CLASSES]
    unseen_classes = unique_classes[NUM_SEEN_CLASSES:]

    print(f"Total unique classes found: {len(unique_classes)}")
    print(f"Seen classes for meta-training: {len(seen_classes)}")
    print(f"Unseen classes for few-shot testing: {len(unseen_classes)}")

    seen_train_data = []
    unseen_test_data = []
    
    # 데이터 분류 및 저장
    for features, label, filename in all_data:
        # 1. 변환된 특징을 .npy 파일로 저장
        output_filename = filename.replace('_keypoints.npy', '_features.npy')
        output_path = os.path.join(PROCESSED_DIR, output_filename)
        
        np.save(output_path, features)
        
        # 2. 메타데이터 분류
        meta_data = {'path': output_path, 'label': label, 'original_file': filename}
        if label in seen_classes:
            seen_train_data.append(meta_data)
        elif label in unseen_classes:
            unseen_test_data.append(meta_data)

    # 최종적으로 Seen/Unseen 클래스 목록 및 데이터 경로를 JSON 파일로 저장
    with open(os.path.join(PROCESSED_DIR, 'seen_train_meta.json'), 'w') as f:
        json.dump(seen_train_data, f, indent=4)
    with open(os.path.join(PROCESSED_DIR, 'unseen_test_meta.json'), 'w') as f:
        json.dump(unseen_test_data, f, indent=4)

    print(f"\nData saved in {PROCESSED_DIR}/. Meta files created.")


if __name__ == '__main__':
    # 주의: 이 스크립트를 실행하기 전에 data/keypoints/ 폴더에 .npy 파일이 존재해야 합니다.
    # 또한, rules_transformer.py 파일에 transform_keypoints_to_rules 함수가 정의되어 있어야 합니다.
    print("Starting rules-based feature processing and Few-Shot data splitting...")
    data = load_and_transform_data()
    if data:
        split_and_save_data(data)