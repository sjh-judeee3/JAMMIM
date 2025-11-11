# GestureGPT - 규칙 기반 특징 변환기
# keypoints_extractor.py에서 추출된 Keypoint 데이터를 규칙 기반 특징 시퀀스로 변환
# MediaPipe Keypoint의 x, y, z 좌표 -> gesture의 의미적 상태를 나타내는 벡터로 변환

"""
Input
: NumPy 파일로 저장된 Keypoint 시퀀스 (data/keypoints/ 폴더)

Output
: 변환된 시퀀스를 data/processed/ 폴더에 NumPy 파일로 저장, LSTM에 input으로 사용 가능
"""


import numpy as np
import os

# --- 설정값 ---
INPUT_DIR = 'data/keypoints'
OUTPUT_DIR = 'data/processed'

# MediaPipe Keypoint Index (21개 랜드마크 중 주요 관절 인덱스)
# 각 손가락의 MCP(손바닥 연결), PIP(첫 번째 마디), DIP(두 번째 마디), TIP(끝)
WRIST = 0
THUMB_TIP = 4
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_PIP = 14
RING_TIP = 16
PINKY_PIP = 18
PINKY_TIP = 20

# -----------------------------------------------------------
# 유틸리티 함수: 세 점을 이용해 각도를 계산 (손가락 굽힘 측정에 사용)
def calculate_angle(a, b, c):
    """3D 좌표 a, b, c를 이용해 b를 중심으로 하는 각도를 계산합니다."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # 벡터 ba와 bc를 구하고, 두 벡터 간의 각도를 계산
    radians = np.arccos(np.dot(b - a, b - c) / (np.linalg.norm(b - a) * np.linalg.norm(b - c)))
    return np.degrees(radians)

# -----------------------------------------------------------
# 1. 규칙 함수 정의

def get_finger_flexion_state(keypoints, finger_tip_idx, finger_pip_idx):
    """
    손가락의 굽힘 상태를 판단합니다. (간단화된 로직)
    팁(Tip)과 중간 마디(PIP)의 거리를 이용해 굽힘 정도를 추정합니다.
    """
    tip_to_pip_dist = np.linalg.norm(keypoints[finger_tip_idx] - keypoints[finger_pip_idx])
    
    # 임계값 (Threshold) 설정: 거리가 작으면 '굽힘'으로 판단 (이 값은 학습으로 최적화 필요)
    FLEXION_THRESHOLD = 0.05 
    
    if tip_to_pip_dist < FLEXION_THRESHOLD:
        return 1  # 굽힘 (Bent)
    else:
        return 0  # 펴짐/중간 (Straight/Other)

def get_finger_proximity_state(keypoints, tip_idx_a, tip_idx_b):
    """
    두 손가락 끝의 근접성을 판단합니다.
    """
    tip_a_coords = keypoints[tip_idx_a]
    tip_b_coords = keypoints[tip_idx_b]
    
    distance = np.linalg.norm(tip_a_coords - tip_b_coords)
    
    # 임계값 설정: 거리가 매우 가까우면 '접촉/근접'으로 판단
    PROXIMITY_THRESHOLD = 0.03
    
    if distance < PROXIMITY_THRESHOLD:
        return 1  # 근접/접촉 (Close)
    else:
        return 0  # 떨어짐 (Apart)

# -----------------------------------------------------------
# 2. 메인 변환 함수

def transform_keypoints_to_rules(keypoint_sequence):
    """
    Keypoint 시퀀스를 규칙 기반 특징 시퀀스로 변환합니다.
    """
    transformed_sequence = []
    
    for frame_keypoints_flat in keypoint_sequence:
        # Keypoint 배열을 21x3 형태로 재구성
        keypoints = frame_keypoints_flat.reshape((-1, 3))
        
        # 각 프레임의 특징 벡터 (1차원)
        frame_features = []
        
        # 1. 손가락 굽힘 (Flexion) 특징 (4개 손가락: 검지, 중지, 약지, 소지)
        frame_features.append(get_finger_flexion_state(keypoints, INDEX_TIP, INDEX_PIP))
        frame_features.append(get_finger_flexion_state(keypoints, MIDDLE_TIP, MIDDLE_PIP))
        frame_features.append(get_finger_flexion_state(keypoints, RING_TIP, RING_PIP))
        frame_features.append(get_finger_flexion_state(keypoints, PINKY_TIP, PINKY_PIP))

        # 2. 손가락 간 근접성 (Proximity) 특징 (3쌍: 검중, 중약, 약소)
        frame_features.append(get_finger_proximity_state(keypoints, INDEX_TIP, MIDDLE_TIP))
        frame_features.append(get_finger_proximity_state(keypoints, MIDDLE_TIP, RING_TIP))
        frame_features.append(get_finger_proximity_state(keypoints, RING_TIP, PINKY_TIP))
        
        # 최종 특징 벡터를 시퀀스에 추가
        transformed_sequence.append(frame_features)

    return np.array(transformed_sequence)

# -----------------------------------------------------------
# 3. 데이터 로드 및 저장 실행

def process_all_keypoint_files():
    """
    INPUT_DIR 내의 모든 .npy Keypoint 파일을 처리하고 변환된 특징을 저장합니다.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith('_keypoints.npy'):
            input_path = os.path.join(INPUT_DIR, filename)
            
            try:
                # 1. Keypoint 데이터 로드
                keypoint_sequence = np.load(input_path)
                
                # 2. 규칙 기반 특징 변환 실행
                transformed_features = transform_keypoints_to_rules(keypoint_sequence)
                
                # 3. 변환된 특징 저장
                output_filename = filename.replace('_keypoints.npy', '_processed_features.npy')
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                
                np.save(output_path, transformed_features)