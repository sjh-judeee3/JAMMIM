# MediaPipe Hands를 초기화
# 저장된 영상 파일을 읽어와 프레임별로 Keypoint를 추출
# 추출된 Keypoint를 data/keypoints/에 저장

import cv2
import mediapipe as mp
import numpy as np
import os

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
# 'Hands' 클래스는 Keypoint를 추출하는 모델입니다.
# max_num_hands=1: 한 손만 인식하도록 설정
# min_detection_confidence: 손을 인식했다고 판단하는 최소 확신도
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# --- 설정값 ---
# Keypoint 데이터를 저장할 경로 (data/keypoints/ 폴더에 저장됨)
OUTPUT_DIR = 'data/keypoints'
# 특징 추출 대상 영상 경로 (AI Hub 영상이나 직접 촬영 영상)
# 예: 'data/raw/sample_gesture_01.mp4'
INPUT_VIDEO_PATH = 'data/raw/sample_video_path.mp4' 

def extract_keypoints(video_path, output_dir):
    """
    영상 파일에서 Keypoint를 추출하고 NumPy 파일로 저장하는 함수.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 영상 파일 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    keypoint_sequence = []
    
    # 2. 영상 프레임별 처리
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 성능을 위해 BGR 이미지를 RGB로 변환 (MediaPipe는 RGB를 선호)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 3. MediaPipe Keypoint 추출
        results = hands.process(image)
        
        # 4. Keypoint 데이터 추출 및 저장
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 손의 21개 랜드마크 (x, y, z) 좌표를 추출
                frame_keypoints = []
                for landmark in hand_landmarks.landmark:
                    # x, y, z 좌표를 리스트에 추가
                    frame_keypoints.extend([landmark.x, landmark.y, landmark.z])
                
                # 추출된 Keypoint를 시퀀스 리스트에 추가
                keypoint_sequence.append(frame_keypoints)
        
        # 예시 실행을 위한 간단한 종료 조건 (옵션)
        # if cv2.waitKey(5) & 0xFF == 27:
        #     break

    cap.release()
    cv2.destroyAllWindows()
    
    # 5. 추출된 Keypoint 시퀀스를 NumPy 파일로 저장
    if keypoint_sequence:
        sequence_array = np.array(keypoint_sequence)
        
        # 저장 파일명 설정 (예: sample_video_01_keypoints.npy)
        file_name = os.path.basename(video_path).replace('.mp4', '_keypoints.npy')
        output_path = os.path.join(output_dir, file_name)
        
        np.save(output_path, sequence_array)
        print(f"Successfully extracted and saved keypoints to: {output_path}")
    else:
        print("Warning: No hand landmarks were detected in the video.")


if __name__ == '__main__':
    # AI Hub 영상 파일 경로를 실제 파일로 변경하여 실행합니다.
    # 예시:
    # extract_keypoints('data/raw/gesture_class_05_user_10_attempt_03.mp4', OUTPUT_DIR)
    
    # 테스트를 위한 기본 경로 사용
    # 주의: 이대로 실행하면 'data/raw/sample_video_path.mp4' 파일이 없어서 오류가 납니다.
    print("Please replace INPUT_VIDEO_PATH with a real video file path before running.")
    # extract_keypoints(INPUT_VIDEO_PATH, OUTPUT_DIR)