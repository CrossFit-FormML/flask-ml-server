import numpy as np
from typing import List

def normalize_landmarks(landmarks: List[float]) -> List[float]:
    """
    관절 좌표를 정규화하는 함수
    
    Args:
        landmarks: 원본 랜드마크 좌표 리스트 [x1, y1, x2, y2, ...]
        
    Returns:
        List[float]: 정규화된 랜드마크 좌표
    """
    # 2차원 배열로 변환 (x, y 쌍)
    landmarks_array = np.array(landmarks).reshape(-1, 2)
    
    # 중심점 계산 (모든 점의 평균)
    center_x = np.mean(landmarks_array[:, 0])
    center_y = np.mean(landmarks_array[:, 1])
    
    # 중심점 기준으로 이동
    centered_landmarks = landmarks_array - np.array([center_x, center_y])
    
    # 스케일 정규화 (최대 절대값으로 나누어 [-1, 1] 범위로)
    max_abs_val = np.max(np.abs(centered_landmarks))
    if max_abs_val > 0:  # 0으로 나누기 방지
        normalized_landmarks = centered_landmarks / max_abs_val
    else:
        normalized_landmarks = centered_landmarks
    
    # 다시 1차원 배열로 변환
    return normalized_landmarks.flatten().tolist()

def remove_outliers(landmarks: List[float], threshold: float = 3.0) -> List[float]:
    """
    이상치를 제거하거나 수정하는 함수
    
    Args:
        landmarks: 랜드마크 좌표 리스트
        threshold: Z-score 임계값
        
    Returns:
        List[float]: 이상치가 처리된 랜드마크 좌표
    """
    # 2차원 배열로 변환
    landmarks_array = np.array(landmarks).reshape(-1, 2)
    
    # x, y 좌표에 대한 평균과 표준편차 계산
    mean_x = np.mean(landmarks_array[:, 0])
    mean_y = np.mean(landmarks_array[:, 1])
    std_x = np.std(landmarks_array[:, 0]) + 1e-6  # 0으로 나누기 방지
    std_y = np.std(landmarks_array[:, 1]) + 1e-6
    
    # Z-score 계산
    z_scores_x = np.abs((landmarks_array[:, 0] - mean_x) / std_x)
    z_scores_y = np.abs((landmarks_array[:, 1] - mean_y) / std_y)
    
    # 이상치 처리 (평균값으로 대체)
    for i in range(len(landmarks_array)):
        if z_scores_x[i] > threshold:
            landmarks_array[i, 0] = mean_x
        if z_scores_y[i] > threshold:
            landmarks_array[i, 1] = mean_y
    
    return landmarks_array.flatten().tolist()

def preprocess_landmarks(landmarks: List[float]) -> List[float]:
    """
    랜드마크 데이터를 전처리하는 메인 함수
    
    Args:
        landmarks: 원본 랜드마크 좌표 리스트 [x1, y1, x2, y2, ...]
        
    Returns:
        List[float]: 전처리된 랜드마크 좌표
    """
    # 이상치 제거
    cleaned_landmarks = remove_outliers(landmarks)
    
    # 정규화
    normalized_landmarks = normalize_landmarks(cleaned_landmarks)
    
    return normalized_landmarks