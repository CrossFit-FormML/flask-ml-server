from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import os
from models.utils.model_loader import load_models
from models.utils.preprocess import preprocess_landmarks

app = Flask(__name__)
CORS(app)  # 모든 출처에서의 요청 허용

# 환경 변수 설정
DEBUG = os.environ.get('FLASK_DEBUG', True)
PORT = int(os.environ.get('PORT', 5000))

# 모델 로드
print("모델 로딩 중...")
models = load_models()
print("모델 로딩 완료!")

@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태 확인용 엔드포인트"""
    return jsonify({"status": "healthy", "models_loaded": bool(models)}), 200

@app.route('/api/predict', methods=['POST'])
def predict_posture():
    """
    랜드마크 좌표를 기반으로 자세를 예측하는 API 엔드포인트
    
    요청 형식:
    {
        "landmarks": [x1, y1, x2, y2, ...]  # 필요한 관절 좌표들의 1차원 배열
    }
    
    응답 형식:
    {
        "best_posture": {
            "type": "데드리프트/스쿼트/프레스/클린",
            "normal_probability": 0.95,  # 정상 자세일 확률
            "is_normal": true  # 정상 자세 여부
        },
        "all_results": {
            "deadlift": {
                "normal_probability": 0.95,
                "abnormal_probability": 0.05,
                "is_normal": true
            },
            ... 다른 자세들에 대한 정보 ...
        }
    }
    """
    if request.method == 'OPTIONS':
        # preflight 요청에 대한 응답
        return jsonify({}), 200
        
    if not models:
        return jsonify({"error": "모델이 로드되지 않았습니다."}), 500
    
    try:
        # 요청 데이터 파싱
        data = request.get_json()
        if not data or 'landmarks' not in data:
            return jsonify({"error": "요청 형식이 잘못되었습니다. 'landmarks' 필드가 필요합니다."}), 400
        
        landmarks = data.get('landmarks', [])
        if not landmarks or len(landmarks) != 34:  # 17개 관절 x 2(x,y) = 34
            return jsonify({
                "error": "잘못된 랜드마크 데이터입니다. 17개 관절의 x, y 좌표가 필요합니다.",
                "received_length": len(landmarks) if landmarks else 0
            }), 400
        
        # 데이터 전처리
        processed_landmarks = preprocess_landmarks(landmarks)
        input_tensor = torch.tensor(processed_landmarks, dtype=torch.float32).unsqueeze(0)
        
        # 각 모델로 예측
        results = {}
        for posture_type, model in models.items():
            with torch.no_grad():
                output = model(input_tensor)
                # 정상 자세 확률
                normal_prob = output[0][0].item()
                # 비정상 자세 확률
                abnormal_prob = output[0][1].item()
                
                results[posture_type] = {
                    "normal_probability": round(normal_prob, 4),
                    "abnormal_probability": round(abnormal_prob, 4),
                    "is_normal": normal_prob > abnormal_prob
                }
        
        # 가장 높은 정상 확률을 가진 자세 찾기
        best_posture = max(results.items(), key=lambda x: x[1]["normal_probability"])
        
        response = {
            "best_posture": {
                "type": best_posture[0],
                "normal_probability": best_posture[1]["normal_probability"],
                "is_normal": best_posture[1]["is_normal"]
            },
            "all_results": results
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"예측 중 오류 발생: {str(e)}")
        return jsonify({"error": f"예측 중 오류 발생: {str(e)}"}), 500

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """로드된 모델 정보를 반환하는 API 엔드포인트"""
    if not models:
        return jsonify({"error": "모델이 로드되지 않았습니다."}), 500
    
    model_info = {
        "loaded_models": list(models.keys()),
        "input_shape": "34 features (17 landmarks with x,y coordinates)",
        "output_shape": "2 classes (normal/abnormal posture)"
    }
    
    return jsonify(model_info)

if __name__ == '__main__':
    app.run(debug=DEBUG, host='0.0.0.0', port=8000)  