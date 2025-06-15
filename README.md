# Flask ML Server

크로스핏 자세 분석을 위한 Flask 기반 ML 서버입니다.

## 프로젝트 구조

```
├── app.py                          # Flask 메인 애플리케이션
├── requirements.txt                # Python 의존성 패키지
├── models/
│   ├── clean_model.pth             # Clean 운동 모델 
│   ├── deadlift_model.pth          # Deadlift 운동 모델 
│   ├── press_model.pth             # Press 운동 모델 
│   ├── squat_model.pth             # Squat 운동 모델 
│   └── utils/
│       ├── model_loader.py         # 모델 로딩 유틸리티
│       └── preprocess.py           # 데이터 전처리
└── README.md
```

## 주요 기능

- **REST API**: `/api/predict` 엔드포인트로 자세 분석 제공
- **실시간 처리**: MediaPipe 랜드마크 데이터를 받아 즉시 분석

## 실행 방법

```bash
# 의존성 설치
pip install -r requirements.txt

# 서버 실행
python app.py
```

서버는 `http://localhost:8000`에서 실행됩니다.

## API 사용법

### POST /api/predict
```json
{
  "landmarks": [x1, y1, x2, y2, ...] // 17개 관절의 x,y 좌표 (총 34개 값)
}
```

### 응답
```json
{
  "best_posture": {
    "type": "deadlift/squat/press/clean",
    "normal_probability": 0.95,
    "is_normal": true
  },
  "all_results": {
    // 모든 운동별 분석 결과
  }
}
```

## 참고사항

- 현재 미완성 상태입니다
- 초기 inital_approach에만 맞는 형태여서 다른 approach 저장소들의 모델을 사용하기 위해서는 수정이 필요합니다. 
