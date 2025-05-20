import torch
import os

def analyze_model_file(model_path):
    """
    모델 파일(.pth)을 분석하여 내부 구조를 출력하는 함수
    """
    try:
        # 모델 로드
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        print(f"\n분석: {os.path.basename(model_path)}")
        print("=" * 50)
        
        # 레이어 수와 종류 확인
        num_layers = len(state_dict.keys())
        print(f"총 레이어 수: {num_layers}")
        
        # 레이어 타입 분류
        layer_types = {}
        for key in state_dict.keys():
            prefix = key.split('.')[0] if '.' in key else key
            if prefix not in layer_types:
                layer_types[prefix] = 0
            layer_types[prefix] += 1
        
        print("\n레이어 타입 분포:")
        for layer_type, count in layer_types.items():
            print(f"  - {layer_type}: {count}개 파라미터")
        
        # 주요 레이어 상세 정보
        print("\n주요 레이어 상세 정보:")
        for key, value in state_dict.items():
            print(f"  - {key}: {value.shape}")
        
        # 분류 레이어 정보 (출력 클래스 수 확인)
        if 'classifier.weight' in state_dict:
            output_classes = state_dict['classifier.weight'].shape[0]
            input_features = state_dict['classifier.weight'].shape[1]
            print(f"\n분류기 정보:")
            print(f"  - 출력 클래스 수: {output_classes}")
            print(f"  - 입력 특성 수: {input_features}")
        
        # 입력 특성 크기 추정
        if 'conv1.weight' in state_dict:
            conv_shape = state_dict['conv1.weight'].shape
            print(f"\n첫 번째 컨볼루션 레이어:")
            print(f"  - 형태: {conv_shape}")
            print(f"  - 이는 입력 채널이 {conv_shape[1]}개임을 의미할 수 있습니다")
        
        return state_dict
    
    except Exception as e:
        print(f"모델 분석 중 오류 발생: {e}")
        return None

if __name__ == "__main__":
    base_dir = 'models'
    model_files = [
        os.path.join(base_dir, 'deadlift_model.pth'),
        os.path.join(base_dir, 'squat_model.pth'),
        os.path.join(base_dir, 'press_model.pth'),
        os.path.join(base_dir, 'clean_model.pth')
    ]
    
    for model_file in model_files:
        analyze_model_file(model_file)