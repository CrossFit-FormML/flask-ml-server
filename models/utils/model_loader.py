import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TransformerPostureModel(nn.Module):
    """크로스핏 자세 분석을 위한 Vision Transformer 기반 모델"""
    def __init__(self, input_channels=68, embed_dim=128, kernel_size=3, num_classes=2):
        super(TransformerPostureModel, self).__init__()
        # Vision Transformer 구조
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, 91, embed_dim))
        
        # 입력 데이터를 임베딩으로 변환하는 컨볼루션 레이어
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=embed_dim, kernel_size=kernel_size, padding=1)
        
        # 트랜스포머 레이어
        transformer_layers = nn.ModuleList([])
        for _ in range(2):
            layer = nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim),
                'norm2': nn.LayerNorm(embed_dim),
                'self_attn': nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True),
                'linear1': nn.Linear(embed_dim, 2048),  # 2048 = embed_dim * 16
                'linear2': nn.Linear(2048, embed_dim)
            })
            transformer_layers.append(layer)
        
        self.transformer = nn.ModuleDict({
            'layers': transformer_layers
        })
        
        # 최종 정규화 레이어
        self.norm = nn.LayerNorm(embed_dim)
        
        # 분류 헤드
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        """
        입력 x의 형태: [batch_size, 17 관절 * 2 좌표 = 34]
        이것을 모델에 맞게 변환해야 합니다.
        """
        batch_size = x.size(0)
        
        # 17개 관절의 x, y 좌표를 68개 채널(x,y,z,v for 17 joints)로 확장
        # 실제 모델은 x,y 외에 추가 특성을 사용하는 것 같음
        expanded_x = torch.zeros(batch_size, 68, 3).to(x.device)
        
        # 기존 34개 좌표를 처음 34개 채널에 할당 (임시 솔루션)
        x_reshaped = x.view(batch_size, 17, 2)  # [batch, 17 joints, 2 coords]
        
        # x, y 좌표를 각각의 채널로 분리
        for i in range(17):
            expanded_x[:, i*4, 0] = x_reshaped[:, i, 0]  # x 좌표
            expanded_x[:, i*4+1, 0] = x_reshaped[:, i, 1]  # y 좌표
            # 나머지 채널은 0으로 둠
        
        x = expanded_x  # [batch_size, 68, 3]
        
        # 컨볼루션 레이어 통과
        x = self.conv1(x)  # [batch_size, 128, 3]
        x = x.transpose(1, 2)  # [batch_size, 3, 128]
        
        # cls 토큰 추가 (첫 번째 차원에)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, 4, 128]
        
        # 포지션 임베딩 추가 (일부만 사용)
        x = x + self.pos_embedding[:, :x.size(1)]
        
        # 트랜스포머 레이어 통과
        for layer_dict in self.transformer['layers']:
            # Self-Attention 블록
            residual = x
            x_ln = layer_dict['norm1'](x)
            attn_output, _ = layer_dict['self_attn'](x_ln, x_ln, x_ln)
            x = residual + attn_output
            
            # Feed-Forward 블록
            residual = x
            x_ln = layer_dict['norm2'](x)
            x_ff = layer_dict['linear1'](x_ln)
            x_ff = F.gelu(x_ff)
            x_ff = layer_dict['linear2'](x_ff)
            x = residual + x_ff
        
        # 최종 정규화
        x = self.norm(x)
        
        # 클래스 토큰을 사용한 분류
        x = x[:, 0]  # [batch_size, 128]
        x = self.classifier(x)  # [batch_size, num_classes]
        
        # 소프트맥스 적용
        x = F.softmax(x, dim=1)
        
        return x

class PostureModel(nn.Module):
    """간단한 다층 퍼셉트론 모델 - 더미 테스트용"""
    def __init__(self, input_size=34, hidden_size=64, dropout_rate=0.2):
        super(PostureModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_size // 2, 2)  # 정상(0)과 비정상(1) 자세 분류
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        
        return x

def create_dummy_model(input_size=34):
    """
    데모 및 테스트용 더미 모델을 생성
    """
    model = PostureModel(input_size)
    # 랜덤 가중치로 초기화
    model.eval()
    return model

def load_models():
    """
    학습된 모델 파일을 로드하는 함수
    
    Returns:
        dict: 각 자세 유형에 대한 모델 객체들을 담은 딕셔너리
    """
    models = {}
    use_real_models = True  # 실제 모델 사용
    
    # 모델 파일 경로 설정
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_paths = {
        'deadlift': os.path.join(base_dir, 'deadlift_model.pth'),
        'squat': os.path.join(base_dir, 'squat_model.pth'),
        'press': os.path.join(base_dir, 'press_model.pth'),
        'clean': os.path.join(base_dir, 'clean_model.pth')
    }
    
    # 모델별 클래스 수 정의
    num_classes = {
        'deadlift': 3,
        'squat': 4,
        'press': 4,
        'clean': 4
    }
    
    # 각 모델 로드 또는 더미 모델 생성
    for posture_type, model_path in model_paths.items():
        try:
            if use_real_models and os.path.exists(model_path):
                # 실제 모델 로드
                state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                
                # 모델 생성 (해당 자세 유형의 클래스 수에 맞게)
                model = TransformerPostureModel(
                    input_channels=68, 
                    embed_dim=128, 
                    kernel_size=3,
                    num_classes=num_classes[posture_type]
                )
                
                # 가중치 로드
                model.load_state_dict(state_dict)
                model.eval()  # 평가 모드로 설정
                models[posture_type] = model
                print(f"{posture_type} 모델 로드 성공")
            else:
                # 테스트용 더미 모델 생성
                dummy_model = create_dummy_model(input_size=34)
                models[posture_type] = dummy_model
                print(f"{posture_type}용 더미 모델 생성")
        except Exception as e:
            print(f"{posture_type} 모델 로드 실패: {e}")
            # 에러 발생 시 더미 모델로 대체
            dummy_model = create_dummy_model(input_size=34)
            models[posture_type] = dummy_model
            print(f"{posture_type}용 더미 모델 생성 (에러 대체)")
    
    return models