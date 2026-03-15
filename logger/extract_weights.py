import torch
import json
import os

def extract_ai_recipe():
    checkpoint_path = 'saved/checkpoint_best.pth'
    
    if not os.path.exists(checkpoint_path):
        print("❌ 학습된 모델 파일(pth)을 찾을 수 없습니다. 먼저 train.py를 실행해주세요.")
        return

    # 모델 불러오기
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    state_dict = checkpoint['state_dict']
    
    # 1. 이평선 가중치 추출 (Softmax 적용 전의 raw 가중치)
    # 모델 내부의 'stock_weights' 파라미터를 가져옵니다.
    stock_w = state_dict['stock_weights']
    # 확률(비중)로 변환
    stock_probs = torch.softmax(stock_w, dim=0).numpy()
    
    # 2. 매크로 가중치 추출
    macro_w = state_dict['macro_weights'].numpy()
    
    # 3. 임계값(Threshold) 및 민감도(Steepness)
    threshold = state_dict['threshold'].item()
    steepness = state_dict['steepness'].item()

    ma_names = ["MA5", "MA20", "MA60", "MA200"]
    
    print("="*50)
    print("🤖 AI가 찾아낸 TQQQ 매매 황금 가중치")
    print("="*50)
    print("\n[1. 이동평균선 비중]")
    for name, prob in zip(ma_names, stock_probs):
        print(f"🔹 {name:<6} : {prob*100:>6.2f}%")
        
    print("\n[2. 매크로 영향력 (음수면 반비례)]")
    print(f"🔹 달러 인덱스 (DXY) : {macro_w[0]:>6.4f}")
    print(f"🔹 미 국채 금리 (TNX) : {macro_w[1]:>6.4f}")
    
    print("\n[3. 진입 판단 기준]")
    print(f"🔹 최소 요구 수익 (Threshold) : {threshold*100:>6.2f}%")
    print(f"🔹 신호 결정 민감도 (Steepness) : {steepness:>6.2f}")
    print("="*50)

if __name__ == "__main__":
    extract_ai_recipe()