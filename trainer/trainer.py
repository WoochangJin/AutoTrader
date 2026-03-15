import torch
import torch.nn.functional as F

def train_full_strategy(model, train_dataset, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # model.parameters()에는 현재 stock_weights만 Parameter로 등록되어 있어 그것만 학습함
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    
    print(f"🚀 [1단계] 이평선 가중치(방향성) 최적화 시작...")

    for epoch in range(config['train']['epochs']):
        model.train()
        entry_price = 0.0
        holding = False
        epoch_probs, epoch_targets = [], []

        for i in range(1, len(train_dataset)):
            prev_feat, _ = train_dataset[i-1]
            curr_feat, target_ret = train_dataset[i]
            curr_price = train_dataset.df['Close'].iloc[i]
            
            prev_feat = prev_feat.to(device).unsqueeze(0)
            curr_feat = curr_feat.to(device).unsqueeze(0)
            
            current_ret = (curr_price - entry_price) / entry_price if holding else 0.0
            current_ret_tensor = torch.tensor([current_ret], device=device).float()

            prob = model(curr_feat, prev_feat, current_ret_tensor)
            
            # 정답: 내일 오르면 1, 내리면 0
            target_dir = torch.tensor([1.0 if target_ret > 0 else 0.0], device=device)
            
            epoch_probs.append(prob)
            epoch_targets.append(target_dir)

            # 시뮬레이션 상태 업데이트
            if not holding and prob.item() > 0.5:
                holding, entry_price = True, curr_price
            elif holding and prob.item() < 0.5:
                holding, entry_price = False, 0.0

        # Loss: 방향성 일치율 (BCE)
        if len(epoch_probs) > 0:
            loss = F.binary_cross_entropy(torch.cat(epoch_probs), torch.cat(epoch_targets))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}] Loss: {loss.item():.4f}")