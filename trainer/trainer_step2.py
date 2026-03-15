import torch
import os

def train_risk_parameters(model, train_dataset, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # 리스크 파라미터들만 최적화
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    
    print(f"🚀 [2단계] 리스크 관리 최적화 시작...")

    for epoch in range(config['train']['epochs']):
        model.train()
        entry_price, holding = 0.0, False
        epoch_strat_returns = []
        prev_signal = torch.tensor([0.0], device=device, requires_grad=True)

        for i in range(1, len(train_dataset)):
            curr_feat, target_ret = train_dataset[i]
            prev_feat, _ = train_dataset[i-1]
            curr_price = train_dataset.df['Close'].iloc[i]

            current_ret = (curr_price - entry_price) / entry_price if holding else 0.0
            signal = model(curr_feat.to(device).unsqueeze(0), 
                           prev_feat.to(device).unsqueeze(0), 
                           torch.tensor([current_ret], device=device).float())
            
            # 미분 가능한 수익률 계산
            epoch_strat_returns.append(prev_signal * target_ret.to(device))

            if not holding and signal.item() > 0.5:
                holding, entry_price = True, curr_price
            elif holding and signal.item() < 0.5:
                holding, entry_price = False, 0.0
            
            prev_signal = signal

        if len(epoch_strat_returns) > 1:
            all_rets = torch.cat(epoch_strat_returns)
            loss = -(torch.mean(all_rets) / (torch.std(all_rets) + 1e-8))
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}] Loss: {loss.item():.4f} | SL: {model.stop_loss_threshold.item():.4f} | TP: {model.take_profit_threshold.item():.4f}")

    torch.save({'state_dict': model.state_dict()}, 'saved/checkpoint_step2.pth')