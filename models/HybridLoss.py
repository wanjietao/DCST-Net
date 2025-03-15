class HybridLoss(nn.Module):
    def __init__(self, 
                 alpha: float = 0.3,  # 相似性损失权重
                 epsilon: float = 1e-4):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        
    def forward(self,
                pred: torch.Tensor,        # 模型预测 [B, T]
                target: torch.Tensor,      # 真实价格 [B, T]
                sim_loss: torch.Tensor     # 图相似性损失
               ) -> torch.Tensor:
        """
        混合损失计算
        """
        # 主损失函数
        mae_loss = F.l1_loss(pred, target)
        
        # MAPE计算（数值稳定版）
        abs_percent = torch.abs((target - pred) / (target + self.epsilon))
        mape_loss = torch.mean(abs_percent) * 100
        
        # 总损失组合
        total_loss = mape_loss + 0.5*mae_loss + self.alpha*sim_loss
        
        return total_loss
