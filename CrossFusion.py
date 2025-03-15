import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossFusion(nn.Module):
    def __init__(self,
                 trend_dim: int = 7,      # 趋势预测维度
                 period_dim: int = 7,     # 周期预测维度
                 hidden_dim: int = 32,    # 隐层维度
                 num_heads: int = 2,      # 注意力头数
                 use_residual: bool = True # 是否使用残差连接
                ):
        super().__init__()
        self.use_residual = use_residual

        # =============== 交叉注意力机制 ===============
        self.trend_proj = nn.Linear(trend_dim, hidden_dim)
        self.period_proj = nn.Linear(period_dim, hidden_dim)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # =============== 门控融合模块 ================
        self.gate_net = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # =============== 输出投影层 ================
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, trend_dim)  # 保持输出维度与输入一致
        )
        
        # =============== 归一化层 ================
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 初始化参数
        self._init_weights()

    def forward(self,
                trend_feat: torch.Tensor,  # 趋势预测 [B, T]
                period_feat: torch.Tensor  # 周期预测 [B, T]
               ) -> torch.Tensor:
        """
        双通道特征融合
        Args:
            trend_feat: 趋势分支预测结果 [batch_size, pred_steps]
            period_feat: 周期分支预测结果 [batch_size, pred_steps]
        Returns:
            融合后的预测结果 [batch_size, pred_steps]
        """
        # 特征投影
        Q = self.trend_proj(trend_feat.unsqueeze(1))  # [B, 1, D]
        K = self.period_proj(period_feat.unsqueeze(1))
        V = K
        
        # 交叉注意力
        attn_out, _ = self.cross_attn(
            query=Q,
            key=K,
            value=V
        )  # [B, 1, D]
        
        # 门控融合
        trend_flat = self.trend_proj(trend_feat)  # [B, D]
        period_flat = self.period_proj(period_feat)
        combined = torch.cat([trend_flat, period_flat], dim=-1)
        
        gate = self.gate_net(combined)  # [B, D]
        fused = gate * trend_flat + (1 - gate) * period_flat
        
        # 残差连接
        if self.use_residual:
            fused = fused + attn_out.squeeze(1)
        
        # 输出投影
        fused = self.layer_norm(fused)
        output = self.output_proj(fused)
        
        return output

    def _init_weights(self):
        """ 参数初始化 """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def get_attention_map(self,
                         trend_feat: torch.Tensor,
                         period_feat: torch.Tensor) -> torch.Tensor:
        """
        获取注意力权重（可视化用）
        Returns:
            注意力权重矩阵 [B, num_heads, 1, 1]
        """
        Q = self.trend_proj(trend_feat.unsqueeze(1))
        K = self.period_proj(period_feat.unsqueeze(1))
        
        _, attn_weights = self.cross_attn(
            Q, K, K,
            average_attn_weights=False
        )
        return attn_weights
