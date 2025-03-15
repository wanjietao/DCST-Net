import torch
import torch.nn as nn
import torch.nn.functional as F

class PyramidLSTM(nn.Module):
    def __init__(self,
                 input_dim: int = 385,         # 输入维度 (图特征384 + 时序特征1)
                 pred_steps: int = 7,         # 预测步长
                 pyramid_levels: list = [7, 14, 21],  # 多周期视野
                 lstm_hidden: int = 64,       # LSTM隐藏单元数
                 num_layers: int = 2,         # LSTM层数
                 use_attention: bool = True): # 是否使用注意力融合
        super().__init__()
        self.pyramid_levels = pyramid_levels
        self.use_attention = use_attention

        # ================= 多尺度LSTM网络 =================
        self.lstm_towers = nn.ModuleList([
            nn.LSTM(input_size=input_dim,
                    hidden_size=lstm_hidden,
                    num_layers=num_layers,
                    batch_first=True,
                    bidirectional=False)
            for _ in pyramid_levels
        ])
        
        # ================= 特征融合组件 ==================
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_hidden,
                num_heads=4,
                dropout=0.1
            )
            self.fusion = nn.Sequential(
                nn.Linear(lstm_hidden * len(pyramid_levels), 128),
                nn.ReLU(),
                nn.Linear(128, pred_steps)
            )
        else:
            self.fusion = nn.Linear(lstm_hidden * len(pyramid_levels), pred_steps)

        # ================ 时序位置编码 ================
        self.pos_encoder = PositionalEncoding(lstm_hidden)

    def forward(self, 
                x: torch.Tensor,              # 输入时序数据 [B, L]
                graph_feat: torch.Tensor      # 图特征 [B, 384]
               ) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 时序数据B [batch_size, seq_len]
            graph_feat: 图特征 [batch_size, 384]
        Returns:
            周期预测结果 [batch_size, pred_steps]
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # 特征拼接 (时序数据 + 图特征)
        x = x.unsqueeze(-1)  # [B, L, 1]
        graph_feat = graph_feat.unsqueeze(1).expand(-1, seq_len, -1)  # [B, L, 384]
        combined = torch.cat([x, graph_feat], dim=-1)  # [B, L, 385]
        
        # 多尺度特征提取
        pyramid_features = []
        for level, lstm in zip(self.pyramid_levels, self.lstm_towers):
            # 下采样处理
            pooled = self._temporal_pooling(combined, level)  # [B, L//level, 385]
            
            # 位置编码增强
            pooled = self.pos_encoder(pooled)
            
            # LSTM处理
            lstm_out, _ = lstm(pooled)  # [B, L//level, lstm_hidden]
            
            # 取最后时间步特征
            last_step = lstm_out[:, -1, :]  # [B, lstm_hidden]
            pyramid_features.append(last_step)
        
        # 多尺度特征融合
        if self.use_attention:
            # 注意力融合
            attn_in = torch.stack(pyramid_features, dim=1)  # [B, num_levels, lstm_hidden]
            attn_out, _ = self.attention(
                attn_in.transpose(0,1),  # [num_levels, B, lstm_hidden]
                attn_in.transpose(0,1),
                attn_in.transpose(0,1)
            )  # [num_levels, B, lstm_hidden]
            fused = attn_out.transpose(0,1).reshape(batch_size, -1)  # [B, num_levels*lstm_hidden]
        else:
            # 直接拼接
            fused = torch.cat(pyramid_features, dim=-1)  # [B, num_levels*lstm_hidden]
        
        # 最终预测
        return self.fusion(fused)  # [B, pred_steps]

    def _temporal_pooling(self, 
                         x: torch.Tensor, 
                         pool_size: int) -> torch.Tensor:
        """
        时序金字塔池化
        Args:
            x: 输入特征 [B, L, D]
            pool_size: 池化窗口大小
        Returns:
            池化后特征 [B, L//pool_size, D]
        """
        L = x.size(1)
        if L % pool_size != 0:
            pad_size = pool_size - (L % pool_size)
            x = F.pad(x, (0,0,0,pad_size))
        
        # 转换为3D张量进行池化 [B, L/pool_size, pool_size, D]
        x_pool = x.view(x.size(0), -1, pool_size, x.size(2))
        # 平均池化 [B, new_L, D]
        return x_pool.mean(dim=2)

class PositionalEncoding(nn.Module):
    """ 可学习的位置编码 """
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        self.position_emb = nn.Embedding(max_len, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device)
        pos_emb = self.position_emb(positions).unsqueeze(0)
        return x + pos_emb
