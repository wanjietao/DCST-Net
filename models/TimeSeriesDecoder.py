import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeSeriesDecoder(nn.Module):
    def __init__(self,
                 input_dim: int = 384,  # 默认输入维度=3*128 (来自HierarchicalGNN的输出)
                 pred_steps: int = 7,
                 trend_hidden: int = 64,
                 pyramid_levels: list = [7, 14, 21],
                 use_positional_encoding: bool = True):
        """
        时序解码器模块
        Args:
            input_dim: 输入特征维度 (图特征 + 原始时序特征的拼接)
            pred_steps: 预测步长
            trend_hidden: 趋势扩散模型的隐藏层维度
            pyramid_levels: 周期金字塔的层级配置
            use_positional_encoding: 是否使用时序位置编码
        """
        super().__init__()
        self.pred_steps = pred_steps
        self.use_pos_enc = use_positional_encoding

        # ====================== 趋势扩散分支 ======================
        self.trend_diffusion = nn.ModuleDict({
            'noise_net': nn.Sequential(
                nn.Linear(input_dim + 1, trend_hidden),  # +1为时间步嵌入
                nn.GELU(),
                nn.LayerNorm(trend_hidden),
                nn.Linear(trend_hidden, input_dim)
            ),
            'time_emb': nn.Embedding(1000, 1),  # 扩散时间步嵌入
            'output_norm': nn.LayerNorm(input_dim)
        })

        # ====================== 周期金字塔分支 ======================
        self.periodic_pyramid = nn.ModuleDict({
            'pyramid_lstms': nn.ModuleList([
                nn.LSTM(input_size=input_dim,
                        hidden_size=32,
                        num_layers=2,
                        batch_first=True)
                for _ in pyramid_levels
            ]),
            'pyramid_levels': pyramid_levels,
            'attention': nn.MultiheadAttention(embed_dim=32, num_heads=2),
            'fusion': nn.Sequential(
                nn.Linear(32 * len(pyramid_levels), 64),
                nn.ReLU(),
                nn.Linear(64, pred_steps)
            )
        })

        # ====================== 共享组件 ======================
        if use_positional_encoding:
            self.position_enc = PositionalEncoding(d_model=input_dim)

    def forward(self,
                hist_price_a: torch.Tensor,  # 时序A: (B, L_a)
                hist_price_b: torch.Tensor,  # 时序B: (B, L_b)
                graph_feat: torch.Tensor     # 图特征: (B, input_dim)
               ) -> tuple:
        """
        前向传播
        Args:
            hist_price_a: 时序数据A [batch_size, seq_len_a]
            hist_price_b: 时序数据B [batch_size, seq_len_b] 
            graph_feat: 图聚合特征 [batch_size, input_dim]
        
        Returns:
            tuple: (trend_pred, periodic_pred)
            - trend_pred: 趋势预测结果 [batch_size, pred_steps]
            - periodic_pred: 周期预测结果 [batch_size, pred_steps]
        """
        # 特征拼接与增强
        batch_size = graph_feat.size(0)
        fused_feat_a = self._fuse_features(graph_feat, hist_price_a)  # (B, input_dim)
        fused_feat_b = self._fuse_features(graph_feat, hist_price_b)
        
        # =============== 趋势扩散预测 ===============
        # 扩散过程
        noisy_series = self._forward_diffusion(hist_price_a)
        
        # 反向去噪
        trend_pred = []
        for t in reversed(range(100)):
            time_emb = self.trend_diffusion['time_emb'](
                torch.tensor([t], device=graph_feat.device)
            ).view(1, 1).expand(batch_size, -1)
            
            noise_pred = self.trend_diffusion['noise_net'](
                torch.cat([noisy_series[:, t], fused_feat_a, time_emb], dim=-1)
            )
            noisy_series[:, t] = (noisy_series[:, t] - (1 - self.betas[t]) * noise_pred) / self.alphas[t].sqrt()
        
        trend_pred = self.trend_diffusion['output_norm'](
            noisy_series[:, -self.pred_steps:]
        )  # (B, pred_steps)

        # =============== 周期金字塔预测 ===============
        # 多尺度特征提取
        pyramid_outputs = []
        for level, lstm in zip(self.periodic_pyramid['pyramid_levels'], 
                             self.periodic_pyramid['pyramid_lstms']):
            # 下采样
            pooled = F.avg_pool1d(
                hist_price_b.unsqueeze(1), 
                kernel_size=level, 
                stride=level
            ).squeeze(1)  # (B, L//level)
            
            # 时序特征扩展
            expanded_feat = fused_feat_b.unsqueeze(1).expand(-1, pooled.size(1), -1)
            lstm_input = torch.cat([pooled.unsqueeze(-1), expanded_feat], dim=-1)
            
            # LSTM处理
            lstm_out, _ = lstm(lstm_input)  # (B, L//level, 32)
            pyramid_outputs.append(lstm_out[:, -1, :])  # 取最后时间步
        
        # 注意力融合
        attn_in = torch.stack(pyramid_outputs, dim=1)  # (B, num_levels, 32)
        attn_out, _ = self.periodic_pyramid['attention'](
            attn_in, attn_in, attn_in
        )
        
        # 最终预测
        periodic_pred = self.periodic_pyramid['fusion'](
            attn_out.view(batch_size, -1)
        )  # (B, pred_steps)

        return trend_pred, periodic_pred

    def _fuse_features(self, graph_feat: torch.Tensor, 
                      ts_data: torch.Tensor) -> torch.Tensor:
        """
        融合图特征与时序特征
        Args:
            graph_feat: (B, D_graph)
            ts_data: (B, L)
        
        Returns:
            fused_feat: (B, D_input)
        """
        # 时序特征编码
        ts_feat = ts_data.mean(dim=-1)  # (B,)
        
        # 拼接图特征
        fused = torch.cat([
            graph_feat, 
            ts_feat.unsqueeze(-1)
        ], dim=-1)  # (B, D_graph + 1)
        
        # 位置编码增强
        if self.use_pos_enc:
            fused = self.position_enc(fused)
            
        return fused

    def _forward_diffusion(self, x: torch.Tensor, num_steps: int = 100) -> torch.Tensor:
        """ 前向扩散过程 """
        self.betas = torch.linspace(1e-4, 0.1, num_steps, device=x.device)
        self.alphas = 1 - self.betas
        
        x_t = x.clone()
        for t in range(num_steps):
            noise = torch.randn_like(x_t)
            x_t = torch.sqrt(self.alphas[t]) * x_t + torch.sqrt(self.betas[t]) * noise
        
        return x_t

class PositionalEncoding(nn.Module):
    """ 时序位置编码器 """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.encoding = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.xavier_normal_(self.encoding)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.encoding[:seq_len]
