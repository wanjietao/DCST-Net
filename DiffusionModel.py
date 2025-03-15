import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class DiffusionModel(nn.Module):
    def __init__(self,
                 cond_dim: int = 384,       # 条件特征维度（图特征+时序）
                 seq_len: int = 30,         # 输入序列长度
                 pred_steps: int = 7,       # 预测步长
                 noise_dim: int = 64,       # 噪声隐空间维度
                 num_timesteps: int = 1000, # 扩散总步数
                 schedule: str = 'cosine'): # 噪声调度策略
        super().__init__()
        self.num_timesteps = num_timesteps
        self.pred_steps = pred_steps
        self.seq_len = seq_len
        
        # ================= 噪声调度策略 =================
        self.register_buffer('betas', self._get_beta_schedule(schedule, num_timesteps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - self.alphas_cumprod))

        # =============== 条件噪声预测网络 ================
        self.noise_pred_net = nn.Sequential(
            # 时间步嵌入
            TimestepEmbedding(num_timesteps, 64),
            # 条件特征处理
            nn.Linear(cond_dim, 256),
            nn.GELU(),
            # 时空交叉注意力
            SpatialTemporalAttention(seq_len, noise_dim, 256),
            # 残差块
            ResidualBlock(256),
            # 输出层
            nn.Linear(256, seq_len)
        )

        # ================ 初始化参数 ==================
        self._init_weights()

    def forward(self, 
                x: torch.Tensor,          # 输入序列 [B, seq_len]
                cond_feat: torch.Tensor,   # 条件特征 [B, cond_dim]
                noise: torch.Tensor = None,
                t: torch.Tensor = None) -> torch.Tensor:
        """
        训练阶段前向传播
        Args:
            x: 原始输入序列
            cond_feat: 条件特征（图特征+时序特征）
            noise: 可选预先生成的噪声
            t: 可选指定时间步
        Returns:
            预测的噪声 [B, seq_len]
        """
        # 随机采样时间步
        if t is None:
            t = torch.randint(0, self.num_timesteps, (x.size(0), device=x.device)
        
        # 前向扩散过程
        x_noisy, noise_real = self.q_sample(x, t, noise)
        
        # 预测噪声
        noise_pred = self.noise_pred_net(x_noisy, cond_feat, t)
        
        return noise_pred

    def q_sample(self, 
                x0: torch.Tensor, 
                t: torch.Tensor, 
                noise: torch.Tensor = None) -> torch.Tensor:
        """
        前向扩散过程（加噪）
        Args:
            x0: 原始序列 [B, seq_len]
            t: 时间步 [B,]
            noise: 可选噪声
        Returns:
            加噪后的序列 [B, seq_len]
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        x_noisy = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha * noise
        return x_noisy, noise

    def p_sample(self, 
                x: torch.Tensor,
                cond_feat: torch.Tensor,
                t: int) -> torch.Tensor:
        """
        反向去噪过程（单步）
        Args:
            x: 当前噪声序列 [B, seq_len]
            cond_feat: 条件特征 [B, cond_dim]
            t: 当前时间步
        Returns:
            去噪后的序列 [B, seq_len]
        """
        # 预测噪声
        noise_pred = self.noise_pred_net(x, cond_feat, t)
        
        # 计算去噪结果
        alpha_t = self.alphas[t]
        beta_t = self.betas[t]
        
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        
        x_prev = (1 / sqrt(alpha_t)) * (x - beta_t / sqrt(1 - self.alphas_cumprod[t]) * noise_pred)
        x_prev += sqrt(beta_t) * noise
        
        return x_prev

    def predict(self,
                cond_feat: torch.Tensor) -> torch.Tensor:
        """
        完整反向生成过程
        Args:
            cond_feat: 条件特征 [B, cond_dim]
        Returns:
            预测序列 [B, pred_steps]
        """
        # 初始化随机噪声
        x = torch.randn(
            (cond_feat.size(0), self.seq_len),
            device=cond_feat.device
        )
        
        # 逐步去噪
        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(x, cond_feat, t)
        
        # 提取预测部分
        return x[:, -self.pred_steps:]

    def _get_beta_schedule(self, 
                          schedule: str,
                          num_timesteps: int) -> torch.Tensor:
        """ 生成beta调度表 """
        if schedule == 'linear':
            return torch.linspace(1e-4, 0.02, num_timesteps)
        elif schedule == 'cosine':
            steps = torch.arange(num_timesteps + 1, dtype=torch.float32)
            s = 0.008
            f = torch.cos((steps / num_timesteps + s) / (1 + s) * torch.pi * 0.5) ** 2
            return torch.clip(1 - f[1:] / f[:-1], 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def _init_weights(self):
        """ 参数初始化 """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class TimestepEmbedding(nn.Module):
    """ 时间步嵌入模块 """
    def __init__(self, num_timesteps: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Embedding(num_timesteps, embed_dim)
        
    def forward(self, x: torch.Tensor, cond_feat: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_embed = self.embed(t).unsqueeze(1)  # [B, 1, D]
        cond_feat = cond_feat.unsqueeze(1)    # [B, 1, D_cond]
        fused = torch.cat([x, t_embed, cond_feat], dim=-1)
        return fused

class SpatialTemporalAttention(nn.Module):
    """ 时空交叉注意力模块 """
    def __init__(self, seq_len: int, noise_dim: int, feat_dim: int):
        super().__init__()
        self.query = nn.Linear(noise_dim, feat_dim)
        self.key = nn.Linear(feat_dim, feat_dim)
        self.value = nn.Linear(feat_dim, feat_dim)
        
    def forward(self, x: torch.Tensor, cond_feat: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, D_noise]
        # cond_feat: [B, D_cond]
        B, L, D = x.shape
        
        # 扩展条件特征
        cond_feat = cond_feat.unsqueeze(1).expand(-1, L, -1)  # [B, L, D_cond]
        
        # 计算注意力
        q = self.query(x)  # [B, L, D]
        k = self.key(cond_feat)  
        v = self.value(cond_feat)
        
        attn = F.softmax(torch.bmm(q, k.transpose(1,2)) / sqrt(D)
        out = torch.bmm(attn, v) + x  # 残差连接
        return out

class ResidualBlock(nn.Module):
    """ 残差块 """
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)
