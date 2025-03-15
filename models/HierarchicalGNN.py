import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Batch

class HierarchicalGNN(nn.Module):
    def __init__(self,
                 node_dim: int = 32,
                 hidden_dim: int = 128,
                 num_heads: tuple = (3, 2, 2),
                 dropout: float = 0.1):
        """
        层次化图神经网络模块
        Args:
            node_dim (int): 节点特征维度
            hidden_dim (int): 隐藏层维度
            num_heads (tuple): 各层级GAT的头数 (site, route, flight)
            dropout (float): Dropout概率
        """
        super().__init__()
        
        # ----------------- 站点图网络 -----------------
        self.site_gnn = nn.ModuleList([
            GATConv(in_channels=node_dim,
                    out_channels=hidden_dim // num_heads[0],  # 保证输出维度一致
                    heads=num_heads[0],
                    dropout=dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            GATConv(in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=1,  # 第二层单头
                    concat=False,
                    dropout=dropout)
        ])
        
        # ----------------- 路线图网络 -----------------
        self.route_gnn = nn.ModuleList([
            GATConv(node_dim, hidden_dim // num_heads[1],
                    heads=num_heads[1],
                    dropout=dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            GATConv(hidden_dim, hidden_dim,
                    heads=1,
                    concat=False,
                    dropout=dropout)
        ])
        
        # ----------------- 航班图网络 -----------------
        self.flight_gnn = nn.ModuleList([
            GATConv(node_dim, hidden_dim // num_heads[2],
                    heads=num_heads[2],
                    dropout=dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            GATConv(hidden_dim, hidden_dim,
                    heads=1,
                    concat=False,
                    dropout=dropout)
        ])
        
        # ----------------- 相似性约束 -----------------
        self.sim_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
        self.sim_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 16)  # 降维到低维空间计算相似度
        )

    def forward(self,
                site_data: Batch,
                route_data: Batch,
                flight_data: Batch) -> tuple:
        """
        前向传播
        Args:
            site_data (Batch): 站点图数据，包含:
                - x: (num_nodes, node_dim)
                - edge_index: (2, num_edges)
                - batch: (num_nodes,) 用于图池化的批索引
            route_data (Batch): 路线图数据
            flight_data (Batch): 航班图数据
        
        Returns:
            tuple: (combined_feat, sim_loss)
            - combined_feat: 聚合特征 (batch_size, 3*hidden_dim)
            - sim_loss: 相似性约束损失值
        """
        
        # =============== 站点表征学习 ===============
        x = site_data.x
        edge_index = site_data.edge_index
        for layer in self.site_gnn:
            if isinstance(layer, GATConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        site_feat = x
        
        # =============== 路线表征学习 ===============
        x = route_data.x
        edge_index = route_data.edge_index
        for layer in self.route_gnn:
            if isinstance(layer, GATConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        route_feat = x
        
        # =============== 航班表征学习 ===============
        x = flight_data.x
        edge_index = flight_data.edge_index
        for layer in self.flight_gnn:
            if isinstance(layer, GATConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        flight_feat = x
        
        # =============== 层次化特征聚合 ===============
        site_pool = global_mean_pool(site_feat, site_data.batch)  # (batch_size, hidden_dim)
        route_pool = global_mean_pool(route_feat, route_data.batch)
        flight_pool = global_mean_pool(flight_feat, flight_data.batch)
        combined_feat = torch.cat([site_pool, route_pool, flight_pool], dim=1)  # (batch_size, 3*hidden_dim)
        
        # =============== 相似性约束计算 ===============
        # 从同一批次随机采样三元组
        anchor_idx = torch.randint(0, site_pool.size(0), (site_pool.size(0)//2,))
        pos_idx = (anchor_idx + 1) % site_pool.size(0)  # 模拟相似样本
        neg_idx = torch.randint(0, site_pool.size(0), (anchor_idx.size(0),)
        
        # 投影到低维空间
        anchor = self.sim_proj(site_pool[anchor_idx])
        positive = self.sim_proj(site_pool[pos_idx])
        negative = self.sim_proj(site_pool[neg_idx])
        
        sim_loss = self.sim_loss_fn(anchor, positive, negative)
        
        return combined_feat, sim_loss

    def _init_weights(self):
        """ 参数初始化 """
        for m in self.modules():
            if isinstance(m, GATConv):
                nn.init.xavier_normal_(m.lin_src.weight)
                nn.init.xavier_normal_(m.lin_dst.weight)
                nn.init.xavier_normal_(m.att)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
