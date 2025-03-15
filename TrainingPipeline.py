import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

class TrainingPipeline:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = 'cuda',
                 init_lr: float = 1e-4,
                 early_stop: int = 10):
        """
        完整训练管理类
        Args:
            model: 待训练模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 训练设备
            init_lr: 初始学习率
            early_stop: 早停轮次
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.early_stop = early_stop
        
        # 优化器配置
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=init_lr,
            weight_decay=1e-5
        )
        
        # 学习率调度
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # 损失函数
        self.criterion = HybridLoss(alpha=0.3)
        
        # 训练状态跟踪
        self.best_val_loss = float('inf')
        self.no_improve_epochs = 0

    def train_epoch(self) -> float:
        """ 单个训练epoch """
        self.model.train()
        total_loss = 0.0
        
        for batch in self.train_loader:
            # 数据解包 (根据实际数据格式调整)
            (site_data, route_data, flight_data), ts_a, ts_b, targets = batch
            ts_a = ts_a.to(self.device)
            ts_b = ts_b.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播
            graph_feat, sim_loss = self.model.gnn(site_data, route_data, flight_data)
            trend_pred, periodic_pred = self.model.decoder(ts_a, ts_b, graph_feat)
            final_pred = self.model.fusion(trend_pred, periodic_pred)
            
            # 损失计算
            loss = self.criterion(final_pred, targets, sim_loss)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)

    def validate(self) -> float:
        """ 验证步骤 """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                (site_data, route_data, flight_data), ts_a, ts_b, targets = batch
                ts_a = ts_a.to(self.device)
                ts_b = ts_b.to(self.device)
                targets = targets.to(self.device)
                
                graph_feat, sim_loss = self.model.gnn(site_data, route_data, flight_data)
                trend_pred, periodic_pred = self.model.decoder(ts_a, ts_b, graph_feat)
                final_pred = self.model.fusion(trend_pred, periodic_pred)
                
                loss = self.criterion(final_pred, targets, sim_loss)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)

    def run(self, epochs: int = 100):
        """ 完整训练循环 """
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            # 学习率调整
            self.scheduler.step(val_loss)
            
            # 早停机制
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.no_improve_epochs = 0
                torch.save(self.model.state_dict(), f'best_model_epoch{epoch}.pt')
            else:
                self.no_improve_epochs += 1
                if self.no_improve_epochs >= self.early_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # 训练状态输出
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            print("-"*50)
