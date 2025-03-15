# 初始化组件
model = AirfarePredictor(gnn_dim=128, pred_steps=7)
train_loader = create_dataloader(train_dataset, batch_size=32)
val_loader = create_dataloader(val_dataset, batch_size=32, shuffle=False)

# 训练管道
pipeline = TrainingPipeline(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cuda',
    init_lr=1e-4,
    early_stop=10
)

# 启动训练
pipeline.run(epochs=100)

# 加载最佳模型
model.load_state_dict(torch.load('best_model_epoch_v001.pt'))
