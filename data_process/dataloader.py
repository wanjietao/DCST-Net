def dataloader(dataset: dict, 
                     batch_size: int = 32,
                     shuffle: bool = True):
                       
    """
    创建PyTorch数据加载器
    Args:
        dataset: 包含以下键的字典
            - site_graphs: 站点图数据列表
            - route_graphs: 路线图数据列表  
            - flight_graphs: 航班图数据列表
            - ts_a: 时序A数据 [num_samples, seq_len_a]
            - ts_b: 时序B数据 [num_samples, seq_len_b]
            - targets: 目标价格 [num_samples, pred_steps]
    """
    # 转换为PyG Batch对象
    site_batch = Batch.from_data_list(dataset['site_graphs'])
    route_batch = Batch.from_data_list(dataset['route_graphs'])
    flight_batch = Batch.from_data_list(dataset['flight_graphs'])
    
    # 创建TensorDataset
    full_dataset = TensorDataset(
        site_batch,
        route_batch,
        flight_batch,
        torch.FloatTensor(dataset['ts_a']),
        torch.FloatTensor(dataset['ts_b']),
        torch.FloatTensor(dataset['targets'])
    )
    
    return DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )
