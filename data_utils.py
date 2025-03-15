import torch
from torch_geometric.data import Data, Batch
import numpy as np


def preprocess_prediction_data(samples, graph_data_dir):
    # 加载图元数据
    stations = load_station_metadata(f"{graph_data_dir}/stations.csv")
    routes = load_route_connections(f"{graph_data_dir}/routes.csv")
    
    processed_samples = []
    for sample in samples:
        # 构建图数据
        site_graph = build_site_graph(sample['route'], stations)
        route_graph = build_route_graph(sample['route'], routes)
        flight_graph = build_flight_graph(sample['route'])
        
        # 标准化时序数据
        ts_a = normalize_series(sample['historical_prices']['series_a'])
        ts_b = normalize_series(sample['historical_prices']['series_b'])
        
        processed_samples.append({
            'site_graph': site_graph,
            'route_graph': route_graph,
            'flight_graph': flight_graph,
            'ts_a': ts_a,
            'ts_b': ts_b
        })
    
    # 批量处理图数据
    site_batch = Batch.from_data_list([s['site_graph'] for s in processed_samples])
    route_batch = Batch.from_data_list([s['route_graph'] for s in processed_samples])
    flight_batch = Batch.from_data_list([s['flight_graph'] for s in processed_samples])
    
    return {
        'site_graph': site_batch,
        'route_graph': route_batch,
        'flight_graph': flight_batch,
        'ts_a': np.array([s['ts_a'] for s in processed_samples]),
        'ts_b': np.array([s['ts_b'] for s in processed_samples]),
        'price_scaler': {'min': 300, 'max': 800}  # 从训练数据中实际获取
    }

def normalize_series(series):
    # 实际应根据训练数据统计进行标准化
    return (np.array(series) - 300) / 500  
