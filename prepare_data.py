"""
prepare_data.py - 航空票价预测数据预处理脚本

本脚本用于将原始航空数据转换为模型可用的训练格式，包含：
1. 图结构数据构建（站点/路线/航班）
2. 时序数据预处理
3. 数据集划分与标准化
"""


import os
import argparse
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data, InMemoryDataset
from typing import Tuple, Dict

# 原始数据列名定义
STATION_COLS = ['station_id', 'latitude', 'longitude', 'city', 'country']
ROUTE_COLS = ['route_id', 'origin_id', 'destination_id', 'distance', 'avg_duration']
FLIGHT_COLS = ['flight_no', 'departure_date', 'aircraft_type', 'base_price']
PRICE_COLS = ['flight_no', 'query_date', 'price']

def load_raw_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """加载原始数据文件"""
    stations = pd.read_csv(os.path.join(data_dir, 'graph/stations.csv'), usecols=STATION_COLS)
    routes = pd.read_csv(os.path.join(data_dir, 'graph/routes.csv'), usecols=ROUTE_COLS)
    flights = pd.read_csv(os.path.join(data_dir, 'graph/flights.csv'), usecols=FLIGHT_COLS)
    prices = pd.read_csv(os.path.join(data_dir, 'temporal/prices.csv'), usecols=PRICE_COLS)
    return stations, routes, flights, prices

def build_graph_data(stations: pd.DataFrame, 
                    routes: pd.DataFrame,
                    flights: pd.DataFrame) -> Dict[str, Data]:
    """构建层次化图结构数据"""
    
    # 站点图：节点特征+地理连接
    station_features = stations[['latitude', 'longitude']].values
    station_edges = _create_geo_edges(stations)  # 根据地理位置创建边
    
    station_graph = Data(
        x=torch.FloatTensor(station_features),
        edge_index=torch.LongTensor(station_edges).t().contiguous()
    )
    
    # 路线图：路线连接关系
    route_edges = routes[['origin_id', 'destination_id']].values
    route_features = _extract_route_features(routes)
    
    route_graph = Data(
        x=torch.FloatTensor(route_features),
        edge_index=torch.LongTensor(route_edges).t().contiguous()
    )
    
    # 航班图：航班-路线关联
    flight_edges, flight_features = _build_flight_graph(flights, routes)
    
    flight_graph = Data(
        x=torch.FloatTensor(flight_features),
        edge_index=torch.LongTensor(flight_edges).t().contiguous()
    )
    
    return {
        'station': station_graph,
        'route': route_graph,
        'flight': flight_graph
    }

def process_temporal_data(prices: pd.DataFrame, 
                         window_size: int = 30) -> Dict[str, np.ndarray]:
    """处理时序数据并生成样本"""
    
    # 数据标准化
    scaler = MinMaxScaler()
    prices['price_norm'] = scaler.fit_transform(prices[['price']])
    
    # 按航班分组处理
    grouped = prices.groupby('flight_no')
    
    samples = []
    for flight_no, group in grouped:
        # 生成时序A（固定出发日期）
        series_a = _generate_series_a(group, window_size)
        
        # 生成时序B（固定购票间隔）
        series_b = _generate_series_b(group, window_size)
        
        # 对齐样本
        for a, b in zip(series_a, series_b):
            samples.append({
                'flight_no': flight_no,
                'series_a': a,
                'series_b': b,
                'target': a[-1]  # 假设预测最后一天
            })
    
    # 转换为numpy数组
    return {
        'series_a': np.array([s['series_a'] for s in samples]),
        'series_b': np.array([s['series_b'] for s in samples]),
        'targets': np.array([s['target'] for s in samples]),
        'scaler': {'min': scaler.min_[0], 'max': scaler.scale_[0]}
    }

def save_processed_data(data: dict, output_dir: str):
    """保存处理后的数据"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存图数据
    torch.save(data['graphs']['station'], os.path.join(output_dir, 'station_graph.pt'))
    torch.save(data['graphs']['route'], os.path.join(output_dir, 'route_graph.pt')) 
    torch.save(data['graphs']['flight'], os.path.join(output_dir, 'flight_graph.pt'))
    
    # 保存时序数据
    np.savez(os.path.join(output_dir, 'temporal.npz'),
            series_a=data['temporal']['series_a'],
            series_b=data['temporal']['series_b'],
            targets=data['temporal']['targets'])
    
    # 保存标准化参数
    with open(os.path.join(output_dir, 'scaler.json'), 'w') as f:
        json.dump(data['temporal']['scaler'], f)

def _create_geo_edges(stations: pd.DataFrame, 
                     threshold: float = 1.0) -> np.ndarray:
    """基于地理位置创建站点连接边（示例逻辑）"""
    # 实际应用中应根据真实距离计算
    edges = []
    for i in range(len(stations)):
        for j in range(i+1, len(stations)):
            if np.random.rand() < threshold:  # 简化示例
                edges.append([i, j])
    return np.array(edges)

def _extract_route_features(routes: pd.DataFrame) -> np.ndarray:
    """提取路线特征"""
    return routes[['distance', 'avg_duration']].values

def _build_flight_graph(flights: pd.DataFrame,
                       routes: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """构建航班-路线关联图"""
    # 示例逻辑：航班与路线连接
    merged = pd.merge(flights, routes, on='route_id')
    flight_edges = merged[['flight_no', 'route_id']].drop_duplicates().values
    flight_features = merged[['base_price', 'aircraft_type']].values
    return flight_edges, flight_features

def _generate_series_a(group: pd.DataFrame, 
                      window_size: int) -> list:
    """生成固定出发日期时序"""
    group = group.sort_values('query_date')
    return [group['price_norm'].iloc[i:i+window_size].values 
           for i in range(len(group)-window_size)]

def _generate_series_b(group: pd.DataFrame,
                      window_size: int) -> list:
    """生成固定购票间隔时序""" 
    # 假设按周周期处理
    return [group['price_norm'].iloc[i::7][:window_size].values  # 每周同一天采样
           for i in range(7)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data', help='原始数据目录')
    parser.add_argument('--output_dir', default='processed', help='输出目录')
    parser.add_argument('--window_size', type=int, default=30, help='时序窗口大小')
    args = parser.parse_args()
    
    # 执行预处理流程
    stations, routes, flights, prices = load_raw_data(args.data_dir)
    
    graph_data = build_graph_data(stations, routes, flights)
    temporal_data = process_temporal_data(prices, args.window_size)
    
    processed_data = {
        'graphs': graph_data,
        'temporal': temporal_data
    }
    
    save_processed_data(processed_data, args.output_dir)
    print(f"数据预处理完成！处理结果已保存至 {args.output_dir}")
