"""
时间依赖图 - 从真实轨迹学习通行时间 (F9)
"""

import heapq
import math
import pickle
from collections import defaultdict
from datetime import datetime
import numpy as np


class TimePeriod:
    """时段定义"""
    MORNING_PEAK = 0    # 早高峰 7:00-9:00
    EVENING_PEAK = 1    # 晚高峰 17:00-19:00
    OFF_PEAK = 2        # 平峰
    
    @classmethod
    def get_period(cls, timestamp):
        """根据时间戳获取时段"""
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        hour = timestamp.hour
        if 7 <= hour < 9:
            return cls.MORNING_PEAK
        elif 17 <= hour < 19:
            return cls.EVENING_PEAK
        else:
            return cls.OFF_PEAK
    
    @classmethod
    def get_period_name(cls, period):
        names = {0: "早高峰 (7:00-9:00)", 1: "晚高峰 (17:00-19:00)", 2: "平峰"}
        return names.get(period, "未知")
    
    @classmethod
    def get_all_periods(cls):
        return [cls.MORNING_PEAK, cls.EVENING_PEAK, cls.OFF_PEAK]


class TimeDependentGraph:
    """时间依赖图 - 支持不同时段的最短路径"""
    
    def __init__(self):
        self.adjacency = defaultdict(list)   # node_id -> [(neighbor_id, {period: time})]
        self.nodes = {}                       # node_id -> (lon, lat)
        self.node_counter = 0
        self.edge_samples = defaultdict(dict)  # (from, to) -> {period: [times]}
    
    def add_node(self, lon, lat):
        """添加节点"""
        node_id = self.node_counter
        self.nodes[node_id] = (lon, lat)
        self.node_counter += 1
        return node_id
    
    def add_edge(self, from_id, to_id, distance_meters):
        """添加边，初始化权重为估算值"""
        self.adjacency[from_id].append((to_id, {
            TimePeriod.MORNING_PEAK: distance_meters / 8.0,
            TimePeriod.EVENING_PEAK: distance_meters / 8.0,
            TimePeriod.OFF_PEAK: distance_meters / 15.0,
        }))
        self.adjacency[to_id].append((from_id, {
            TimePeriod.MORNING_PEAK: distance_meters / 8.0,
            TimePeriod.EVENING_PEAK: distance_meters / 8.0,
            TimePeriod.OFF_PEAK: distance_meters / 15.0,
        }))
    
    def add_travel_sample(self, from_id, to_id, period, time_seconds):
        """添加一个通行时间样本（从真实轨迹学习）"""
        key = (from_id, to_id)
        if period not in self.edge_samples[key]:
            self.edge_samples[key][period] = []
        self.edge_samples[key][period].append(time_seconds)
    
    def update_all_edges_from_samples(self):
        """从收集的样本中更新所有边的通行时间（取中位数）"""
        print("📊 从轨迹样本更新边权重...")
        updated_count = 0
        
        for (from_id, to_id), period_times in self.edge_samples.items():
            for period, times in period_times.items():
                if len(times) >= 3:  # 至少3个样本才更新
                    # 使用中位数，避免异常值影响
                    median_time = np.median(times)
                    # 限制最大时间（不超过10分钟）
                    median_time = min(median_time, 600)
                    
                    # 更新正向边
                    for i, (neighbor, weights) in enumerate(self.adjacency[from_id]):
                        if neighbor == to_id:
                            weights[period] = median_time
                            updated_count += 1
                            break
                    
                    # 更新反向边
                    for i, (neighbor, weights) in enumerate(self.adjacency[to_id]):
                        if neighbor == from_id:
                            weights[period] = median_time
                            break
        
        print(f"✅ 更新了 {updated_count} 条边的权重")
        
        # 打印统计信息
        total_samples = sum(len(times) for times_dict in self.edge_samples.values() 
                           for times in times_dict.values())
        print(f"📊 总样本数: {total_samples}")
    
    def get_edge_time(self, from_id, to_id, period):
        """获取边在指定时段的通行时间"""
        for neighbor, weights in self.adjacency[from_id]:
            if neighbor == to_id:
                return weights.get(period, weights[TimePeriod.OFF_PEAK])
        return float('inf')
    
    def get_node_coords(self, node_id):
        return self.nodes.get(node_id)
    
    def haversine_distance(self, lon1, lat1, lon2, lat2):
        """计算两点间的球面距离（米）"""
        R = 6371000
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        
        a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
    
    def find_nearest_node(self, lon, lat):
        """找到距离给定点最近的节点"""
        min_dist = float('inf')
        nearest_node = None
        for node_id, (node_lon, node_lat) in self.nodes.items():
            dist = self.haversine_distance(lon, lat, node_lon, node_lat)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node_id
        return nearest_node, min_dist
    
    def dijkstra_time(self, start_id, end_id, period):
        """时间依赖 Dijkstra 算法"""
        pq = [(0, start_id)]
        dist = {start_id: 0}
        prev = {start_id: None}
        visited = set()
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            visited.add(current)
            
            if current == end_id:
                break
            
            for neighbor, weights in self.adjacency.get(current, []):
                if neighbor in visited:
                    continue
                
                travel_time = weights.get(period, weights[TimePeriod.OFF_PEAK])
                new_dist = current_dist + travel_time
                
                if new_dist < dist.get(neighbor, float('inf')):
                    dist[neighbor] = new_dist
                    prev[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))
        
        if end_id not in prev:
            return None, None
        
        # 重建路径
        path = []
        current = end_id
        while current is not None:
            path.append(current)
            current = prev.get(current)
        path.reverse()
        
        return path, dist.get(end_id)
    
    def get_path_coords(self, path):
        """根据路径节点ID列表，返回坐标列表"""
        coords = []
        for node_id in path:
            lon, lat = self.nodes[node_id]
            coords.append([lon, lat])
        return coords
    
    def get_stats(self):
        """获取图统计信息"""
        edges = sum(len(neighbors) for neighbors in self.adjacency.values()) // 2
        total_samples = sum(len(times) for times_dict in self.edge_samples.values() 
                           for times in times_dict.values())
        return {
            'nodes': len(self.nodes),
            'edges': edges,
            'samples': total_samples
        }
    
    def save(self, filepath):
        """保存到文件"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"💾 时间依赖图已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """从文件加载"""
        with open(filepath, 'rb') as f:
            graph = pickle.load(f)
        print(f"📂 时间依赖图已从文件加载: {filepath}")
        return graph


class TimeDependentRoadNetworkBuilder:
    """从真实轨迹数据构建时间依赖路网"""
    
    def __init__(self, grid_size=100):
        self.grid_size = grid_size
    
    def build_from_loader(self, data_loader, bounds, sample_vehicles=500, learn_from_trajectories=True):
        """
        构建时间依赖路网
        
        Args:
            data_loader: TDriveDataLoader 实例
            bounds: (x_min, y_min, x_max, y_max)
            sample_vehicles: 采样车辆数（构建图用）
            learn_from_trajectories: 是否从轨迹学习通行时间
        """
        print("🏗️ 开始构建时间依赖路网...")
        print(f"   网格大小: {self.grid_size}x{self.grid_size}")
        print(f"   采样车辆: {sample_vehicles}")
        print(f"   轨迹学习: {'启用' if learn_from_trajectories else '禁用'}")
        
        x_min, y_min, x_max, y_max = bounds
        x_step = (x_max - x_min) / self.grid_size
        y_step = (y_max - y_min) / self.grid_size
        
        graph = TimeDependentGraph()
        grid_to_node = {}
        
        # 第一步：收集所有车辆的点，用于确定节点位置
        vehicle_list = data_loader.get_vehicle_list()
        sampled_vehicles = vehicle_list[:sample_vehicles]
        
        print(f"📊 采样 {len(sampled_vehicles)} 辆车构建节点...")
        
        # 收集所有点
        all_points = []
        for vid in sampled_vehicles:
            df = data_loader.load_vehicle_trajectory(vid)
            if df is not None and len(df) > 0:
                # 每隔3个点取一个
                sample_df = df.iloc[::3]
                for _, row in sample_df.iterrows():
                    lon = float(row['longitude'])
                    lat = float(row['latitude'])
                    if x_min <= lon <= x_max and y_min <= lat <= y_max:
                        all_points.append((lon, lat))
        
        print(f"📍 收集到 {len(all_points)} 个轨迹点")
        
        # 创建网格节点
        for lon, lat in all_points:
            grid_x = int((lon - x_min) / x_step)
            grid_y = int((lat - y_min) / y_step)
            grid_x = min(grid_x, self.grid_size - 1)
            grid_y = min(grid_y, self.grid_size - 1)
            grid_key = (grid_x, grid_y)
            
            if grid_key not in grid_to_node:
                center_lon = x_min + (grid_x + 0.5) * x_step
                center_lat = y_min + (grid_y + 0.5) * y_step
                node_id = graph.add_node(center_lon, center_lat)
                grid_to_node[grid_key] = node_id
        
        print(f"🏙️ 生成 {len(grid_to_node)} 个节点")
        
        # 第二步：添加边（相邻网格连接）
        for (grid_x, grid_y), node_id in grid_to_node.items():
            neighbors = [
                (grid_x + 1, grid_y), (grid_x - 1, grid_y),
                (grid_x, grid_y + 1), (grid_x, grid_y - 1),
            ]
            for nx, ny in neighbors:
                neighbor_key = (nx, ny)
                if neighbor_key in grid_to_node:
                    neighbor_id = grid_to_node[neighbor_key]
                    lon1, lat1 = graph.get_node_coords(node_id)
                    lon2, lat2 = graph.get_node_coords(neighbor_id)
                    distance = graph.haversine_distance(lon1, lat1, lon2, lat2)
                    graph.add_edge(node_id, neighbor_id, distance)
        
        print(f"🔗 添加 {graph.get_stats()['edges']} 条边")
        
        # 第三步：从真实轨迹学习通行时间
        if learn_from_trajectories:
            print("📊 开始从真实轨迹学习通行时间...")
            self._learn_travel_times(graph, data_loader, grid_to_node, bounds, sampled_vehicles)
        
        # 更新边权重
        graph.update_all_edges_from_samples()
        
        stats = graph.get_stats()
        print(f"✅ 时间依赖路网构建完成")
        print(f"   节点数: {stats['nodes']}")
        print(f"   边数: {stats['edges']}")
        print(f"   样本数: {stats['samples']}")
        
        return graph
    
    def _learn_travel_times(self, graph, data_loader, grid_to_node, bounds, sampled_vehicles):
        """从轨迹学习通行时间"""
        x_min, y_min, x_max, y_max = bounds
        x_step = (x_max - x_min) / self.grid_size
        y_step = (y_max - y_min) / self.grid_size
        
        sample_count = 0
        edge_sample_count = 0
        
        for vid in sampled_vehicles:
            df = data_loader.load_vehicle_trajectory(vid)
            if df is None or len(df) < 5:
                continue
            
            # 将轨迹点映射到节点序列
            node_sequence = []
            time_sequence = []
            
            for _, row in df.iterrows():
                lon = float(row['longitude'])
                lat = float(row['latitude'])
                ts = row['timestamp']
                
                if x_min <= lon <= x_max and y_min <= lat <= y_max:
                    grid_x = int((lon - x_min) / x_step)
                    grid_y = int((lat - y_min) / y_step)
                    grid_x = min(grid_x, self.grid_size - 1)
                    grid_y = min(grid_y, self.grid_size - 1)
                    grid_key = (grid_x, grid_y)
                    
                    if grid_key in grid_to_node:
                        node_sequence.append(grid_to_node[grid_key])
                        time_sequence.append(ts)
            
            # 计算相邻节点间的通行时间
            for i in range(len(node_sequence) - 1):
                if node_sequence[i] == node_sequence[i+1]:
                    continue
                
                from_node = node_sequence[i]
                to_node = node_sequence[i+1]
                
                # 时间差（秒）
                time_diff = (time_sequence[i+1] - time_sequence[i]).total_seconds()
                
                # 过滤异常值：时间差应在 1秒 到 5分钟 之间
                if 1 <= time_diff <= 300:
                    period = TimePeriod.get_period(time_sequence[i])
                    graph.add_travel_sample(from_node, to_node, period, time_diff)
                    edge_sample_count += 1
            
            sample_count += 1
            if sample_count % 50 == 0:
                print(f"   已处理 {sample_count} 辆车，收集 {edge_sample_count} 个样本...")
        
        print(f"   轨迹学习完成: 处理 {sample_count} 辆车，收集 {edge_sample_count} 个样本")


def get_time_dependent_graph(data_loader=None, bounds=None, rebuild=False, 
                             cache_file="time_dependent_graph.pkl", 
                             sample_vehicles=500,
                             learn_from_trajectories=True):
    """获取时间依赖图单例（带缓存）"""
    import os
    
    if not rebuild and os.path.exists(cache_file):
        return TimeDependentGraph.load(cache_file)
    
    if data_loader is None or bounds is None:
        raise ValueError("首次构建需要提供 data_loader 和 bounds")
    
    builder = TimeDependentRoadNetworkBuilder(grid_size=500)
    graph = builder.build_from_loader(data_loader, bounds, 
                                       sample_vehicles=sample_vehicles,
                                       learn_from_trajectories=learn_from_trajectories)
    graph.save(cache_file)
    return graph