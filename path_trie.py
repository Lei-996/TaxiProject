"""
路径前缀树 - 用于频繁路径挖掘 (F7/F8)
"""

import pickle
from collections import deque
import math


class TrieNode:
    """前缀树节点"""
    def __init__(self):
        self.children = {}
        self.count = 0
        self.is_end = False
        self.end_count = 0


class PathTrie:
    """路径前缀树"""
    
    def __init__(self):
        self.root = TrieNode()
        self.total_paths = 0
    
    def insert(self, path_sequence):
        if not path_sequence:
            return
        
        node = self.root
        for grid_id in path_sequence:
            if grid_id not in node.children:
                node.children[grid_id] = TrieNode()
            node = node.children[grid_id]
            node.count += 1
        
        node.is_end = True
        node.end_count += 1
        self.total_paths += 1
    
    def search_prefix(self, prefix):
        node = self.root
        for grid_id in prefix:
            if grid_id not in node.children:
                return []
            node = node.children[grid_id]
        return self._collect_paths(node, prefix)
    
    def _collect_paths(self, node, current_path):
        results = []
        stack = [(node, current_path.copy())]
        
        while stack:
            cur_node, cur_path = stack.pop()
            if cur_node.is_end:
                results.append((cur_path.copy(), cur_node.end_count))
            for grid_id, child in cur_node.children.items():
                cur_path.append(grid_id)
                stack.append((child, cur_path.copy()))
                cur_path.pop()
        
        return results
    
    def get_all_paths_with_counts(self):
        all_paths = []
        stack = [(self.root, [])]
        
        while stack:
            node, current_path = stack.pop()
            if node.is_end:
                all_paths.append((current_path.copy(), node.end_count))
            for grid_id, child in node.children.items():
                current_path.append(grid_id)
                stack.append((child, current_path.copy()))
                current_path.pop()
        
        return all_paths
    
    def get_top_k_paths(self, k=10, min_length=3):
        all_paths = self.get_all_paths_with_counts()
        filtered = [(p, c) for p, c in all_paths if len(p) >= min_length]
        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered[:k]
    
    def get_paths_between_regions(self, start_grids, end_grids, k=10, min_length=2):
        all_paths = self.get_all_paths_with_counts()
        filtered = []
        for path, count in all_paths:
            if len(path) >= min_length:
                if path[0] in start_grids and path[-1] in end_grids:
                    filtered.append((path, count))
        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered[:k]
    
    def get_stats(self):
        stats = {
            'total_paths': self.total_paths,
            'total_nodes': self._count_nodes(),
            'max_depth': self._get_max_depth()
        }
        return stats
    
    def _count_nodes(self):
        count = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            count += 1
            for child in node.children.values():
                stack.append(child)
        return count
    
    def _get_max_depth(self):
        if not self.root.children:
            return 0
        max_depth = 0
        stack = [(self.root, 0)]
        while stack:
            node, depth = stack.pop()
            if not node.children:
                max_depth = max(max_depth, depth)
            for child in node.children.values():
                stack.append((child, depth + 1))
        return max_depth
    
    def save(self, filepath):
        data = self._serialize_iterative()
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"💾 路径 Trie 已保存到: {filepath}")
    
    def _serialize_iterative(self):
        nodes = []
        edges = []
        node_id_map = {}
        
        stack = [(self.root, 0)]
        node_id_map[self.root] = 0
        nodes.append({
            'count': self.root.count,
            'is_end': self.root.is_end,
            'end_count': self.root.end_count
        })
        
        while stack:
            node, node_id = stack.pop()
            for grid_id, child in node.children.items():
                if child not in node_id_map:
                    child_id = len(nodes)
                    node_id_map[child] = child_id
                    nodes.append({
                        'count': child.count,
                        'is_end': child.is_end,
                        'end_count': child.end_count
                    })
                    stack.append((child, child_id))
                edges.append({
                    'from': node_id,
                    'to': node_id_map[child],
                    'grid_id': grid_id
                })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'total_paths': self.total_paths
        }
    
    @classmethod
    def load(cls, filepath):
        """从缓存文件加载 Trie（扁平序列化格式，非递归 pickle 对象树）"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        trie = cls()
        nodes = [TrieNode() for _ in data['nodes']]
        for i, node_data in enumerate(data['nodes']):
            nodes[i].count = node_data['count']
            nodes[i].is_end = node_data['is_end']
            nodes[i].end_count = node_data['end_count']

        for edge in data['edges']:
            nodes[edge['from']].children[edge['grid_id']] = nodes[edge['to']]

        trie.root = nodes[0]
        trie.total_paths = data['total_paths']
        print(f"📂 路径 Trie 已从文件加载: {filepath}")
        return trie


def haversine_distance(lon1, lat1, lon2, lat2):
    """计算两点间的球面距离（米）"""
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


def get_path_distance(coords):
    """计算路径的实际距离（米）"""
    if len(coords) < 2:
        return 0
    total = 0
    for i in range(len(coords)-1):
        total += haversine_distance(
            coords[i][0], coords[i][1],
            coords[i+1][0], coords[i+1][1]
        )
    return total


def trajectory_to_path_sequence(trip_points, min_grids=2):
    from grid_utils import lonlat_to_grid_id
    sequence = []
    for point in trip_points:
        grid_id = lonlat_to_grid_id(point['lon'], point['lat'])
        if grid_id != -1:
            if not sequence or sequence[-1] != grid_id:
                sequence.append(grid_id)
    if len(sequence) < min_grids:
        return []
    return sequence


def trip_to_display_coords(trip_points, max_points=500):
    """将行程 GPS 转为地图折线坐标（抽稀以控制点数）"""
    if not trip_points:
        return []
    coords = [[float(p['lon']), float(p['lat'])] for p in trip_points]
    if len(coords) <= max_points:
        return coords
    step = max(1, len(coords) // max_points)
    sampled = coords[::step]
    if sampled[-1] != coords[-1]:
        sampled.append(coords[-1])
    return sampled[:max_points]


def _path_key(path_seq):
    return tuple(int(g) for g in path_seq)


class PathExemplarIndex:
    """
    F7/F8 展示索引：网格路径序列 -> 一条代表性 GPS 折线
    统计仍用 Trie；地图绘制回溯真实轨迹样例。
    """

    def __init__(self, max_points_per_path=800):
        self.max_points_per_path = max_points_per_path
        self.exemplars = {}

    def register(self, path_seq, trip_points):
        key = _path_key(path_seq)
        coords = trip_to_display_coords(trip_points, self.max_points_per_path)
        if len(coords) < 2:
            return
        existing = self.exemplars.get(key)
        if existing is None or len(coords) > len(existing):
            self.exemplars[key] = coords

    def get_display_coords(self, path_seq, fallback_coords):
        coords = self.exemplars.get(_path_key(path_seq))
        if coords and len(coords) >= 2:
            return coords, True
        return fallback_coords, False

    def __len__(self):
        return len(self.exemplars)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(
                {'exemplars': self.exemplars, 'max_points': self.max_points_per_path},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"💾 路径 GPS 样例索引已保存: {filepath} ({len(self.exemplars):,} 条)")

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        store = cls(max_points_per_path=data.get('max_points', 800))
        store.exemplars = data.get('exemplars', {})
        print(f"📂 路径 GPS 样例索引已加载: {filepath} ({len(store.exemplars):,} 条)")
        return store


def rebuild_path_exemplars(
    data_loader,
    vehicle_list,
    split_trips_func,
    exemplar_cache_file,
    sample_rate=1.0,
):
    """仅重建 GPS 样例索引（Trie 已存在、缺 exemplar 缓存时用）"""
    import os
    import time

    print("📍 重建路径 GPS 样例索引（不重建 Trie）...")
    start = time.time()
    store = PathExemplarIndex()
    sampled = vehicle_list[: int(len(vehicle_list) * sample_rate)]
    trip_count = 0

    for vid in sampled:
        df = data_loader.load_vehicle_trajectory(vid)
        if df is None or len(df) < 5:
            continue
        for trip in split_trips_func(df):
            if len(trip) < 3:
                continue
            trip_count += 1
            path_seq = trajectory_to_path_sequence(trip, min_grids=2)
            if len(path_seq) >= 2:
                store.register(path_seq, trip)
        if trip_count and trip_count % 5000 == 0:
            print(f"   已扫描 {trip_count} 个行程，样例 {len(store):,} 条...")

    store.save(exemplar_cache_file)
    print(f"✅ GPS 样例索引完成: {len(store):,} 条, 耗时 {time.time() - start:.1f}s")
    return store


def build_path_trie(
    data_loader,
    vehicle_list,
    split_trips_func,
    cache_file="path_trie_cache.pkl",
    exemplar_cache_file="path_exemplars_cache.pkl",
    sample_rate=0.3,
    force_recompute=False,
):
    import os
    import time

    exemplar_store = PathExemplarIndex()

    if not force_recompute and os.path.exists(cache_file):
        try:
            trie = PathTrie.load(cache_file)
            if os.path.exists(exemplar_cache_file):
                exemplar_store = PathExemplarIndex.load(exemplar_cache_file)
            if not os.path.exists(exemplar_cache_file) or len(exemplar_store) == 0:
                print("⚠️ GPS 样例缓存缺失或为空，开始自动补建...")
                exemplar_store = rebuild_path_exemplars(
                    data_loader,
                    vehicle_list,
                    split_trips_func,
                    exemplar_cache_file,
                    sample_rate=sample_rate,
                )
            return trie, exemplar_store
        except (EOFError, pickle.UnpicklingError, AttributeError, KeyError, OSError) as e:
            print(f"⚠️ 缓存无法加载 ({type(e).__name__}: {e})，重新构建...")
            for f in (cache_file, exemplar_cache_file):
                if os.path.exists(f):
                    os.remove(f)

    print("🚗 构建路径前缀树 + GPS 样例索引...")
    start_time = time.time()

    trie = PathTrie()
    sampled_vehicles = vehicle_list[: int(len(vehicle_list) * sample_rate)]
    print(f"   采样车辆: {len(sampled_vehicles)} (总车辆 {len(vehicle_list)})")

    trip_count = 0
    path_count = 0

    for vid in sampled_vehicles:
        df = data_loader.load_vehicle_trajectory(vid)
        if df is None or len(df) < 5:
            continue

        trips = split_trips_func(df)
        for trip in trips:
            if len(trip) < 3:
                continue

            trip_count += 1
            path_seq = trajectory_to_path_sequence(trip, min_grids=2)
            if len(path_seq) >= 2:
                trie.insert(path_seq)
                exemplar_store.register(path_seq, trip)
                path_count += 1

        if trip_count % 5000 == 0:
            print(
                f"   已处理 {trip_count} 个行程，"
                f"路径 {path_count}，GPS 样例 {len(exemplar_store):,}..."
            )

    trie.save(cache_file)
    exemplar_store.save(exemplar_cache_file)

    elapsed = time.time() - start_time
    stats = trie.get_stats()
    print(f"✅ 路径前缀树构建完成")
    print(f"   总行程数: {trip_count}")
    print(f"   总路径数: {path_count}")
    print(f"   GPS 样例索引: {len(exemplar_store):,}")
    print(f"   节点数: {stats['total_nodes']}")
    print(f"   最大深度: {stats['max_depth']}")
    print(f"   耗时: {elapsed:.2f} 秒")

    return trie, exemplar_store