"""
四叉树空间索引 - 用于快速区域搜索
"""

import pickle

class QuadTree:
    """四叉树空间索引"""
    
    def __init__(self, x_min, y_min, x_max, y_max, capacity=10, depth=0, max_depth=50):
        self.bounds = (x_min, y_min, x_max, y_max)
        self.capacity = capacity
        self.points = []
        self.children = None
        self.depth = depth
        self.max_depth = max_depth
    
    def contains(self, point):
        x, y = point[0], point[1]
        x_min, y_min, x_max, y_max = self.bounds
        return x_min <= x < x_max and y_min <= y < y_max
    
    def subdivide(self):
        x_min, y_min, x_max, y_max = self.bounds
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        next_depth = self.depth + 1
        
        self.children = [
            QuadTree(x_min, y_mid, x_mid, y_max, self.capacity, next_depth, self.max_depth),
            QuadTree(x_mid, y_mid, x_max, y_max, self.capacity, next_depth, self.max_depth),
            QuadTree(x_min, y_min, x_mid, y_mid, self.capacity, next_depth, self.max_depth),
            QuadTree(x_mid, y_min, x_max, y_mid, self.capacity, next_depth, self.max_depth),
        ]
        
        for p in self.points:
            for child in self.children:
                if child.contains(p):
                    child.insert(p)
                    break
        self.points = []
    
    def insert(self, point):
        if not self.contains(point):
            return False
        
        if self.children is None:
            if len(self.points) < self.capacity or self.depth >= self.max_depth:
                self.points.append(point)
                return True
            else:
                self.subdivide()
                return self.insert(point)
        
        for child in self.children:
            if child.insert(point):
                return True
        return False
    
    def query_range(self, rect):
        result = []
        x_min, y_min, x_max, y_max = rect
        
        if not self.intersects(rect):
            return result
        
        for p in self.points:
            x, y = p[0], p[1]
            if x_min <= x <= x_max and y_min <= y <= y_max:
                result.append(p)
        
        if self.children is not None:
            for child in self.children:
                result.extend(child.query_range(rect))
        
        return result
    
    def intersects(self, rect):
        x_min, y_min, x_max, y_max = self.bounds
        r_xmin, r_ymin, r_xmax, r_ymax = rect
        return not (x_max < r_xmin or x_min > r_xmax or y_max < r_ymin or y_min > r_ymax)
    
    def get_stats(self):
        stats = {'nodes': 1, 'total_points': len(self.points), 'leaf_nodes': 0 if self.children else 1}
        if self.children:
            for child in self.children:
                child_stats = child.get_stats()
                stats['nodes'] += child_stats['nodes']
                stats['total_points'] += child_stats['total_points']
                stats['leaf_nodes'] += child_stats['leaf_nodes']
        return stats
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"💾 四叉树已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            tree = pickle.load(f)
        print(f"📂 四叉树已从文件加载: {filepath}")
        return tree
    
    @classmethod
    def build_from_generator(cls, data_loader, bounds, capacity=10):
        x_min, y_min, x_max, y_max = bounds
        tree = cls(x_min, y_min, x_max, y_max, capacity)
        total = 0
        
        print("🌳 开始构建四叉树...")
        for batch in data_loader.load_all_points_generator(batch_size=5000):
            for point in batch:
                pt = (point['lon'], point['lat'], point['taxi_id'], point['timestamp'])
                tree.insert(pt)
            total += len(batch)
            if total % 50000 == 0:
                print(f"   已插入 {total:,} 个点...")
        
        stats = tree.get_stats()
        print(f"✅ 四叉树构建完成")
        print(f"   总点数: {total:,}")
        print(f"   节点数: {stats['nodes']}")
        print(f"   叶子节点: {stats['leaf_nodes']}")
        return tree