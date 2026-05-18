"""
T-Drive 数据集加载器
"""

import pandas as pd
import os
import glob

class TDriveDataLoader:
    def __init__(self, data_dir, sample_rate=1.0):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.file_list = []
        self._index_files()
    
    def _index_files(self):
        pattern = os.path.join(self.data_dir, "*.txt")
        all_files = glob.glob(pattern)
        
        if self.sample_rate < 1.0:
            import random
            random.seed(42)
            all_files = random.sample(all_files, int(len(all_files) * self.sample_rate))
        
        for file_path in all_files:
            filename = os.path.basename(file_path)
            vehicle_id = int(filename.replace('.txt', ''))
            file_size = os.path.getsize(file_path)
            self.file_list.append({'vehicle_id': vehicle_id, 'file_path': file_path, 'file_size': file_size})
        
        self.file_list.sort(key=lambda x: x['vehicle_id'])
        self.vehicle_map = {item['vehicle_id']: item['file_path'] for item in self.file_list}
        print(f"📁 索引完成: {len(self.file_list)} 个文件 (目录: {os.path.abspath(self.data_dir)})")
        if len(self.file_list) == 0:
            print(f"⚠️ 未匹配到任何 *.txt，请检查路径: {pattern}")
    
    def load_vehicle_trajectory(self, vehicle_id):
        file_path = self.vehicle_map.get(vehicle_id)
        if not file_path:
            return None
        df = pd.read_csv(file_path, header=None,
                         names=['taxi_id', 'timestamp', 'longitude', 'latitude'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')
    
    def load_all_points_generator(self, batch_size=10000):
        for item in self.file_list:
            try:
                df = pd.read_csv(item['file_path'], header=None,
                                 names=['taxi_id', 'timestamp', 'longitude', 'latitude'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                points = []
                for _, row in df.iterrows():
                    points.append({
                        'taxi_id': int(row['taxi_id']),
                        'lon': float(row['longitude']),
                        'lat': float(row['latitude']),
                        'timestamp': row['timestamp']
                    })
                for i in range(0, len(points), batch_size):
                    yield points[i:i+batch_size]
            except Exception as e:
                print(f"⚠️ 读取文件失败: {e}")
                continue
    
    def get_vehicle_list(self):
        return [item['vehicle_id'] for item in self.file_list]
    
    def get_total_estimate(self):
        total = 0
        for item in self.file_list:
            total += max(1, item['file_size'] // 50)
        return total


_loader = None

def get_loader(data_dir=None, sample_rate=0.1):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    global _loader
    if _loader is None:
        print("🔧 初始化数据加载器...")
        _loader = TDriveDataLoader(data_dir, sample_rate=sample_rate)
        print(f"📊 预计总点数: {_loader.get_total_estimate():,}")
    return _loader