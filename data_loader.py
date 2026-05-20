"""
T-Drive 数据集加载器
"""

import os
import glob

import pandas as pd
import numpy as np


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
            self.file_list.append({
                'vehicle_id': vehicle_id,
                'file_path': file_path,
                'file_size': file_size,
            })

        self.file_list.sort(key=lambda x: x['vehicle_id'])
        self.vehicle_map = {item['vehicle_id']: item['file_path'] for item in self.file_list}
        print(f"📁 索引完成: {len(self.file_list)} 个文件 (目录: {os.path.abspath(self.data_dir)})")
        if len(self.file_list) == 0:
            print(f"⚠️ 未匹配到任何 *.txt，请检查路径: {pattern}")

    def load_vehicle_trajectory(self, vehicle_id):
        file_path = self.vehicle_map.get(vehicle_id)
        if not file_path:
            return None
        df = pd.read_csv(
            file_path,
            header=None,
            names=['taxi_id', 'timestamp', 'longitude', 'latitude'],
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')

    def iter_points_fast(self, bounds=None, point_stride=1, chunksize=200_000):
        """
        流式读取 GPS 点（分块 csv + 向量化抽样，避免 iterrows）。
        bounds: (x_min, y_min, x_max, y_max) 经度纬度范围，None 表示不过滤。
        """
        stride = max(1, int(point_stride))
        total_read = 0
        if bounds is not None:
            x_min, y_min, x_max, y_max = bounds
        else:
            x_min = y_min = x_max = y_max = None

        for item in self.file_list:
            try:
                reader = pd.read_csv(
                    item['file_path'],
                    header=None,
                    names=['taxi_id', 'timestamp', 'longitude', 'latitude'],
                    chunksize=chunksize,
                )
                for chunk in reader:
                    if bounds is not None:
                        mask = (
                            (chunk['longitude'] >= x_min)
                            & (chunk['longitude'] < x_max)
                            & (chunk['latitude'] >= y_min)
                            & (chunk['latitude'] < y_max)
                        )
                        chunk = chunk.loc[mask]
                    if chunk.empty:
                        continue

                    ts = pd.to_datetime(chunk['timestamp'], errors='coerce')
                    taxis = chunk['taxi_id'].to_numpy(dtype=np.int64, copy=False)
                    lons = chunk['longitude'].to_numpy(dtype=np.float64, copy=False)
                    lats = chunk['latitude'].to_numpy(dtype=np.float64, copy=False)
                    n = len(chunk)
                    offset = total_read % stride
                    for i in range(offset, n, stride):
                        yield {
                            'taxi_id': int(taxis[i]),
                            'lon': float(lons[i]),
                            'lat': float(lats[i]),
                            'timestamp': ts.iloc[i],
                        }
                    total_read += n
            except Exception as e:
                print(f"⚠️ 读取文件失败 {item['file_path']}: {e}")
                continue

    def load_all_points_generator(self, batch_size=10000, bounds=None, point_stride=1):
        """兼容旧接口：按批 yield 点列表"""
        batch = []
        for pt in self.iter_points_fast(bounds=bounds, point_stride=point_stride):
            batch.append(pt)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def get_vehicle_list(self):
        return [item['vehicle_id'] for item in self.file_list]

    def get_total_estimate(self):
        total = 0
        for item in self.file_list:
            total += max(1, item['file_size'] // 50)
        return total


_loader = None


def get_loader(data_dir=None, sample_rate=1.0):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    global _loader
    if _loader is None:
        print("🔧 初始化数据加载器...")
        _loader = TDriveDataLoader(data_dir, sample_rate=sample_rate)
        print(f"📊 预计总点数: {_loader.get_total_estimate():,}")
    return _loader
