"""
出租车轨迹分析系统 - 主程序
F1-F9 完整实现
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import time
import pickle
import glob
import math
from collections import defaultdict
from data_loader import get_loader
from quadtree import QuadTree
from graph import get_time_dependent_graph
from grid_utils import (
    lonlat_to_grid_id, rect_to_grid_ids, grid_id_to_bounds,
    TimePeriod, GRID_SIZE
)
from path_trie import build_path_trie, get_path_distance
from map_view import (
    DEFAULT_VIEW_STATE, build_payload, scatter_layer, path_layer,
    column_layer, polygon_layer, arc_layer, thin_quadtree_points, bbox_from_view,
    view_state_for_coords, trajectory_paths_to_layers,
)

app = Flask(__name__)
CORS(app)

# ========== 配置参数 ==========
# 默认使用项目根目录下的 data/；可用环境变量覆盖: $env:DATA_DIR="其他路径"
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(_BASE_DIR, "data"))
SAMPLE_RATE = 0.1                     # 开发阶段用10%数据
POINT_SAMPLE_RATE = 5
QUADTREE_CAPACITY = 10
QUADTREE_FILE = "quadtree_cache.pkl"
TIME_GRAPH_FILE = "time_dependent_graph_learned.pkl"
MAX_SEARCH_RESULTS = 20000000

DENSITY_GRID_SIZE = 100

# 路网构建参数
GRAPH_SAMPLE_VEHICLES = 5000000
LEARN_FROM_TRAJECTORIES = True

OD_MATRIX_FILE = "od_matrix_by_period.pkl"
OD_SAMPLE_RATE = 1.0

# 路径分析配置
PATH_TRIE_FILE = "path_trie_cache.pkl"
PATH_SAMPLE_RATE = 1.0

# F1 轨迹可视化
F1_MAX_VEHICLES_ALL = 30
F1_MAX_POINTS_PER_VEHICLE = 3000

# F5/F6 OD 弧线（每条网格路径一条弧，上限避免过密）
OD_MAX_ARCS = 300

BOUNDS = (116.0, 39.4, 117.0, 40.1)

PATH_FREQ_COLORS = [
    [255, 100, 50, 200], [100, 150, 255, 200], [50, 200, 100, 200],
    [255, 200, 50, 200], [200, 50, 150, 200], [50, 150, 200, 200],
]


# ========== 初始化 ==========
print("=" * 60)
print("🚀 初始化出租车轨迹分析系统")
print("=" * 60)

data_loader = get_loader(data_dir=DATA_DIR, sample_rate=SAMPLE_RATE)
vehicle_list = data_loader.get_vehicle_list()
print(f"📊 车辆数: {len(vehicle_list)}")
if len(vehicle_list) == 0:
    print("⚠️ 未找到轨迹数据！请确认：")
    print(f"   1. 目录存在且内含 *.txt 文件: {os.path.abspath(DATA_DIR)}")
    print("   2. 或设置环境变量 DATA_DIR 指向 T-Drive 数据目录")
    print("      PowerShell: $env:DATA_DIR=\"你的数据路径\"")

# 四叉树
if os.path.exists(QUADTREE_FILE):
    print("📂 加载缓存四叉树...")
    start = time.time()
    quad_tree = QuadTree.load(QUADTREE_FILE)
    print(f"✅ 加载完成，耗时 {time.time()-start:.2f} 秒")
else:
    print("🌳 构建四叉树...")
    start = time.time()
    quad_tree = QuadTree.build_from_generator(data_loader, BOUNDS, capacity=QUADTREE_CAPACITY)
    quad_tree.save(QUADTREE_FILE)
    print(f"✅ 构建完成，耗时 {time.time()-start:.2f} 秒")

# 时间依赖图
print("\n🏗️ 初始化时间依赖路网...")
print(f"   轨迹学习: {'启用' if LEARN_FROM_TRAJECTORIES else '禁用'}")
print(f"   采样车辆: {GRAPH_SAMPLE_VEHICLES}")
start = time.time()
graph = get_time_dependent_graph(
    data_loader, BOUNDS,
    rebuild=False,
    cache_file=TIME_GRAPH_FILE,
    sample_vehicles=GRAPH_SAMPLE_VEHICLES,
    learn_from_trajectories=LEARN_FROM_TRAJECTORIES
)
print(f"⏱️ 路网构建/加载耗时: {time.time()-start:.2f} 秒")


# ========== F2 视野地图 / JSON 载荷 ==========
def _clip_bbox(bbox):
    x_min, y_min, x_max, y_max = bbox
    bx0, by0, bx1, by1 = BOUNDS
    return [
        max(x_min, bx0), max(y_min, by0),
        min(x_max, bx1), min(y_max, by1),
    ]


def _pad_bbox(bbox, pad_ratio=0.06):
    """略扩大查询范围，减少视野边缘漏点"""
    x_min, y_min, x_max, y_max = bbox
    pw = (x_max - x_min) * pad_ratio
    ph = (y_max - y_min) * pad_ratio
    return [x_min - pw, y_min - ph, x_max + pw, y_max + ph]


def build_viewport_map_payload(
    longitude=None,
    latitude=None,
    zoom=11,
    width=1200,
    height=800,
    bbox=None,
):
    """F2: 按当前视野与 zoom 从四叉树取点并 LOD 采样"""
    if bbox is not None and len(bbox) >= 4:
        query_bbox = _clip_bbox(_pad_bbox([float(b) for b in bbox[:4]]))
    else:
        query_bbox = _clip_bbox(_pad_bbox(
            bbox_from_view(longitude, latitude, zoom, width, height)
        ))
    raw = quad_tree.query_range(
        (query_bbox[0], query_bbox[1], query_bbox[2], query_bbox[3])
    )
    points, lod = thin_quadtree_points(raw, zoom, bbox=query_bbox)
    layers = [scatter_layer(points, radius=lod['radius'], pickable=False)]
    if longitude is None or latitude is None:
        longitude = (query_bbox[0] + query_bbox[2]) / 2
        latitude = (query_bbox[1] + query_bbox[3]) / 2
    vs = {
        'longitude': float(longitude),
        'latitude': float(latitude),
        'zoom': float(zoom),
        'pitch': 0,
        'bearing': 0,
    }
    meta = {
        'zoom': float(zoom),
        'point_count': len(points),
        'sample_stride': lod['stride'],
        'bbox': query_bbox,
        'raw_in_view': len(raw),
    }
    return build_payload(vs, layers, '车辆ID: {taxi_id}', meta=meta, lock_viewport=False)


# ========== F1 轨迹可视化 ==========
def df_to_path_coords(df, point_sample=1, max_points=None):
    """将轨迹 DataFrame 转为 [lon, lat] 折线坐标"""
    if df is None or len(df) < 2:
        return None
    step = max(1, int(point_sample))
    sample_df = df.iloc[::step]
    if max_points and len(sample_df) > max_points:
        step = max(1, len(sample_df) // max_points)
        sample_df = sample_df.iloc[::step][:max_points]
    coords = sample_df[['longitude', 'latitude']].astype(float).values.tolist()
    return coords if len(coords) >= 2 else None


def build_trajectory_paths(mode, vehicle_id=None, max_vehicles=F1_MAX_VEHICLES_ALL, point_sample=5):
    """F1: 构建单车或多车轨迹路径数据"""
    paths = []
    if mode == 'single':
        if vehicle_id is None:
            return None, '请填写车辆 ID'
        try:
            vid = int(vehicle_id)
        except (TypeError, ValueError):
            return None, '车辆 ID 必须为整数'
        if vid not in data_loader.vehicle_map:
            return None, f'未找到车辆 {vid}，请从已有 ID 中选择'
        df = data_loader.load_vehicle_trajectory(vid)
        coords = df_to_path_coords(df, point_sample, F1_MAX_POINTS_PER_VEHICLE)
        if not coords:
            return None, f'车辆 {vid} 无有效轨迹点'
        paths.append({'path': coords, 'taxi_id': vid})
        return paths, None

    max_vehicles = max(1, min(int(max_vehicles), len(vehicle_list), F1_MAX_VEHICLES_ALL))
    for vid in vehicle_list[:max_vehicles]:
        df = data_loader.load_vehicle_trajectory(vid)
        coords = df_to_path_coords(df, point_sample, F1_MAX_POINTS_PER_VEHICLE)
        if coords:
            paths.append({'path': coords, 'taxi_id': vid})
    if not paths:
        return None, '没有可显示的轨迹'
    return paths, None


def generate_trajectory_map_payload(trajectory_paths):
    """F1: 轨迹折线 JSON 载荷"""
    all_lons, all_lats = [], []
    for item in trajectory_paths:
        for p in item['path']:
            all_lons.append(p[0])
            all_lats.append(p[1])
    layers, tooltip = trajectory_paths_to_layers(trajectory_paths)
    vs = view_state_for_coords(all_lons, all_lats)
    return build_payload(vs, layers, tooltip, lock_viewport=True)


# ========== 行程分割函数 ==========
def split_trips(df, gap_threshold_seconds=300):
    trips = []
    current_trip = []
    prev_time = None

    for _, row in df.iterrows():
        point = {
            'lon': float(row['longitude']),
            'lat': float(row['latitude']),
            'timestamp': row['timestamp']
        }

        if prev_time is None:
            current_trip.append(point)
        else:
            time_diff = (point['timestamp'] - prev_time).total_seconds()
            if time_diff > gap_threshold_seconds:
                if len(current_trip) >= 2:
                    trips.append(current_trip)
                current_trip = [point]
            else:
                current_trip.append(point)
        prev_time = point['timestamp']

    if len(current_trip) >= 2:
        trips.append(current_trip)

    return trips


# ========== 密度分析 ==========
_density_cache = {}

def get_density_cache_filename(period):
    period_names = {
        None: 'all',
        TimePeriod.MORNING_PEAK: 'morning',
        TimePeriod.EVENING_PEAK: 'evening',
        TimePeriod.OFF_PEAK: 'offpeak'
    }
    return f"density_cache_{DENSITY_GRID_SIZE}x{DENSITY_GRID_SIZE}_{period_names.get(period, 'unknown')}.pkl"

def compute_density_for_period(period, force_recompute=False):
    cache_file = get_density_cache_filename(period)

    if not force_recompute and os.path.exists(cache_file):
        print(f"📂 加载密度缓存: {cache_file}")
        start = time.time()
        with open(cache_file, 'rb') as f:
            grid_data, max_count, total_points = pickle.load(f)
        print(f"✅ 密度缓存加载完成，耗时 {time.time()-start:.2f} 秒")
        print(f"   非空网格: {len(grid_data)}")
        return grid_data, max_count, total_points

    period_name = TimePeriod.get_period_name(period)
    print(f"📊 计算密度网格: {DENSITY_GRID_SIZE}x{DENSITY_GRID_SIZE}, 时段={period_name}")
    start_time = time.time()

    x_min, y_min, x_max, y_max = BOUNDS
    x_step = (x_max - x_min) / DENSITY_GRID_SIZE
    y_step = (y_max - y_min) / DENSITY_GRID_SIZE

    grid_counts = defaultdict(int)
    total_points = 0

    for batch in data_loader.load_all_points_generator(batch_size=10000):
        for point in batch:
            lon = point['lon']
            lat = point['lat']
            ts = point['timestamp']

            point_period = TimePeriod.get_period(ts)
            if point_period != period:
                continue

            if x_min <= lon < x_max and y_min <= lat < y_max:
                grid_x = int((lon - x_min) / x_step)
                grid_y = int((lat - y_min) / y_step)
                grid_x = min(grid_x, DENSITY_GRID_SIZE - 1)
                grid_y = min(grid_y, DENSITY_GRID_SIZE - 1)

                grid_counts[(grid_x, grid_y)] += 1
                total_points += 1

    raw_max_count = max(grid_counts.values()) if grid_counts else 1
    log_max = math.log(raw_max_count + 1)

    grid_data = []
    for (grid_x, grid_y), count in grid_counts.items():
        center_lon = x_min + (grid_x + 0.5) * x_step
        center_lat = y_min + (grid_y + 0.5) * y_step
        log_count = math.log(count + 1)
        intensity = log_count / log_max if log_max > 0 else 0

        grid_data.append({
            'lon': center_lon,
            'lat': center_lat,
            'count': count,
            'intensity': intensity
        })

    with open(cache_file, 'wb') as f:
        pickle.dump((grid_data, raw_max_count, total_points), f)

    elapsed = time.time() - start_time
    print(f"✅ 密度计算完成，耗时 {elapsed:.2f} 秒")
    print(f"   总点数: {total_points:,}")
    print(f"   非空网格: {len(grid_data)}")
    print(f"   原始最大密度: {raw_max_count}")

    return grid_data, raw_max_count, total_points

def get_density_data_by_period(period, force_recompute=False):
    if period not in [0, 1, 2]:
        raise ValueError(f"无效的时段: {period}")
    cache_key = period
    if cache_key not in _density_cache or force_recompute:
        _density_cache[cache_key] = compute_density_for_period(period, force_recompute)
    return _density_cache[cache_key]

def get_all_day_density():
    print("📊 计算全天密度（从三个时段累加）...")
    start_time = time.time()

    morning_data, _, morning_total = get_density_data_by_period(TimePeriod.MORNING_PEAK)
    evening_data, _, evening_total = get_density_data_by_period(TimePeriod.EVENING_PEAK)
    offpeak_data, _, offpeak_total = get_density_data_by_period(TimePeriod.OFF_PEAK)

    grid_counts = {}
    for data in [morning_data, evening_data, offpeak_data]:
        for item in data:
            key = (item['lon'], item['lat'])
            grid_counts[key] = grid_counts.get(key, 0) + item['count']

    raw_max_count = max(grid_counts.values()) if grid_counts else 1
    log_max = math.log(raw_max_count + 1)

    grid_data = []
    for (lon, lat), count in grid_counts.items():
        log_count = math.log(count + 1)
        intensity = log_count / log_max if log_max > 0 else 0
        grid_data.append({
            'lon': lon,
            'lat': lat,
            'count': count,
            'intensity': intensity
        })

    total_points = morning_total + evening_total + offpeak_total
    elapsed = time.time() - start_time
    print(f"✅ 全天密度计算完成，耗时 {elapsed:.2f} 秒")
    print(f"   总点数: {total_points:,}")
    print(f"   非空网格: {len(grid_data)}")

    return grid_data, raw_max_count, total_points


def get_color_from_intensity(intensity):
    if intensity < 0.15:
        r = 0
        g = int(50 + 50 * (intensity / 0.15))
        b = 150 + 105 * (intensity / 0.15)
    elif intensity < 0.35:
        r = 0
        g = int(100 + 155 * ((intensity - 0.15) / 0.20))
        b = int(255 - 100 * ((intensity - 0.15) / 0.20))
    elif intensity < 0.55:
        r = int(0 + 200 * ((intensity - 0.35) / 0.20))
        g = 255
        b = int(255 - 200 * ((intensity - 0.35) / 0.20))
    elif intensity < 0.75:
        r = 200 + 55 * ((intensity - 0.55) / 0.20)
        g = 255 - 55 * ((intensity - 0.55) / 0.20)
        b = 0
    else:
        r = 255
        g = int(200 - 150 * ((intensity - 0.75) / 0.25))
        b = 0
    return [r, g, b, 200]


# ========== 分时段 OD 矩阵构建 ==========
_od_matrix_by_period = None

def build_od_matrix_by_period(force_recompute=False):
    """构建分时段的 OD 矩阵"""
    global _od_matrix_by_period

    if not force_recompute and os.path.exists(OD_MATRIX_FILE):
        print(f"📂 加载分时段 OD 矩阵缓存: {OD_MATRIX_FILE}")
        start = time.time()
        with open(OD_MATRIX_FILE, 'rb') as f:
            _od_matrix_by_period = pickle.load(f)
        print(f"✅ OD 矩阵加载完成，耗时 {time.time()-start:.2f} 秒")
        print(f"   非零 OD 对: {len(_od_matrix_by_period)}")
        return _od_matrix_by_period

    print("🚗 构建分时段 OD 矩阵...")
    start_time = time.time()

    od_counts = defaultdict(int)
    sampled_vehicles = vehicle_list[:int(len(vehicle_list) * OD_SAMPLE_RATE)]
    print(f"   采样车辆: {len(sampled_vehicles)}")

    trip_count = 0

    for vid in sampled_vehicles:
        df = data_loader.load_vehicle_trajectory(vid)
        if df is None or len(df) < 2:
            continue

        trips = split_trips(df)
        for trip in trips:
            if len(trip) < 2:
                continue

            start = trip[0]
            end = trip[-1]
            start_grid = lonlat_to_grid_id(start['lon'], start['lat'])
            end_grid = lonlat_to_grid_id(end['lon'], end['lat'])

            if start_grid != -1 and end_grid != -1 and start_grid != end_grid:
                period = TimePeriod.get_period(start['timestamp'])
                key = (start_grid, end_grid, period)
                od_counts[key] += 1
                trip_count += 1

        if trip_count % 5000 == 0:
            print(f"   已处理 {trip_count} 个行程...")

    _od_matrix_by_period = dict(od_counts)

    with open(OD_MATRIX_FILE, 'wb') as f:
        pickle.dump(_od_matrix_by_period, f)

    elapsed = time.time() - start_time
    print(f"✅ 分时段 OD 矩阵构建完成")
    print(f"   总行程数: {trip_count}")
    print(f"   非零 OD 对: {len(_od_matrix_by_period)}")
    print(f"   耗时: {elapsed:.2f} 秒")

    return _od_matrix_by_period

def get_od_matrix_by_period():
    global _od_matrix_by_period
    if _od_matrix_by_period is None:
        _od_matrix_by_period = build_od_matrix_by_period()
    return _od_matrix_by_period


def get_flow_between_regions_by_period(rect1, rect2, period=None):
    od = get_od_matrix_by_period()
    grid_ids_1 = set(rect_to_grid_ids(rect1))
    grid_ids_2 = set(rect_to_grid_ids(rect2))

    if period is not None:
        flow_1_to_2 = 0
        flow_2_to_1 = 0
        for (start, end, per), count in od.items():
            if per != period:
                continue
            if start in grid_ids_1 and end in grid_ids_2:
                flow_1_to_2 += count
            elif start in grid_ids_2 and end in grid_ids_1:
                flow_2_to_1 += count
        return {
            'from_rect1_to_rect2': flow_1_to_2,
            'from_rect2_to_rect1': flow_2_to_1,
            'total': flow_1_to_2 + flow_2_to_1
        }

    result = {}
    for p in TimePeriod.get_all_periods():
        flow_1_to_2 = 0
        flow_2_to_1 = 0
        for (start, end, per), count in od.items():
            if per != p:
                continue
            if start in grid_ids_1 and end in grid_ids_2:
                flow_1_to_2 += count
            elif start in grid_ids_2 and end in grid_ids_1:
                flow_2_to_1 += count
        result[TimePeriod.get_period_name(p)] = {
            'from_rect1_to_rect2': flow_1_to_2,
            'from_rect2_to_rect1': flow_2_to_1,
            'total': flow_1_to_2 + flow_2_to_1
        }

    return result


def get_od_paths_between_regions(rect1, rect2, period=None):
    """
    F5: 列出两区域之间每一条非零 OD 网格路径（用于逐条画弧线）
    period=None 表示全天（三个时段都包含）
    """
    od = get_od_matrix_by_period()
    grid_ids_1 = set(rect_to_grid_ids(rect1))
    grid_ids_2 = set(rect_to_grid_ids(rect2))
    periods_filter = set(TimePeriod.get_all_periods() if period is None else [period])

    paths = []
    for (start, end, per), count in od.items():
        if per not in periods_filter or count <= 0:
            continue
        if start in grid_ids_1 and end in grid_ids_2:
            direction = 'A→B'
        elif start in grid_ids_2 and end in grid_ids_1:
            direction = 'B→A'
        else:
            continue

        start_bounds = grid_id_to_bounds(start)
        end_bounds = grid_id_to_bounds(end)
        if not start_bounds or not end_bounds:
            continue

        paths.append({
            'source': [
                (start_bounds[0] + start_bounds[2]) / 2,
                (start_bounds[1] + start_bounds[3]) / 2,
            ],
            'target': [
                (end_bounds[0] + end_bounds[2]) / 2,
                (end_bounds[1] + end_bounds[3]) / 2,
            ],
            'count': int(count),
            'period': TimePeriod.get_period_name(per),
            'direction': direction,
            'start_grid': start,
            'end_grid': end,
            'label': (
                f"{TimePeriod.get_period_name(per)} {direction} "
                f"网格{start}→{end}"
            ),
        })

    paths.sort(key=lambda x: x['count'], reverse=True)
    return paths


def _period_from_param(period_param):
    period_map = {
        'all': None,
        'morning': TimePeriod.MORNING_PEAK,
        'evening': TimePeriod.EVENING_PEAK,
        'offpeak': TimePeriod.OFF_PEAK,
    }
    return period_map.get(period_param, None)


def _grid_od_path_entry(start, end, per, count, direction):
    start_bounds = grid_id_to_bounds(start)
    end_bounds = grid_id_to_bounds(end)
    if not start_bounds or not end_bounds:
        return None
    pname = TimePeriod.get_period_name(per)
    return {
        'source': [
            (start_bounds[0] + start_bounds[2]) / 2,
            (start_bounds[1] + start_bounds[3]) / 2,
        ],
        'target': [
            (end_bounds[0] + end_bounds[2]) / 2,
            (end_bounds[1] + end_bounds[3]) / 2,
        ],
        'count': int(count),
        'period': pname,
        'direction': direction,
        'start_grid': start,
        'end_grid': end,
        'label': f"{pname} {direction} 网格{start}→{end}",
    }


def get_od_paths_for_region(rect, period=None):
    """
    F6: 目标区域与区外每一条非零 OD 路径（出发=从区内到外区，到达=从外区到区内）
    """
    od = get_od_matrix_by_period()
    grid_ids = set(rect_to_grid_ids(rect))
    periods_filter = set(TimePeriod.get_all_periods() if period is None else [period])

    paths = []
    for (start, end, per), count in od.items():
        if per not in periods_filter or count <= 0:
            continue
        start_in = start in grid_ids
        end_in = end in grid_ids
        if start_in == end_in:
            continue
        if start_in and not end_in:
            entry = _grid_od_path_entry(start, end, per, count, '出发')
        elif end_in and not start_in:
            entry = _grid_od_path_entry(start, end, per, count, '到达')
        else:
            continue
        if entry:
            paths.append(entry)

    paths.sort(key=lambda x: x['count'], reverse=True)
    return paths


def get_region_flows_by_period(rect, period=None):
    od = get_od_matrix_by_period()
    grid_ids = set(rect_to_grid_ids(rect))

    if period is not None:
        departures = defaultdict(int)
        arrivals = defaultdict(int)
        total_departures = 0
        total_arrivals = 0

        for (start, end, per), count in od.items():
            if per != period:
                continue
            if start in grid_ids:
                departures[end] += count
                total_departures += count
            if end in grid_ids:
                arrivals[start] += count
                total_arrivals += count

        top_destinations = []
        top_sources = []

        for gid, count in sorted(departures.items(), key=lambda x: x[1], reverse=True)[:10]:
            top_destinations.append({'grid_id': gid, 'flow': count})

        for gid, count in sorted(arrivals.items(), key=lambda x: x[1], reverse=True)[:10]:
            top_sources.append({'grid_id': gid, 'flow': count})

        return {
            'total_departures': total_departures,
            'total_arrivals': total_arrivals,
            'top_destinations': top_destinations,
            'top_sources': top_sources
        }

    result = {}
    for p in TimePeriod.get_all_periods():
        departures = defaultdict(int)
        arrivals = defaultdict(int)
        total_departures = 0
        total_arrivals = 0

        for (start, end, per), count in od.items():
            if per != p:
                continue
            if start in grid_ids:
                departures[end] += count
                total_departures += count
            if end in grid_ids:
                arrivals[start] += count
                total_arrivals += count

        top_destinations = []
        top_sources = []

        for gid, count in sorted(departures.items(), key=lambda x: x[1], reverse=True)[:10]:
            top_destinations.append({'grid_id': gid, 'flow': count})

        for gid, count in sorted(arrivals.items(), key=lambda x: x[1], reverse=True)[:10]:
            top_sources.append({'grid_id': gid, 'flow': count})

        result[TimePeriod.get_period_name(p)] = {
            'total_departures': total_departures,
            'total_arrivals': total_arrivals,
            'top_destinations': top_destinations,
            'top_sources': top_sources
        }

    return result


# ========== 路径前缀树初始化 ==========
print("\n🚗 构建路径前缀树...")
path_trie = build_path_trie(
    data_loader,
    vehicle_list,
    split_trips,
    cache_file=PATH_TRIE_FILE,
    sample_rate=PATH_SAMPLE_RATE,
    force_recompute=False
)


def _filter_paths_by_distance(path_counts, min_distance, k):
    """从 Trie 候选路径中按距离过滤并取 Top-K"""
    result_paths = []
    for path, count in path_counts:
        coords = []
        for grid_id in path:
            bounds = grid_id_to_bounds(grid_id)
            if bounds:
                coords.append([(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2])
        distance = get_path_distance(coords) if len(coords) >= 2 else 0
        if distance >= min_distance:
            result_paths.append({
                'grid_ids': path,
                'count': count,
                'coords': coords,
                'distance': round(distance),
            })
    result_paths.sort(key=lambda x: x['count'], reverse=True)
    return result_paths[:k]


# ========== 地图 JSON 载荷 ==========
def _paths_data_to_layer(paths_data):
    records = []
    all_lons, all_lats = [], []
    for i, (coords, count, color) in enumerate(paths_data):
        if len(coords) < 2:
            continue
        all_lons.extend(c[0] for c in coords)
        all_lats.extend(c[1] for c in coords)
        records.append({
            'path': coords,
            'color': color or PATH_FREQ_COLORS[i % len(PATH_FREQ_COLORS)],
            'width': 2 + min(8, count / 50),
        })
    return records, all_lons, all_lats


def generate_paths_map_payload(paths_data):
    records, all_lons, all_lats = _paths_data_to_layer(paths_data)
    layers = [path_layer(records, 'freq-paths')] if records else []
    vs = view_state_for_coords(all_lons, all_lats) if all_lons else dict(DEFAULT_VIEW_STATE)
    return build_payload(vs, layers, '{type}', lock_viewport=True)


def generate_paths_with_regions_map_payload(paths_data, rect1_bounds, rect2_bounds):
    x_min1, y_min1, x_max1, y_max1 = rect1_bounds
    x_min2, y_min2, x_max2, y_max2 = rect2_bounds
    polygons = [
        {
            'polygon': [[x_min1, y_min1], [x_max1, y_min1], [x_max1, y_max1], [x_min1, y_max1]],
            'fill_color': [0, 200, 0, 15],
            'line_color': [0, 255, 0, 255],
        },
        {
            'polygon': [[x_min2, y_min2], [x_max2, y_min2], [x_max2, y_max2], [x_min2, y_max2]],
            'fill_color': [255, 100, 0, 15],
            'line_color': [255, 100, 0, 255],
        },
    ]
    records, all_lons, all_lats = _paths_data_to_layer(paths_data)
    layers = [polygon_layer(polygons)]
    if records:
        layers.append(path_layer(records, 'freq-paths'))
    vs = view_state_for_coords(
        [x_min1, x_max2], [y_min1, y_max2]
    )
    return build_payload(vs, layers, '车辆ID: {taxi_id}', lock_viewport=True)


def _rect_center(rect_bounds):
    x_min, y_min, x_max, y_max = rect_bounds
    return (x_min + x_max) / 2, (y_min + y_max) / 2


def _od_paths_to_arcs(od_paths, mode='f5'):
    """每条网格 OD 路径对应一条拱形弧线。mode: f5 | f6"""
    total = len(od_paths)
    if total > OD_MAX_ARCS:
        od_paths = od_paths[:OD_MAX_ARCS]

    period_base_height = {'早高峰': 0.22, '晚高峰': 0.32, '平峰': 0.42}
    if mode == 'f6':
        color_primary = {
            '早高峰': [80, 255, 140, 190],
            '晚高峰': [255, 210, 90, 190],
            '平峰': [120, 190, 255, 190],
        }
        color_secondary = {
            '早高峰': [100, 180, 255, 170],
            '晚高峰': [255, 150, 120, 170],
            '平峰': [180, 140, 255, 170],
        }
        dir_primary, dir_secondary = '出发', '到达'
    else:
        color_primary = {
            '早高峰': [80, 255, 140, 200],
            '晚高峰': [255, 210, 90, 200],
            '平峰': [120, 190, 255, 200],
        }
        color_secondary = {
            '早高峰': [50, 210, 100, 170],
            '晚高峰': [255, 150, 50, 170],
            '平峰': [90, 160, 230, 170],
        }
        dir_primary, dir_secondary = 'A→B', 'B→A'

    arcs = []
    for p in od_paths:
        pname = p['period']
        jitter = ((p['start_grid'] * 13 + p['end_grid'] * 7) % 15) * 0.015
        height = period_base_height.get(pname, 0.3) + jitter
        if p['direction'] == dir_primary:
            color = color_primary.get(pname, [80, 255, 120, 200])
        else:
            color = color_secondary.get(pname, [255, 160, 60, 170])

        arcs.append({
            'source': p['source'],
            'target': p['target'],
            'count': p['count'],
            'label': p['label'],
            'color': color,
            'width': min(8, 1 + math.log10(p['count'] + 1) * 1.5),
            'height': height,
        })

    return arcs, total


def generate_f5_map_payload(rect1_bounds, rect2_bounds, period_param='all'):
    """F5: 区域高亮 + 每条 OD 网格路径一条拱形连线"""
    x_min1, y_min1, x_max1, y_max1 = rect1_bounds
    x_min2, y_min2, x_max2, y_max2 = rect2_bounds
    polygons = [
        {
            'polygon': [[x_min1, y_min1], [x_max1, y_min1], [x_max1, y_max1], [x_min1, y_max1]],
            'fill_color': [0, 200, 0, 15],
            'line_color': [0, 255, 0, 255],
        },
        {
            'polygon': [[x_min2, y_min2], [x_max2, y_min2], [x_max2, y_max2], [x_min2, y_max2]],
            'fill_color': [255, 100, 0, 15],
            'line_color': [255, 100, 0, 255],
        },
    ]

    period = _period_from_param(period_param)
    od_paths = get_od_paths_between_regions(rect1_bounds, rect2_bounds, period)
    arcs, total_paths = _od_paths_to_arcs(od_paths, mode='f5')

    layers = [polygon_layer(polygons)]
    if arcs:
        layers.append(arc_layer(arcs))
    vs = view_state_for_coords([x_min1, x_max2], [y_min1, y_max2])
    meta = {
        'arc_count': len(arcs),
        'total_paths': total_paths,
        'capped': total_paths > OD_MAX_ARCS,
    }
    return build_payload(
        vs, layers, '{label}: {count} 次', meta=meta, lock_viewport=True
    )


def generate_f6_map_payload(rect_bounds, period_param='all'):
    """F6: 单区域高亮 + 与区外每条 OD 路径一条拱形连线"""
    x_min, y_min, x_max, y_max = rect_bounds
    polygons = [{
        'polygon': [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
        'fill_color': [0, 200, 0, 20],
        'line_color': [0, 255, 0, 255],
    }]

    period = _period_from_param(period_param)
    od_paths = get_od_paths_for_region(rect_bounds, period)
    arcs, total_paths = _od_paths_to_arcs(od_paths, mode='f6')

    layers = [polygon_layer(polygons)]
    if arcs:
        layers.append(arc_layer(arcs))
    vs = view_state_for_coords([x_min, x_max], [y_min, y_max])
    vs['zoom'] = max(vs.get('zoom', 11), 11)
    meta = {
        'arc_count': len(arcs),
        'total_paths': total_paths,
        'capped': total_paths > OD_MAX_ARCS,
    }
    return build_payload(
        vs, layers, '{label}: {count} 次', meta=meta, lock_viewport=True
    )


def generate_region_highlight_payload(rect1_bounds, rect2_bounds=None, highlight_type='f5'):
    x_min1, y_min1, x_max1, y_max1 = rect1_bounds
    polygons = [{
        'polygon': [[x_min1, y_min1], [x_max1, y_min1], [x_max1, y_max1], [x_min1, y_max1]],
        'fill_color': [0, 200, 0, 15],
        'line_color': [0, 255, 0, 255],
    }]
    if highlight_type == 'f5' and rect2_bounds:
        x_min2, y_min2, x_max2, y_max2 = rect2_bounds
        polygons.append({
            'polygon': [[x_min2, y_min2], [x_max2, y_min2], [x_max2, y_max2], [x_min2, y_max2]],
            'fill_color': [255, 100, 0, 15],
            'line_color': [255, 100, 0, 255],
        })
        vs = view_state_for_coords([x_min1, x_max2], [y_min1, y_max2])
    else:
        vs = view_state_for_coords([x_min1, x_max1], [y_min1, y_max1])
        vs['zoom'] = 12
    layers = [polygon_layer(polygons)]
    return build_payload(vs, layers, '', lock_viewport=True)


def generate_map_payload(points_data=None, show_heatmap=False, path_coords=None,
                       start_lon=None, start_lat=None, end_lon=None, end_lat=None,
                       heatmap_period=None):
    layers = []
    tooltip_text = ''
    vs = dict(DEFAULT_VIEW_STATE)

    if show_heatmap:
        if heatmap_period is None:
            grid_data, max_count, _ = get_all_day_density()
            period_name = "全天"
        else:
            grid_data, max_count, _ = get_density_data_by_period(heatmap_period)
            period_name = TimePeriod.get_period_name(heatmap_period)
        for item in grid_data:
            item['color'] = get_color_from_intensity(item['intensity'])
        layers.append(column_layer(grid_data, max_count, period_name))
        tooltip_text = f'密度: {{count}} 辆 | 时段: {period_name}'
        vs['pitch'] = 45

    elif path_coords and len(path_coords) > 0:
        layers.append(path_layer([{'path': path_coords, 'color': [255, 100, 50, 255], 'width': 5}]))
        markers = []
        if start_lon is not None and start_lat is not None:
            markers.append({'lon': start_lon, 'lat': start_lat, 'type': '起点', 'color': [50, 200, 50, 255]})
        if end_lon is not None and end_lat is not None:
            markers.append({'lon': end_lon, 'lat': end_lat, 'type': '终点', 'color': [200, 50, 50, 255]})
        if markers:
            layers.append(scatter_layer(markers, layer_id='path-markers', radius=80))
        all_lons = [p[0] for p in path_coords]
        all_lats = [p[1] for p in path_coords]
        vs = view_state_for_coords(all_lons, all_lats)
        tooltip_text = '{type}'

    if points_data and len(points_data) > 0:
        layers.append(scatter_layer(
            points_data, layer_id='search-highlight',
            radius=50, color=[230, 80, 50, 200], opacity=0.9
        ))
        tooltip_text = '车辆ID: {taxi_id}'

    return build_payload(vs, layers, tooltip_text, lock_viewport=True)


# ========== 初始化地图 ==========
print("🗺️ 生成初始地图（F2 视野 LOD）...")
initial_map = build_viewport_map_payload(
    DEFAULT_VIEW_STATE['longitude'],
    DEFAULT_VIEW_STATE['latitude'],
    DEFAULT_VIEW_STATE['zoom'],
)


# ========== Flask 路由 ==========
@app.route('/')
def index():
    stats = quad_tree.get_stats()
    graph_stats = graph.get_stats()
    path_stats = path_trie.get_stats()
    return render_template('index.html',
                          initial_map=initial_map,
                          sample_vehicle_id=vehicle_list[0] if vehicle_list else '',
                          vehicle_count=len(vehicle_list),
                          total_points=f"{stats['total_points']:,}",
                          graph_nodes=graph_stats['nodes'],
                          graph_edges=graph_stats['edges'],
                          graph_samples=graph_stats['samples'],
                          path_total=path_stats['total_paths'],
                          path_nodes=path_stats['total_nodes'])


@app.route('/api/trajectory', methods=['POST'])
def show_trajectory():
    """F1: 显示全部或指定出租车的 GPS 轨迹（折线）"""
    start = time.time()
    data = request.json or {}
    mode = data.get('mode', 'single')
    vehicle_id = data.get('vehicle_id')
    max_vehicles = data.get('max_vehicles', F1_MAX_VEHICLES_ALL)
    point_sample = data.get('point_sample', POINT_SAMPLE_RATE)

    paths, err = build_trajectory_paths(mode, vehicle_id, max_vehicles, point_sample)
    if err:
        return jsonify({'error': err}), 400

    total_points = sum(len(p['path']) for p in paths)
    elapsed = time.time() - start
    print(f"🛤️ F1 轨迹: mode={mode}, 车辆数={len(paths)}, 点数={total_points}, 耗时={elapsed:.3f}s")

    return jsonify({
        'success': True,
        'mode': mode,
        'vehicle_count': len(paths),
        'point_count': total_points,
        'query_time': elapsed,
        'vehicle_ids': [p['taxi_id'] for p in paths],
        'map': generate_trajectory_map_payload(paths),
    })


@app.route('/api/trajectory/vehicle_ids', methods=['GET'])
def trajectory_vehicle_ids():
    """返回可选车辆 ID 列表（供 F1 参考）"""
    limit = min(int(request.args.get('limit', 50)), len(vehicle_list))
    return jsonify({'vehicle_ids': vehicle_list[:limit], 'total': len(vehicle_list)})


@app.route('/api/map_viewport', methods=['POST'])
def map_viewport():
    """F2: 按视野与缩放级别返回 LOD 散点（优先使用前端 map.getBounds()）"""
    data = request.json or {}
    zoom = data.get('zoom', DEFAULT_VIEW_STATE['zoom'])
    client_bbox = data.get('bbox')
    if client_bbox and len(client_bbox) >= 4:
        payload = build_viewport_map_payload(zoom=zoom, bbox=client_bbox)
    else:
        lon = data.get('longitude', DEFAULT_VIEW_STATE['longitude'])
        lat = data.get('latitude', DEFAULT_VIEW_STATE['latitude'])
        width = max(int(data.get('width', 1200)), 400)
        height = max(int(data.get('height', 800)), 300)
        payload = build_viewport_map_payload(lon, lat, zoom, width, height)
    return jsonify({'map': payload})


@app.route('/api/search', methods=['POST'])
def region_search():
    start = time.time()
    rect = request.json.get('rect', {})
    points = quad_tree.query_range((rect.get('x_min'), rect.get('y_min'),
                                     rect.get('x_max'), rect.get('y_max')))
    total = len(points)
    if total > MAX_SEARCH_RESULTS:
        points = points[:MAX_SEARCH_RESULTS]
    result = [{'lon': p[0], 'lat': p[1], 'taxi_id': p[2], 'timestamp': str(p[3])} for p in points]
    print(f"🔍 搜索: 找到={total}, 返回={len(result)}, 耗时={time.time()-start:.3f}s")
    return jsonify({'count': total, 'returned': len(result), 'query_time': time.time()-start,
                    'map': generate_map_payload(points_data=result)})


@app.route('/api/density_view', methods=['GET'])
def density_view():
    return jsonify({'map': generate_map_payload(show_heatmap=True, heatmap_period=None)})


@app.route('/api/density_by_period', methods=['POST'])
def density_by_period():
    data = request.json
    period = data.get('period')
    if period == 'all' or period is None:
        period = None
    else:
        period = int(period)

    if period is None:
        _, _, total_points = get_all_day_density()
        period_name = "全天"
    else:
        _, _, total_points = get_density_data_by_period(period)
        period_name = TimePeriod.get_period_name(period)

    return jsonify({
        'period': period,
        'period_name': period_name,
        'total_points': total_points,
        'map': generate_map_payload(show_heatmap=True, heatmap_period=period)
    })


@app.route('/api/density_stats', methods=['GET'])
def density_stats():
    stats = []
    _, _, total_points = get_all_day_density()
    stats.append({'name': '全天', 'total_points': total_points})
    for period in TimePeriod.get_all_periods():
        _, _, total_points = get_density_data_by_period(period)
        stats.append({'name': TimePeriod.get_period_name(period), 'total_points': total_points})
    return jsonify(stats)


@app.route('/api/normal_view', methods=['GET'])
def normal_view():
    return jsonify({'map': build_viewport_map_payload(
        DEFAULT_VIEW_STATE['longitude'],
        DEFAULT_VIEW_STATE['latitude'],
        DEFAULT_VIEW_STATE['zoom'],
    )})


@app.route('/api/shortest_path', methods=['POST'])
def shortest_path():
    start = time.time()
    data = request.json
    slon, slat = data.get('start_lon'), data.get('start_lat')
    elon, elat = data.get('end_lon'), data.get('end_lat')
    period = data.get('period', TimePeriod.OFF_PEAK)

    if None in [slon, slat, elon, elat]:
        return jsonify({'error': '缺少坐标'}), 400

    start_node, _ = graph.find_nearest_node(slon, slat)
    end_node, _ = graph.find_nearest_node(elon, elat)

    if start_node is None or end_node is None:
        return jsonify({'error': '无法找到附近节点'}), 400

    path, total_time = graph.dijkstra_time(start_node, end_node, period)

    if path is None:
        return jsonify({'error': '未找到路径'}), 400

    coords = graph.get_path_coords(path)
    elapsed = time.time() - start
    time_str = f"{total_time/60:.1f}分钟" if total_time > 60 else f"{total_time:.0f}秒"
    period_name = TimePeriod.get_period_name(period)

    print(f"🛣️ 路径: {len(path)}节点, {time_str}, {period_name}, 耗时{elapsed:.3f}s")

    return jsonify({
        'success': True,
        'time_seconds': round(total_time),
        'time_str': time_str,
        'period': period_name,
        'node_count': len(path),
        'query_time': elapsed,
        'map': generate_map_payload(path_coords=coords,
                                    start_lon=slon, start_lat=slat,
                                    end_lon=elon, end_lat=elat)
    })


@app.route('/api/compare_paths', methods=['POST'])
def compare_paths():
    data = request.json
    slon, slat = data.get('start_lon'), data.get('start_lat')
    elon, elat = data.get('end_lon'), data.get('end_lat')

    start_node, _ = graph.find_nearest_node(slon, slat)
    end_node, _ = graph.find_nearest_node(elon, elat)

    results = {}
    for period in TimePeriod.get_all_periods():
        if start_node and end_node:
            _, t = graph.dijkstra_time(start_node, end_node, period)
            if t:
                results[TimePeriod.get_period_name(period)] = {
                    'time_seconds': round(t),
                    'time_str': f"{t/60:.1f}分钟" if t > 60 else f"{t:.0f}秒"
                }
    return jsonify(results)


@app.route('/api/od_flow', methods=['POST'])
def od_flow():
    data = request.json
    rect1 = data.get('rect1')
    rect2 = data.get('rect2')
    period_param = data.get('period', 'all')

    if not rect1 or not rect2:
        return jsonify({'error': '缺少区域参数'}), 400

    rect1_bounds = (rect1['x_min'], rect1['y_min'], rect1['x_max'], rect1['y_max'])
    rect2_bounds = (rect2['x_min'], rect2['y_min'], rect2['x_max'], rect2['y_max'])

    period_map = {
        'all': None,
        'morning': TimePeriod.MORNING_PEAK,
        'evening': TimePeriod.EVENING_PEAK,
        'offpeak': TimePeriod.OFF_PEAK
    }
    period = period_map.get(period_param, None)

    flows = get_flow_between_regions_by_period(rect1_bounds, rect2_bounds, period)
    map_payload = generate_f5_map_payload(rect1_bounds, rect2_bounds, period_param)

    return jsonify({
        'success': True,
        'period': period_param,
        'flows': flows,
        'path_count': map_payload.get('meta', {}).get('total_paths', 0),
        'arcs_drawn': map_payload.get('meta', {}).get('arc_count', 0),
        'arcs_capped': map_payload.get('meta', {}).get('capped', False),
        'map': map_payload,
    })


@app.route('/api/od_region_flow', methods=['POST'])
def od_region_flow():
    data = request.json
    rect = data.get('rect')
    period_param = data.get('period', 'all')

    if not rect:
        return jsonify({'error': '缺少区域参数'}), 400

    rect_bounds = (rect['x_min'], rect['y_min'], rect['x_max'], rect['y_max'])
    period = _period_from_param(period_param)

    flows = get_region_flows_by_period(rect_bounds, period)
    map_payload = generate_f6_map_payload(rect_bounds, period_param)

    return jsonify({
        'success': True,
        'period': period_param,
        'flows': flows,
        'path_count': map_payload.get('meta', {}).get('total_paths', 0),
        'arcs_drawn': map_payload.get('meta', {}).get('arc_count', 0),
        'arcs_capped': map_payload.get('meta', {}).get('capped', False),
        'map': map_payload,
    })


@app.route('/api/frequent_paths_global', methods=['POST'])
def frequent_paths_global():
    """F7: 全局 Top-k 最频繁路径（只按距离过滤）"""
    data = request.json
    k = data.get('k', 10)
    min_distance = data.get('min_distance', 0)
    
    candidates = path_trie.get_top_k_paths(k=max(k * 8, 80), min_length=2)
    result_paths = _filter_paths_by_distance(candidates, min_distance, k)
    paths_for_map = [(p['coords'], p['count'], None) for p in result_paths]

    return jsonify({
        'success': True,
        'k': k,
        'min_distance': min_distance,
        'total_paths_found': len(result_paths),
        'top_paths': result_paths,
        'map': generate_paths_map_payload(paths_for_map),
    })


@app.route('/api/frequent_paths_between', methods=['POST'])
def frequent_paths_between():
    """F8: 两区域间 Top-k 最频繁路径（只按距离过滤）"""
    data = request.json
    rect1 = data.get('rect1')
    rect2 = data.get('rect2')
    k = data.get('k', 10)
    min_distance = data.get('min_distance', 0)

    if not rect1 or not rect2:
        return jsonify({'error': '缺少区域参数'}), 400

    rect1_bounds = (rect1['x_min'], rect1['y_min'], rect1['x_max'], rect1['y_max'])
    rect2_bounds = (rect2['x_min'], rect2['y_min'], rect2['x_max'], rect2['y_max'])
    
    start_grids = set(rect_to_grid_ids(rect1_bounds))
    end_grids = set(rect_to_grid_ids(rect2_bounds))
    
    candidates = path_trie.get_paths_between_regions(
        start_grids, end_grids, k=max(k * 8, 80), min_length=2
    )
    result_paths = _filter_paths_by_distance(candidates, min_distance, k)
    paths_for_map = [(p['coords'], p['count'], None) for p in result_paths]

    return jsonify({
        'success': True,
        'k': k,
        'min_distance': min_distance,
        'total_paths_found': len(result_paths),
        'top_paths': result_paths,
        'map': generate_paths_with_regions_map_payload(paths_for_map, rect1_bounds, rect2_bounds),
    })


@app.route('/api/path_stats', methods=['GET'])
def path_stats():
    stats = path_trie.get_stats()
    return jsonify({
        'total_paths': stats['total_paths'],
        'total_nodes': stats['total_nodes'],
        'max_depth': stats['max_depth']
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    qs = quad_tree.get_stats()
    gs = graph.get_stats()
    od = get_od_matrix_by_period()
    ps = path_trie.get_stats()
    return jsonify({
        'vehicles': len(vehicle_list),
        'points': qs['total_points'],
        'quadtree_nodes': qs['nodes'],
        'graph_nodes': gs['nodes'],
        'graph_edges': gs['edges'],
        'graph_samples': gs['samples'],
        'od_pairs': len(od),
        'path_total': ps['total_paths'],
        'path_nodes': ps['total_nodes']
    })


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 启动服务器: http://127.0.0.1:5000")
    print("📋 功能: F1-F9 完整实现")
    print("   - F1: 出租车轨迹可视化（全部/指定车辆）")
    print("   - F2: 地图缩放 + 视野 LOD 动态渲染")
    print("   - F3: 四叉树区域搜索")
    print("   - F4: 密度分析（对数变换 + 缓存）")
    print("   - F5/F6: 分时段 OD 矩阵 + 区域高亮地图")
    print("   - F7/F8: 频繁路径挖掘（按距离过滤）")
    print("   - F9: 时间依赖 Dijkstra 最短路径")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000, use_reloader=False)