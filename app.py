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
import json
import heapq
import numpy as np
import pandas as pd
from collections import defaultdict
from data_loader import get_loader
from quadtree import QuadTree
from graph import get_time_dependent_graph
from grid_utils import (
    lonlat_to_grid_id, rect_to_grid_ids, grid_id_to_bounds,
    TimePeriod, GRID_SIZE
)
from path_trie import (
    build_path_trie,
    get_path_distance,
    trip_to_display_coords,
    trajectory_to_path_sequence,
)
from map_view import (
    DEFAULT_VIEW_STATE, build_payload, scatter_layer, path_layer,
    column_layer, heatmap_layer, polygon_layer, arc_layer, thin_quadtree_points, bbox_from_view,
    view_state_for_coords, trajectory_paths_to_layers,
)

app = Flask(__name__)
CORS(app)

# ========== 配置参数 ==========
# 默认使用项目根目录下的 data/；可用环境变量覆盖: $env:DATA_DIR="其他路径"
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(_BASE_DIR, "data"))
CACHE_VERSION = 2


def _env_float(name, default):
    v = os.environ.get(name)
    return float(v) if v is not None else default


def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v is not None else default


# 全量数据策略：索引 100% 车辆文件；对内存大户单独限流（见启动日志）
SAMPLE_RATE = _env_float("SAMPLE_RATE", 1.0)
POINT_SAMPLE_RATE = _env_int("POINT_SAMPLE_RATE", 5)
QUADTREE_CAPACITY = 10
QUADTREE_FILE = os.path.join(_BASE_DIR, "quadtree_cache.pkl")
QUADTREE_POINT_STRIDE = max(1, _env_int("QUADTREE_POINT_STRIDE", 15))
TIME_GRAPH_FILE = os.path.join(_BASE_DIR, "time_dependent_graph_learned.pkl")
MAX_SEARCH_RESULTS = 20000000

DENSITY_GRID_SIZE = 100

# 路网：在已加载车辆中最多用多少辆学习 F9（全量 txt 时建议 2000~4000）
GRAPH_SAMPLE_VEHICLES = _env_int("GRAPH_SAMPLE_VEHICLES", 3000)
LEARN_FROM_TRAJECTORIES = True

OD_MATRIX_FILE = os.path.join(_BASE_DIR, "od_matrix_by_period.pkl")
OD_SAMPLE_RATE = _env_float("OD_SAMPLE_RATE", 1.0)

# F7/F8：在已加载车辆中用于路径 Trie 的比例（全量时建议 0.2~0.35）
PATH_TRIE_FILE = os.path.join(_BASE_DIR, "path_trie_cache.pkl")
PATH_EXEMPLAR_FILE = os.path.join(_BASE_DIR, "path_exemplars_cache.pkl")
PATH_TRIE_SAMPLE_RATE = _env_float("PATH_TRIE_SAMPLE_RATE", 0.3)
PATH_EXEMPLAR_MAX_ENTRIES = _env_int("PATH_EXEMPLAR_MAX_ENTRIES", 300_000)
REGION_GPS_SCAN_VEHICLES = _env_int("REGION_GPS_SCAN_VEHICLES", 300)
TRIE_REGION_MAX_DEPTH = 15
PATH_QUERY_CACHE_SIZE = 512
PATH_DISTANCE_CACHE_MAX = 500_000

# F7/F8 性能：距离缓存、区域查询 LRU、区域 GPS 行程缓存
_distance_cache = {}
_region_gps_trip_cache = {}
_opt_stats = defaultdict(int)


class _LRUCache:
    def __init__(self, maxsize=512):
        self.maxsize = maxsize
        self._data = {}

    def get(self, key):
        if key not in self._data:
            return None
        self._data[key] = self._data.pop(key)
        return self._data[key]

    def put(self, key, value):
        if key in self._data:
            self._data.pop(key)
        elif len(self._data) >= self.maxsize:
            self._data.pop(next(iter(self._data)))
        self._data[key] = value

    def __len__(self):
        return len(self._data)


_region_query_lru = _LRUCache(PATH_QUERY_CACHE_SIZE)


def _meta_path(cache_file):
    return cache_file + ".meta.json"


def _read_cache_meta(cache_file):
    path = _meta_path(cache_file)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache_meta(cache_file, meta):
    with open(_meta_path(cache_file), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _meta_matches(cache_file, expected):
    meta = _read_cache_meta(cache_file)
    if meta is None:
        return False
    return meta == expected


# F1 轨迹可视化
F1_MAX_VEHICLES_ALL = 30
F1_MAX_POINTS_PER_VEHICLE = 3000
F1_SNAP_KEYFRAME_STEP = 10
F1_SNAP_MAX_SEGMENTS = 100

# F9 区域间最短路径：边缘网格 + 方向剪枝 + 早停
F9_REGION_NEAREST_K = 8
F9_EARLY_STOP_SECONDS = 30

# F5/F6 OD 弧线（每条网格路径一条弧，上限避免过密）
OD_MAX_ARCS = 300
F6_BALANCE_MIN_TOTAL_PATHS = 40
F6_BALANCE_DIRECTION_IMBALANCE_THRESHOLD = 0.72
F6_BALANCE_EDGE_MARGIN_RATIO = 0.08
F6_GPS_BACKTRACK_REPLACE_RATIO_CAP = 0.35

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
print(f"📊 已索引车辆文件: {len(vehicle_list)} (SAMPLE_RATE={SAMPLE_RATE})")
print("📋 全量运行策略（启动时按需重建缓存，流式读取加速）:")
print(f"   · 四叉树 F2/F3: 每 {QUADTREE_POINT_STRIDE} 个点取 1 个入库")
print(f"   · 密度 F4 / OD F5-F6: 流式扫描，可用全量车辆 (OD_SAMPLE_RATE={OD_SAMPLE_RATE})")
print(f"   · 路径 Trie F7/F8: 其中 {PATH_TRIE_SAMPLE_RATE:.0%} 车辆建 Trie")
print(f"   · GPS 样例上限: {PATH_EXEMPLAR_MAX_ENTRIES:,} 条")
print(f"   · 路网 F9: 最多 {GRAPH_SAMPLE_VEHICLES} 辆车学习")
if len(vehicle_list) == 0:
    print("⚠️ 未找到轨迹数据！请确认：")
    print(f"   1. 目录存在且内含 *.txt 文件: {os.path.abspath(DATA_DIR)}")
    print("   2. 或设置环境变量 DATA_DIR 指向 T-Drive 数据目录")
    print("      PowerShell: $env:DATA_DIR=\"你的数据路径\"")

_quadtree_meta = {
    "version": CACHE_VERSION,
    "sample_rate": SAMPLE_RATE,
    "quadtree_point_stride": QUADTREE_POINT_STRIDE,
    "vehicle_count": len(vehicle_list),
    "bounds": list(BOUNDS),
}

if os.path.exists(QUADTREE_FILE) and _meta_matches(QUADTREE_FILE, _quadtree_meta):
    print("📂 加载缓存四叉树...")
    start = time.time()
    quad_tree = QuadTree.load(QUADTREE_FILE)
    print(f"✅ 加载完成，耗时 {time.time() - start:.2f} 秒")
else:
    if os.path.exists(QUADTREE_FILE):
        print("⚠️ 四叉树缓存与当前配置不一致，将重建")
    print(f"🌳 构建四叉树（点抽样 1/{QUADTREE_POINT_STRIDE}，流式快速读取）...")
    start = time.time()
    quad_tree = QuadTree.build_from_generator(
        data_loader, BOUNDS,
        capacity=QUADTREE_CAPACITY,
        point_stride=QUADTREE_POINT_STRIDE,
    )
    quad_tree.save(QUADTREE_FILE)
    _write_cache_meta(QUADTREE_FILE, _quadtree_meta)
    print(f"✅ 构建完成，耗时 {time.time() - start:.2f} 秒")

print("\n🏗️ 初始化时间依赖路网...")
print(f"   轨迹学习: {'启用' if LEARN_FROM_TRAJECTORIES else '禁用'}")
print(f"   采样车辆: {min(GRAPH_SAMPLE_VEHICLES, len(vehicle_list))}")
start = time.time()
_graph_meta = {
    "version": CACHE_VERSION,
    "sample_rate": SAMPLE_RATE,
    "graph_sample_vehicles": GRAPH_SAMPLE_VEHICLES,
    "vehicle_count": len(vehicle_list),
    "bounds": list(BOUNDS),
}
_graph_rebuild = not (
    os.path.exists(TIME_GRAPH_FILE)
    and _meta_matches(TIME_GRAPH_FILE, _graph_meta)
)
if _graph_rebuild and os.path.exists(TIME_GRAPH_FILE):
    print("⚠️ 路网缓存与当前配置不一致，将重建")
graph = get_time_dependent_graph(
    data_loader, BOUNDS,
    rebuild=_graph_rebuild,
    cache_file=TIME_GRAPH_FILE,
    sample_vehicles=min(GRAPH_SAMPLE_VEHICLES, len(vehicle_list)),
    learn_from_trajectories=LEARN_FROM_TRAJECTORIES,
)
if _graph_rebuild:
    _write_cache_meta(TIME_GRAPH_FILE, _graph_meta)
print(f"⏱️ 路网构建/加载耗时: {time.time() - start:.2f} 秒")


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


def _subsample_node_chain(nodes, max_segments):
    if len(nodes) <= max_segments + 1:
        return nodes
    picks = {0, len(nodes) - 1}
    inner = list(range(1, len(nodes) - 1))
    step = max(1, len(inner) // max(1, max_segments - 1))
    picks.update(inner[::step])
    return [nodes[i] for i in sorted(picks)]


def _snap_trajectory_to_road_network(coords):
    """
    沿已学习的时间依赖路网贴合轨迹（近似马路走向，非高德级路网）。
    在关键 GPS 点上找最近路网节点，再用最短时间路径串联。
    """
    if graph is None or len(coords) < 2:
        return coords, False

    key_idx = list(range(0, len(coords), F1_SNAP_KEYFRAME_STEP))
    if key_idx[-1] != len(coords) - 1:
        key_idx.append(len(coords) - 1)

    nodes = []
    for i in key_idx:
        lon, lat = coords[i]
        nid, _ = graph.find_nearest_node(lon, lat)
        if nid is not None and (not nodes or nodes[-1] != nid):
            nodes.append(nid)

    if len(nodes) < 2:
        return coords, False

    nodes = _subsample_node_chain(nodes, F1_SNAP_MAX_SEGMENTS)
    merged = []
    for i in range(len(nodes) - 1):
        seg, _ = graph.dijkstra_time(nodes[i], nodes[i + 1], TimePeriod.OFF_PEAK)
        if not seg:
            pt = graph.get_node_coords(nodes[i])
            if pt:
                merged.append([float(pt[0]), float(pt[1])])
            continue
        part = _simplify_path_coords(_graph_node_path_coords(seg))
        if i > 0 and merged and part:
            part = part[1:]
        merged.extend(part)

    if len(merged) < 2:
        return coords, False
    return merged, True


def build_trajectory_paths(
    mode, vehicle_id=None, max_vehicles=F1_MAX_VEHICLES_ALL, point_sample=5, snap_to_road=False,
):
    """F1: 构建单车或多车轨迹路径数据"""
    paths = []
    snap_used = False
    if mode == 'single':
        if vehicle_id is None:
            return None, '请填写车辆 ID', False
        try:
            vid = int(vehicle_id)
        except (TypeError, ValueError):
            return None, '车辆 ID 必须为整数', False
        if vid not in data_loader.vehicle_map:
            return None, f'未找到车辆 {vid}，请从已有 ID 中选择', False
        df = data_loader.load_vehicle_trajectory(vid)
        coords = df_to_path_coords(df, point_sample, F1_MAX_POINTS_PER_VEHICLE)
        if not coords:
            return None, f'车辆 {vid} 无有效轨迹点', False
        if snap_to_road:
            coords, snap_used = _snap_trajectory_to_road_network(coords)
        paths.append({'path': coords, 'taxi_id': vid})
        return paths, None, snap_used

    max_vehicles = max(1, min(int(max_vehicles), len(vehicle_list), F1_MAX_VEHICLES_ALL))
    for vid in vehicle_list[:max_vehicles]:
        df = data_loader.load_vehicle_trajectory(vid)
        coords = df_to_path_coords(df, point_sample, F1_MAX_POINTS_PER_VEHICLE)
        if coords:
            paths.append({'path': coords, 'taxi_id': vid})
    if not paths:
        return None, '没有可显示的轨迹', False
    return paths, None, False


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
    """按时间间隔切分行程（向量化，避免 iterrows 卡顿）"""
    if df is None or len(df) < 2:
        return []

    ts = pd.to_datetime(df['timestamp'])
    gaps = ts.diff().dt.total_seconds().fillna(0).to_numpy()
    lons = df['longitude'].to_numpy(dtype=np.float64, copy=False)
    lats = df['latitude'].to_numpy(dtype=np.float64, copy=False)
    timestamps = ts.to_numpy()

    break_idx = np.where(gaps > gap_threshold_seconds)[0]
    starts = np.concatenate(([0], break_idx))
    ends = np.concatenate((break_idx, [len(df)]))

    trips = []
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        trips.append([
            {
                'lon': float(lons[i]),
                'lat': float(lats[i]),
                'timestamp': timestamps[i],
            }
            for i in range(int(s), int(e))
        ])
    return trips


# ========== 路径前缀树初始化 ==========
print("\n🚗 初始化路径前缀树...")
_path_trie_meta = {
    "version": CACHE_VERSION,
    "sample_rate": SAMPLE_RATE,
    "path_trie_sample_rate": PATH_TRIE_SAMPLE_RATE,
    "path_exemplar_max": PATH_EXEMPLAR_MAX_ENTRIES,
    "vehicle_count": len(vehicle_list),
}
_path_trie_force = not (
    os.path.exists(PATH_TRIE_FILE)
    and _meta_matches(PATH_TRIE_FILE, _path_trie_meta)
)
if _path_trie_force and os.path.exists(PATH_TRIE_FILE):
    print("⚠️ 路径 Trie 缓存与当前配置不一致，将重建")
path_trie, path_exemplars = build_path_trie(
    data_loader,
    vehicle_list,
    split_trips,
    cache_file=PATH_TRIE_FILE,
    exemplar_cache_file=PATH_EXEMPLAR_FILE,
    sample_rate=PATH_TRIE_SAMPLE_RATE,
    exemplar_max_entries=PATH_EXEMPLAR_MAX_ENTRIES,
    force_recompute=_path_trie_force,
)
_write_cache_meta(PATH_TRIE_FILE, _path_trie_meta)
print(f"   GPS 样例索引: {len(path_exemplars):,} 条")
if len(path_exemplars) == 0:
    print("   ⚠️ GPS 样例为空：F7/F8 地图将显示网格折线")


# ========== 密度分析 ==========
_density_cache = {}

F4_EXPAND_RATIOS = [0.05, 0.10, 0.15]
F4_DEFAULT_EXPAND_RATIO = 0.10


def _normalize_f4_expand_ratio(expand_ratio):
    """F4: 仅允许 5%/10%/15%，默认 10%"""
    try:
        ratio = float(expand_ratio)
    except (TypeError, ValueError):
        return F4_DEFAULT_EXPAND_RATIO
    for allowed in F4_EXPAND_RATIOS:
        if abs(ratio - allowed) < 1e-9:
            return allowed
    return F4_DEFAULT_EXPAND_RATIO


def _get_f4_expanded_bounds(expand_ratio):
    """在全局 BOUNDS 基础上统一扩展边界（仅 F4 使用）"""
    ratio = _normalize_f4_expand_ratio(expand_ratio)
    x_min, y_min, x_max, y_max = BOUNDS
    dx = (x_max - x_min) * ratio
    dy = (y_max - y_min) * ratio
    return (x_min - dx, y_min - dy, x_max + dx, y_max + dy), ratio


def get_density_cache_filename(period, expand_ratio):
    period_names = {
        None: 'all',
        TimePeriod.MORNING_PEAK: 'morning',
        TimePeriod.EVENING_PEAK: 'evening',
        TimePeriod.OFF_PEAK: 'offpeak'
    }
    ratio = _normalize_f4_expand_ratio(expand_ratio)
    ratio_pct = int(round(ratio * 100))
    return (
        f"density_cache_{DENSITY_GRID_SIZE}x{DENSITY_GRID_SIZE}_"
        f"{period_names.get(period, 'unknown')}_f4exp{ratio_pct}.pkl"
    )


def _density_ratio_files_complete(ratio):
    period_list = TimePeriod.get_all_periods()
    files = [get_density_cache_filename(p, ratio) for p in period_list]
    files.append(get_density_cache_filename(None, ratio))
    return all(os.path.exists(f) for f in files)


def _load_density_bundle_from_disk(ratio):
    """从磁盘加载某一扩展比例的三时段+全天缓存到内存"""
    period_list = TimePeriod.get_all_periods()
    if not _density_ratio_files_complete(ratio):
        return None
    bundle = {}
    for p in period_list:
        with open(get_density_cache_filename(p, ratio), 'rb') as f:
            bundle[p] = pickle.load(f)
    with open(get_density_cache_filename(None, ratio), 'rb') as f:
        bundle[None] = pickle.load(f)
    _density_cache[('bundle', ratio)] = bundle
    return bundle


def _build_density_bundle(expand_ratio=F4_DEFAULT_EXPAND_RATIO, force_recompute=False):
    """F4: 按最大扩展(15%)单次扫描，构建 5/10/15% 三套三时段+全天缓存"""
    ratio = _normalize_f4_expand_ratio(expand_ratio)
    bundle_key = ('bundle', ratio)
    if bundle_key in _density_cache and not force_recompute:
        return _density_cache[bundle_key]

    period_list = TimePeriod.get_all_periods()

    if not force_recompute:
        for r in F4_EXPAND_RATIOS:
            if ('bundle', r) not in _density_cache:
                loaded = _load_density_bundle_from_disk(r)
                if loaded:
                    print(f"📂 F4 密度缓存已加载(扩展={int(round(r * 100))}%)")
        if bundle_key in _density_cache:
            return _density_cache[bundle_key]

    max_ratio = max(F4_EXPAND_RATIOS)
    max_bounds, _ = _get_f4_expanded_bounds(max_ratio)

    ratio_cfg = {}
    for r in F4_EXPAND_RATIOS:
        b, _ = _get_f4_expanded_bounds(r)
        x0, y0, x1, y1 = b
        x_step = (x1 - x0) / DENSITY_GRID_SIZE
        y_step = (y1 - y0) / DENSITY_GRID_SIZE
        ratio_cfg[r] = {
            'bounds': b,
            'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
            'x_step': x_step, 'y_step': y_step,
            'counts': {p: defaultdict(int) for p in period_list},
            'totals': {p: 0 for p in period_list},
        }

    print(
        f"📊 F4 单次扫描构建多扩展缓存: {DENSITY_GRID_SIZE}x{DENSITY_GRID_SIZE}, "
        f"扩展={','.join(str(int(r*100))+'%' for r in F4_EXPAND_RATIOS)}"
    )
    start_time = time.time()

    scanned = 0
    for point in data_loader.iter_points_fast(bounds=max_bounds, point_stride=1):
        p = TimePeriod.get_period(point['timestamp'])
        if p not in period_list:
            continue
        lon = point['lon']
        lat = point['lat']

        # 扩展边界嵌套：15% ⊃ 10% ⊃ 5%，从外到内判断，减少无效网格计算
        for r in reversed(F4_EXPAND_RATIOS):
            cfg = ratio_cfg[r]
            if not (cfg['x0'] <= lon < cfg['x1'] and cfg['y0'] <= lat < cfg['y1']):
                break
            grid_x = int((lon - cfg['x0']) / cfg['x_step'])
            grid_y = int((lat - cfg['y0']) / cfg['y_step'])
            grid_x = max(0, min(grid_x, DENSITY_GRID_SIZE - 1))
            grid_y = max(0, min(grid_y, DENSITY_GRID_SIZE - 1))
            cfg['counts'][p][(grid_x, grid_y)] += 1
            cfg['totals'][p] += 1

        scanned += 1
        if scanned % 2_000_000 == 0:
            print(f"   F4 已扫描 {scanned:,} 点...")

    for r in F4_EXPAND_RATIOS:
        cfg = ratio_cfg[r]
        per_files = {p: get_density_cache_filename(p, r) for p in period_list}
        all_file = get_density_cache_filename(None, r)
        rbundle = {}

        for p in period_list:
            counts = cfg['counts'][p]
            raw_max = max(counts.values()) if counts else 1
            log_max = math.log(raw_max + 1)
            grid_data = []
            for (gx, gy), count in counts.items():
                center_lon = cfg['x0'] + (gx + 0.5) * cfg['x_step']
                center_lat = cfg['y0'] + (gy + 0.5) * cfg['y_step']
                intensity = (math.log(count + 1) / log_max) if log_max > 0 else 0
                grid_data.append({'lon': center_lon, 'lat': center_lat, 'count': count, 'intensity': intensity})
            rbundle[p] = (grid_data, raw_max, cfg['totals'][p])
            with open(per_files[p], 'wb') as f:
                pickle.dump(rbundle[p], f)

        all_counts = defaultdict(int)
        all_total = 0
        for p in period_list:
            for (gx, gy), c in cfg['counts'][p].items():
                all_counts[(gx, gy)] += c
            all_total += cfg['totals'][p]
        all_raw_max = max(all_counts.values()) if all_counts else 1
        all_log_max = math.log(all_raw_max + 1)
        all_grid_data = []
        for (gx, gy), count in all_counts.items():
            center_lon = cfg['x0'] + (gx + 0.5) * cfg['x_step']
            center_lat = cfg['y0'] + (gy + 0.5) * cfg['y_step']
            intensity = (math.log(count + 1) / all_log_max) if all_log_max > 0 else 0
            all_grid_data.append({'lon': center_lon, 'lat': center_lat, 'count': count, 'intensity': intensity})
        rbundle[None] = (all_grid_data, all_raw_max, all_total)
        with open(all_file, 'wb') as f:
            pickle.dump(rbundle[None], f)

        _density_cache[('bundle', r)] = rbundle

    elapsed = time.time() - start_time
    print(f"✅ F4 多扩展缓存构建完成，耗时 {elapsed:.2f} 秒")
    return _density_cache[bundle_key]


def compute_density_for_period(period, expand_ratio=F4_DEFAULT_EXPAND_RATIO, force_recompute=False):
    bundle = _build_density_bundle(expand_ratio=expand_ratio, force_recompute=force_recompute)
    return bundle[period]


def get_density_data_by_period(period, expand_ratio=F4_DEFAULT_EXPAND_RATIO, force_recompute=False):
    if period not in [0, 1, 2]:
        raise ValueError(f"无效的时段: {period}")
    ratio = _normalize_f4_expand_ratio(expand_ratio)
    cache_key = (period, ratio)
    if cache_key not in _density_cache or force_recompute:
        _density_cache[cache_key] = compute_density_for_period(period, ratio, force_recompute)
    return _density_cache[cache_key]


def get_all_day_density(expand_ratio=F4_DEFAULT_EXPAND_RATIO, force_recompute=False):
    bundle = _build_density_bundle(expand_ratio=expand_ratio, force_recompute=force_recompute)
    return bundle[None]


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
_od_by_start = None
_od_by_end = None

def build_od_matrix_by_period(force_recompute=False):
    """构建分时段的 OD 矩阵"""
    global _od_matrix_by_period

    od_meta = {
        "version": CACHE_VERSION,
        "sample_rate": SAMPLE_RATE,
        "od_sample_rate": OD_SAMPLE_RATE,
        "vehicle_count": len(vehicle_list),
    }

    if (
        not force_recompute
        and os.path.exists(OD_MATRIX_FILE)
        and _meta_matches(OD_MATRIX_FILE, od_meta)
    ):
        print(f"📂 加载分时段 OD 矩阵缓存: {OD_MATRIX_FILE}")
        start = time.time()
        with open(OD_MATRIX_FILE, 'rb') as f:
            _od_matrix_by_period = pickle.load(f)
        _reset_od_lookup_index()
        print(f"✅ OD 矩阵加载完成，耗时 {time.time()-start:.2f} 秒")
        print(f"   非零 OD 对: {len(_od_matrix_by_period)}")
        return _od_matrix_by_period

    print("🚗 构建分时段 OD 矩阵...")
    start_time = time.time()

    od_counts = defaultdict(int)
    sampled_vehicles = vehicle_list[:int(len(vehicle_list) * OD_SAMPLE_RATE)]
    print(f"   采样车辆: {len(sampled_vehicles)}")

    trip_count = 0

    for vi, vid in enumerate(sampled_vehicles):
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

        if trip_count and trip_count % 5000 == 0:
            print(f"   已处理 {trip_count} 个行程...")
        if (vi + 1) % 500 == 0:
            print(f"   OD 车辆 {vi + 1}/{len(sampled_vehicles)}, 行程 {trip_count:,}...")

    _od_matrix_by_period = dict(od_counts)
    _reset_od_lookup_index()

    with open(OD_MATRIX_FILE, 'wb') as f:
        pickle.dump(_od_matrix_by_period, f)
    _write_cache_meta(OD_MATRIX_FILE, od_meta)

    elapsed = time.time() - start_time
    print(f"✅ 分时段 OD 矩阵构建完成")
    print(f"   总行程数: {trip_count}")
    print(f"   非零 OD 对: {len(_od_matrix_by_period)}")
    print(f"   耗时: {elapsed:.2f} 秒")

    return _od_matrix_by_period

def _reset_od_lookup_index():
    global _od_by_start, _od_by_end
    _od_by_start = None
    _od_by_end = None


def _ensure_od_lookup_index():
    """按起点/终点建 OD 倒排索引，F5/F6 查询只扫相关网格"""
    global _od_by_start, _od_by_end
    if _od_by_start is not None and _od_by_end is not None:
        return _od_by_start, _od_by_end

    od = get_od_matrix_by_period()
    by_start = defaultdict(list)
    by_end = defaultdict(list)
    t0 = time.time()
    for (start, end, per), count in od.items():
        by_start[start].append((end, per, count))
        by_end[end].append((start, per, count))
    _od_by_start = dict(by_start)
    _od_by_end = dict(by_end)
    print(
        f"📇 OD 查询索引: {len(_od_by_start):,} 起点, "
        f"{len(_od_by_end):,} 终点, 耗时 {time.time() - t0:.2f}s"
    )
    return _od_by_start, _od_by_end


def get_od_matrix_by_period():
    global _od_matrix_by_period
    if _od_matrix_by_period is None:
        _od_matrix_by_period = build_od_matrix_by_period()
    return _od_matrix_by_period


def _flow_between_grid_sets(by_start, grid_ids_1, grid_ids_2, period_filter):
    """period_filter: None=全部时段，或单个 period 整数"""
    flow_1_to_2 = 0
    flow_2_to_1 = 0
    for s in grid_ids_1:
        for e, per, count in by_start.get(s, []):
            if period_filter is not None and per != period_filter:
                continue
            if e in grid_ids_2:
                flow_1_to_2 += count
    for s in grid_ids_2:
        for e, per, count in by_start.get(s, []):
            if period_filter is not None and per != period_filter:
                continue
            if e in grid_ids_1:
                flow_2_to_1 += count
    return flow_1_to_2, flow_2_to_1


def get_flow_between_regions_by_period(rect1, rect2, period=None):
    by_start, _ = _ensure_od_lookup_index()
    grid_ids_1 = set(rect_to_grid_ids(rect1))
    grid_ids_2 = set(rect_to_grid_ids(rect2))

    if period is not None:
        flow_1_to_2, flow_2_to_1 = _flow_between_grid_sets(
            by_start, grid_ids_1, grid_ids_2, period
        )
        return {
            'from_rect1_to_rect2': flow_1_to_2,
            'from_rect2_to_rect1': flow_2_to_1,
            'total': flow_1_to_2 + flow_2_to_1,
        }

    result = {}
    for p in TimePeriod.get_all_periods():
        flow_1_to_2, flow_2_to_1 = _flow_between_grid_sets(
            by_start, grid_ids_1, grid_ids_2, p
        )
        result[TimePeriod.get_period_name(p)] = {
            'from_rect1_to_rect2': flow_1_to_2,
            'from_rect2_to_rect1': flow_2_to_1,
            'total': flow_1_to_2 + flow_2_to_1,
        }
    return result


def get_od_paths_between_regions(rect1, rect2, period=None):
    """
    F5: 列出两区域之间每一条非零 OD 网格路径（用于逐条画弧线）
    period=None 表示全天（三个时段都包含）
    """
    by_start, _ = _ensure_od_lookup_index()
    grid_ids_1 = set(rect_to_grid_ids(rect1))
    grid_ids_2 = set(rect_to_grid_ids(rect2))
    periods_filter = set(TimePeriod.get_all_periods() if period is None else [period])

    paths = []
    for s in grid_ids_1:
        for e, per, count in by_start.get(s, []):
            if per not in periods_filter or count <= 0 or e not in grid_ids_2:
                continue
            entry = _grid_od_path_entry(s, e, per, count, 'A→B')
            if entry:
                paths.append(entry)
    for s in grid_ids_2:
        for e, per, count in by_start.get(s, []):
            if per not in periods_filter or count <= 0 or e not in grid_ids_1:
                continue
            entry = _grid_od_path_entry(s, e, per, count, 'B→A')
            if entry:
                paths.append(entry)

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
    by_start, by_end = _ensure_od_lookup_index()
    grid_ids = set(rect_to_grid_ids(rect))
    periods_filter = set(TimePeriod.get_all_periods() if period is None else [period])

    paths = []
    for s in grid_ids:
        for e, per, count in by_start.get(s, []):
            if per not in periods_filter or count <= 0 or e in grid_ids:
                continue
            entry = _grid_od_path_entry(s, e, per, count, '出发')
            if entry:
                paths.append(entry)
    for e in grid_ids:
        for s, per, count in by_end.get(e, []):
            if per not in periods_filter or count <= 0 or s in grid_ids:
                continue
            entry = _grid_od_path_entry(s, e, per, count, '到达')
            if entry:
                paths.append(entry)

    paths.sort(key=lambda x: x['count'], reverse=True)
    return paths


def _region_flow_for_period(by_start, by_end, grid_ids, period):
    departures = defaultdict(int)
    arrivals = defaultdict(int)
    total_departures = 0
    total_arrivals = 0

    for s in grid_ids:
        for e, per, count in by_start.get(s, []):
            if per != period:
                continue
            departures[e] += count
            total_departures += count
    for e in grid_ids:
        for s, per, count in by_end.get(e, []):
            if per != period:
                continue
            arrivals[s] += count
            total_arrivals += count

    top_destinations = [
        {'grid_id': gid, 'flow': c}
        for gid, c in sorted(departures.items(), key=lambda x: x[1], reverse=True)[:10]
    ]
    top_sources = [
        {'grid_id': gid, 'flow': c}
        for gid, c in sorted(arrivals.items(), key=lambda x: x[1], reverse=True)[:10]
    ]
    return {
        'total_departures': total_departures,
        'total_arrivals': total_arrivals,
        'top_destinations': top_destinations,
        'top_sources': top_sources,
    }


def get_region_flows_by_period(rect, period=None):
    by_start, by_end = _ensure_od_lookup_index()
    grid_ids = set(rect_to_grid_ids(rect))

    if period is not None:
        return _region_flow_for_period(by_start, by_end, grid_ids, period)

    result = {}
    for p in TimePeriod.get_all_periods():
        result[TimePeriod.get_period_name(p)] = _region_flow_for_period(
            by_start, by_end, grid_ids, p
        )
    return result


def _grid_path_coords(path):
    """统计用：网格中心折线"""
    coords = []
    for grid_id in path:
        bounds = grid_id_to_bounds(grid_id)
        if bounds:
            coords.append([(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2])
    return coords


def _rect_bounds_from_dict(rect):
    return (
        float(rect['x_min']), float(rect['y_min']),
        float(rect['x_max']), float(rect['y_max']),
    )


def _point_in_rect(lon, lat, rect_bounds):
    x_min, y_min, x_max, y_max = rect_bounds
    return x_min <= lon <= x_max and y_min <= lat <= y_max


def _trip_crosses_regions(trip, start_grids, end_grids, rect1_bounds, rect2_bounds):
    """行程是否从区域 A 到区域 B（网格首尾或 GPS 首尾）"""
    if len(trip) < 3:
        return False
    seq = trajectory_to_path_sequence(trip, min_grids=2)
    if len(seq) >= 2 and seq[0] in start_grids and seq[-1] in end_grids:
        return True
    p0, p1 = trip[0], trip[-1]
    return (
        _point_in_rect(p0['lon'], p0['lat'], rect1_bounds)
        and _point_in_rect(p1['lon'], p1['lat'], rect2_bounds)
    )


def _backtrack_gps_for_path(path_seq, start_grids=None, end_grids=None):
    """精确匹配 grid 序列；失败则匹配同一起终网格的最长 GPS trip"""
    target = tuple(int(g) for g in path_seq)
    if start_grids is None:
        start_grids = {int(path_seq[0])}
    if end_grids is None:
        end_grids = {int(path_seq[-1])}

    best_coords, best_trip, best_len = None, None, 0
    scan_list = vehicle_list[:REGION_GPS_SCAN_VEHICLES]

    for vid in scan_list:
        df = data_loader.load_vehicle_trajectory(vid)
        if df is None or len(df) < 5:
            continue
        for trip in split_trips(df):
            if len(trip) < 3:
                continue
            seq = trajectory_to_path_sequence(trip, min_grids=2)
            if len(seq) < 2:
                continue
            if seq[0] not in start_grids or seq[-1] not in end_grids:
                continue
            coords = trip_to_display_coords(trip)
            if len(coords) < 2:
                continue
            if tuple(int(g) for g in seq) == target:
                return coords, trip
            if len(coords) > best_len:
                best_len = len(coords)
                best_coords, best_trip = coords, trip

    if best_coords:
        return best_coords, best_trip
    return None, None


def _find_gps_trip_for_regions(rect1_bounds, rect2_bounds):
    """在 A→B 区域间找一条真实 GPS 行程用于展示（不要求 grid 序列完全一致）"""
    cache_key = (
        tuple(round(v, 4) for v in rect1_bounds),
        tuple(round(v, 4) for v in rect2_bounds),
    )
    if cache_key in _region_gps_trip_cache:
        _opt_stats['region_gps_cache_hit'] += 1
        return _region_gps_trip_cache[cache_key]

    start_grids = set(rect_to_grid_ids(rect1_bounds))
    end_grids = set(rect_to_grid_ids(rect2_bounds))
    best_coords, best_trip, best_len = None, None, 0

    for vid in vehicle_list[:REGION_GPS_SCAN_VEHICLES]:
        df = data_loader.load_vehicle_trajectory(vid)
        if df is None or len(df) < 5:
            continue
        for trip in split_trips(df):
            if not _trip_crosses_regions(trip, start_grids, end_grids, rect1_bounds, rect2_bounds):
                continue
            coords = trip_to_display_coords(trip)
            if len(coords) > best_len:
                best_len = len(coords)
                best_coords, best_trip = coords, trip

    _region_gps_trip_cache[cache_key] = (best_coords, best_trip)
    _opt_stats['region_gps_scan'] += 1
    return best_coords, best_trip


def _register_exemplar_from_trip(exemplar_store, path_seq, trip):
    if exemplar_store is None or trip is None or not path_seq:
        return False
    key = tuple(int(g) for g in path_seq)
    prev = exemplar_store.exemplars.get(key)
    exemplar_store.register(path_seq, trip)
    return prev is None or exemplar_store.exemplars.get(key) is not prev


def _resolve_display_coords(
    path,
    grid_coords,
    exemplar_store,
    allow_backtrack=True,
    start_grids=None,
    end_grids=None,
    rect1_bounds=None,
    rect2_bounds=None,
):
    """展示坐标：样例索引 -> 精确/模糊回溯 -> 区域 A→B 真实行程"""
    if exemplar_store is not None:
        coords, ok = exemplar_store.get_display_coords(path, grid_coords)
        if ok and len(coords) >= 2:
            return coords, True, False
    if not allow_backtrack:
        return grid_coords, False, False

    if start_grids is None and path:
        start_grids = {int(path[0])}
    if end_grids is None and path:
        end_grids = {int(path[-1])}

    gps_coords, trip = _backtrack_gps_for_path(path, start_grids, end_grids)
    if gps_coords:
        dirty = _register_exemplar_from_trip(exemplar_store, path, trip)
        return gps_coords, True, dirty

    if rect1_bounds and rect2_bounds:
        gps_coords, trip = _find_gps_trip_for_regions(rect1_bounds, rect2_bounds)
        if gps_coords:
            dirty = _register_exemplar_from_trip(exemplar_store, path, trip)
            return gps_coords, True, dirty

    return grid_coords, False, False


def _gps_display_hint(gps_hits, total):
    if total <= 0:
        return None
    if gps_hits >= total:
        return None
    if gps_hits == 0:
        return (
            '未匹配到真实 GPS 轨迹，地图为网格/路网折线。'
            '请重启并等待 GPS 样例索引补建完成，或扩大区域 A/B。'
        )
    return f'有 {total - gps_hits} 条未匹配 GPS，已用折线代替。'


def _persist_exemplars_if_updated(exemplar_store, size_before, dirty=False):
    if exemplar_store is not None and (len(exemplar_store) > size_before or dirty):
        try:
            exemplar_store.save(PATH_EXEMPLAR_FILE)
        except OSError as e:
            print(f"⚠️ GPS 样例缓存保存失败: {e}")


def _cached_path_distance(path):
    key = tuple(int(g) for g in path)
    cached = _distance_cache.get(key)
    if cached is not None:
        _opt_stats['distance_cache_hit'] += 1
        return cached
    distance = get_path_distance(_grid_path_coords(path))
    if len(_distance_cache) < PATH_DISTANCE_CACHE_MAX:
        _distance_cache[key] = distance
    _opt_stats['distance_computed'] += 1
    return distance


def _select_frequent_paths_by_distance(path_iter, k, min_distance, min_length=2, favor_long_paths=False):
    """先按地理距离过滤，用小顶堆维护 Top-K（流式，不全量排序）"""
    heap = []
    seq = 0
    for path, count in path_iter:
        path_len = len(path)
        if path_len < min_length:
            continue
        distance = _cached_path_distance(path)
        if distance < min_distance:
            continue
        row = {
            'grid_ids': path,
            'count': count,
            'distance': round(distance),
            'grid_coords': _grid_path_coords(path),
            'path_len': path_len,
        }
        seq += 1
        if favor_long_paths:
            length_bonus = 1.0 + min(1.25, math.log1p(max(0, path_len - min_length)) * 0.18)
            score = count * length_bonus
            item = (score, count, seq, row)
            if len(heap) < k:
                heapq.heappush(heap, item)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, item)
        else:
            if len(heap) < k:
                heapq.heappush(heap, (count, seq, row))
            elif count > heap[0][0]:
                heapq.heapreplace(heap, (count, seq, row))
    if favor_long_paths:
        return [row for _, _, _, row in sorted(heap, key=lambda x: (x[0], x[1], x[2]), reverse=True)]
    return [row for _, _, row in sorted(heap, key=lambda x: (x[0], x[1]), reverse=True)]


def _densify_coords(coords, points_per_segment=6):
    if len(coords) < 2:
        return coords
    out = [coords[0]]
    for i in range(len(coords) - 1):
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]
        for t in range(1, points_per_segment + 1):
            a = t / (points_per_segment + 1)
            out.append([x0 + (x1 - x0) * a, y0 + (y1 - y0) * a])
        out.append(coords[i + 1])
    return out


def _smooth_coords(coords, alpha=0.22, rounds=2):
    if len(coords) < 3:
        return coords
    pts = [list(p) for p in coords]
    for _ in range(rounds):
        new_pts = [pts[0]]
        for i in range(1, len(pts) - 1):
            px, py = pts[i - 1]
            cx, cy = pts[i]
            nx, ny = pts[i + 1]
            sx = cx * (1 - alpha) + ((px + nx) / 2.0) * alpha
            sy = cy * (1 - alpha) + ((py + ny) / 2.0) * alpha
            new_pts.append([sx, sy])
        new_pts.append(pts[-1])
        pts = new_pts
    return pts


F7_DISPLAY_COORD_CAP = 120


def _subsample_coords(coords, max_points):
    if len(coords) <= max_points:
        return coords
    if max_points < 2:
        return coords[:1]
    step = max(1, (len(coords) - 1) // (max_points - 1))
    picked = [coords[i] for i in range(0, len(coords), step)]
    if picked[-1] is not coords[-1]:
        picked.append(coords[-1])
    return picked[:max_points]


def _trajectory_like_coords(coords):
    if len(coords) < 2:
        return coords
    coords = _subsample_coords(coords, F7_DISPLAY_COORD_CAP)
    dense = _densify_coords(coords, points_per_segment=6)
    if len(dense) > F7_DISPLAY_COORD_CAP * 3:
        dense = _subsample_coords(dense, F7_DISPLAY_COORD_CAP * 3)
    return _smooth_coords(dense, alpha=0.24, rounds=2)


def _enrich_paths_with_gps_display(
    path_rows,
    exemplar_store=None,
    start_grids=None,
    end_grids=None,
    rect1_bounds=None,
    rect2_bounds=None,
):
    """仅对最终路径解析 GPS 展示坐标；并验证 GPS 起终点是否在区域内"""
    exemplar_before = len(exemplar_store) if exemplar_store is not None else 0
    exemplar_dirty = False
    gps_hits = 0
    filtered_rows = []  # 过滤后的结果
    
    for item in path_rows:
        display_coords, from_gps, dirty = _resolve_display_coords(
            item['grid_ids'],
            item['grid_coords'],
            exemplar_store,
            start_grids=start_grids,
            end_grids=end_grids,
            rect1_bounds=rect1_bounds,
            rect2_bounds=rect2_bounds,
        )
        
        # 验证 GPS 起终点是否在用户框选的区域内
        if from_gps and rect1_bounds and rect2_bounds and len(display_coords) >= 2:
            start_lon, start_lat = display_coords[0]
            end_lon, end_lat = display_coords[-1]
            
            x_min1, y_min1, x_max1, y_max1 = rect1_bounds
            x_min2, y_min2, x_max2, y_max2 = rect2_bounds
            
            start_in_rect1 = (x_min1 <= start_lon <= x_max1 and y_min1 <= start_lat <= y_max1)
            end_in_rect2 = (x_min2 <= end_lon <= x_max2 and y_min2 <= end_lat <= y_max2)
            
            if not start_in_rect1 or not end_in_rect2:
                # GPS 起点或终点不在用户框选的区域内，跳过这条路径
                continue
        
        if not from_gps:
            display_coords = _trajectory_like_coords(display_coords)
        
        item['display_coords'] = display_coords
        item['display_from_gps'] = from_gps
        if from_gps:
            gps_hits += 1
        exemplar_dirty = exemplar_dirty or dirty
        filtered_rows.append(item)
    
    _persist_exemplars_if_updated(exemplar_store, exemplar_before, exemplar_dirty)
    return filtered_rows, gps_hits


def _get_global_frequent_candidates(k, min_distance, min_length=2):
    _opt_stats['f7_queries'] += 1
    favor_long_paths = min_distance <= 0
    return _select_frequent_paths_by_distance(
        path_trie.get_all_paths_iter(),
        k,
        min_distance,
        min_length,
        favor_long_paths=favor_long_paths,
    )


def _get_region_frequent_candidates(start_grids, end_grids, k, min_distance, min_length=2):
    _opt_stats['f8_queries'] += 1
    cache_key = (
        frozenset(start_grids),
        frozenset(end_grids),
        int(k),
        int(min_distance),
        int(min_length),
    )
    cached = _region_query_lru.get(cache_key)
    if cached is not None:
        _opt_stats['region_query_cache_hit'] += 1
        return [dict(r) for r in cached]

    region_paths = path_trie.get_paths_between_regions(
        start_grids,
        end_grids,
        k=max(k * 3, 30),
        min_length=min_length,
        max_depth=TRIE_REGION_MAX_DEPTH,
    )
    result = _select_frequent_paths_by_distance(
        region_paths, k, min_distance, min_length
    )
    _region_query_lru.put(cache_key, result)
    _opt_stats['region_dfs_queries'] += 1
    return result


def _rect_center(rect):
    """矩形 dict 或 (x_min, y_min, x_max, y_max) 元组 -> 中心点"""
    if isinstance(rect, dict):
        x_min = float(rect['x_min'])
        y_min = float(rect['y_min'])
        x_max = float(rect['x_max'])
        y_max = float(rect['y_max'])
    else:
        x_min, y_min, x_max, y_max = (float(v) for v in rect)
    return (x_min + x_max) / 2, (y_min + y_max) / 2


# ========== F9 区域间最短路径优化（参考同学实现） ==========

def _get_valid_grids_in_region(rect_bounds):
    x_min, y_min, x_max, y_max = rect_bounds
    valid_grids = []
    for grid_id in rect_to_grid_ids(rect_bounds):
        bounds = grid_id_to_bounds(grid_id)
        if not bounds:
            continue
        center_lon = (bounds[0] + bounds[2]) / 2
        center_lat = (bounds[1] + bounds[3]) / 2
        if x_min <= center_lon <= x_max and y_min <= center_lat <= y_max:
            valid_grids.append(grid_id)
    return valid_grids


def _get_region_edge_grids_from_valid(valid_grids):
    if not valid_grids:
        return []
    min_x = min(g % GRID_SIZE for g in valid_grids)
    max_x = max(g % GRID_SIZE for g in valid_grids)
    min_y = min(g // GRID_SIZE for g in valid_grids)
    max_y = max(g // GRID_SIZE for g in valid_grids)
    edge_grids = []
    for gid in valid_grids:
        x = gid % GRID_SIZE
        y = gid // GRID_SIZE
        if x == min_x or x == max_x or y == min_y or y == max_y:
            edge_grids.append(gid)
    return edge_grids


def _filter_grids_by_direction(edge_grids, valid_grids, exclude_directions):
    if not edge_grids or not exclude_directions or not valid_grids:
        return edge_grids
    min_x = min(g % GRID_SIZE for g in valid_grids)
    max_x = max(g % GRID_SIZE for g in valid_grids)
    min_y = min(g // GRID_SIZE for g in valid_grids)
    max_y = max(g // GRID_SIZE for g in valid_grids)
    filtered = []
    for gid in edge_grids:
        x = gid % GRID_SIZE
        y = gid // GRID_SIZE
        if 'left' in exclude_directions and x == min_x:
            continue
        if 'right' in exclude_directions and x == max_x:
            continue
        if 'bottom' in exclude_directions and y == min_y:
            continue
        if 'top' in exclude_directions and y == max_y:
            continue
        filtered.append(gid)
    return filtered


def _get_directional_edge_grids_exclude(rect1_bounds, rect2_bounds):
    valid_grids1 = _get_valid_grids_in_region(rect1_bounds)
    valid_grids2 = _get_valid_grids_in_region(rect2_bounds)
    if not valid_grids1 or not valid_grids2:
        return [], []

    all_edge_grids1 = _get_region_edge_grids_from_valid(valid_grids1)
    all_edge_grids2 = _get_region_edge_grids_from_valid(valid_grids2)

    x_min1, y_min1, x_max1, y_max1 = rect1_bounds
    x_min2, y_min2, x_max2, y_max2 = rect2_bounds
    center1_x = (x_min1 + x_max1) / 2
    center1_y = (y_min1 + y_max1) / 2
    center2_x = (x_min2 + x_max2) / 2
    center2_y = (y_min2 + y_max2) / 2
    dx = center2_x - center1_x
    dy = center2_y - center1_y

    min_x1 = min(g % GRID_SIZE for g in valid_grids1)
    max_x1 = max(g % GRID_SIZE for g in valid_grids1)
    min_y1 = min(g // GRID_SIZE for g in valid_grids1)
    max_y1 = max(g // GRID_SIZE for g in valid_grids1)
    min_x2 = min(g % GRID_SIZE for g in valid_grids2)
    max_x2 = max(g % GRID_SIZE for g in valid_grids2)
    min_y2 = min(g // GRID_SIZE for g in valid_grids2)
    max_y2 = max(g // GRID_SIZE for g in valid_grids2)

    has_left1 = any(g % GRID_SIZE == min_x1 for g in all_edge_grids1)
    has_right1 = any(g % GRID_SIZE == max_x1 for g in all_edge_grids1)
    has_bottom1 = any(g // GRID_SIZE == min_y1 for g in all_edge_grids1)
    has_top1 = any(g // GRID_SIZE == max_y1 for g in all_edge_grids1)
    has_left2 = any(g % GRID_SIZE == min_x2 for g in all_edge_grids2)
    has_right2 = any(g % GRID_SIZE == max_x2 for g in all_edge_grids2)
    has_bottom2 = any(g // GRID_SIZE == min_y2 for g in all_edge_grids2)
    has_top2 = any(g // GRID_SIZE == max_y2 for g in all_edge_grids2)

    exclude1, exclude2 = [], []
    if abs(dx) > abs(dy):
        if dx > 0:
            if has_left1 and len(all_edge_grids1) > 4:
                exclude1 = ['left']
            if has_right2 and len(all_edge_grids2) > 4:
                exclude2 = ['right']
        else:
            if has_right1 and len(all_edge_grids1) > 4:
                exclude1 = ['right']
            if has_left2 and len(all_edge_grids2) > 4:
                exclude2 = ['left']
    else:
        if dy > 0:
            if has_bottom1 and len(all_edge_grids1) > 4:
                exclude1 = ['bottom']
            if has_top2 and len(all_edge_grids2) > 4:
                exclude2 = ['top']
        else:
            if has_top1 and len(all_edge_grids1) > 4:
                exclude1 = ['top']
            if has_bottom2 and len(all_edge_grids2) > 4:
                exclude2 = ['bottom']

    grids1 = _filter_grids_by_direction(all_edge_grids1, valid_grids1, exclude1)
    grids2 = _filter_grids_by_direction(all_edge_grids2, valid_grids2, exclude2)
    if not grids1:
        grids1 = all_edge_grids1
    if not grids2:
        grids2 = all_edge_grids2
    return grids1, grids2


def _grids_to_centers(grid_ids):
    centers = []
    for grid_id in grid_ids:
        bounds = grid_id_to_bounds(grid_id)
        if bounds:
            centers.append({
                'grid_id': grid_id,
                'lon': (bounds[0] + bounds[2]) / 2,
                'lat': (bounds[1] + bounds[3]) / 2,
            })
    return centers


def _find_nearest_grid_centers(center_lon, center_lat, target_centers, k=F9_REGION_NEAREST_K):
    distances = []
    for c in target_centers:
        dist = ((c['lon'] - center_lon) ** 2 + (c['lat'] - center_lat) ** 2) ** 0.5
        distances.append((dist, c))
    distances.sort(key=lambda x: x[0])
    return [c for _, c in distances[:min(k, len(distances))]]


def find_shortest_path_between_regions_optimized(rect1_bounds, rect2_bounds, period):
    """区域 A→B：边缘网格候选 + 方向剪枝 + 有限 Dijkstra + 早停"""
    grids1, grids2 = _get_directional_edge_grids_exclude(rect1_bounds, rect2_bounds)
    centers1 = _grids_to_centers(grids1)
    centers2 = _grids_to_centers(grids2)
    if not centers1 or not centers2:
        return None, None, None, None, 0

    k = min(F9_REGION_NEAREST_K, max(len(centers1), len(centers2)))
    best_path = None
    best_time = float('inf')
    best_start = None
    best_end = None
    computed = set()
    total_calculations = 0
    early_stop = False

    if len(centers1) <= len(centers2):
        for c1 in centers1:
            if early_stop:
                break
            nearest = _find_nearest_grid_centers(c1['lon'], c1['lat'], centers2, k)
            start_node, _ = graph.find_nearest_node(c1['lon'], c1['lat'])
            if start_node is None:
                continue
            for c2 in nearest:
                pair_key = (c1['grid_id'], c2['grid_id'])
                if pair_key in computed:
                    continue
                computed.add(pair_key)
                total_calculations += 1
                end_node, _ = graph.find_nearest_node(c2['lon'], c2['lat'])
                if end_node is None:
                    continue
                path, total_time = graph.dijkstra_time(start_node, end_node, period)
                if path and total_time < best_time:
                    best_time = total_time
                    best_path = path
                    best_start = c1
                    best_end = c2
                    if best_time < F9_EARLY_STOP_SECONDS:
                        early_stop = True
                        break
    else:
        for c2 in centers2:
            if early_stop:
                break
            nearest = _find_nearest_grid_centers(c2['lon'], c2['lat'], centers1, k)
            end_node, _ = graph.find_nearest_node(c2['lon'], c2['lat'])
            if end_node is None:
                continue
            for c1 in nearest:
                pair_key = (c1['grid_id'], c2['grid_id'])
                if pair_key in computed:
                    continue
                computed.add(pair_key)
                total_calculations += 1
                start_node, _ = graph.find_nearest_node(c1['lon'], c1['lat'])
                if start_node is None:
                    continue
                path, total_time = graph.dijkstra_time(start_node, end_node, period)
                if path and total_time < best_time:
                    best_time = total_time
                    best_path = path
                    best_start = c1
                    best_end = c2
                    if best_time < F9_EARLY_STOP_SECONDS:
                        early_stop = True
                        break

    return best_path, best_time, best_start, best_end, total_calculations


def _compare_periods_between_regions_optimized(rect1_bounds, rect2_bounds):
    """各时段最短时间（在边缘网格对上快速估计，避免三次全量搜索）"""
    grids1, grids2 = _get_directional_edge_grids_exclude(rect1_bounds, rect2_bounds)
    centers1 = _grids_to_centers(grids1)
    centers2 = _grids_to_centers(grids2)
    best_times = {p: float('inf') for p in TimePeriod.get_all_periods()}
    if not centers1 or not centers2:
        return best_times

    for period in TimePeriod.get_all_periods():
        for c1 in centers1:
            start_node, _ = graph.find_nearest_node(c1['lon'], c1['lat'])
            if start_node is None:
                continue
            for c2 in _find_nearest_grid_centers(c1['lon'], c1['lat'], centers2, 5):
                end_node, _ = graph.find_nearest_node(c2['lon'], c2['lat'])
                if end_node is None:
                    continue
                _, t = graph.dijkstra_time(start_node, end_node, period)
                if t and t < best_times[period]:
                    best_times[period] = t
    return best_times


def _parse_f9_points(data):
    """F9: 支持矩形区域中心或起终点坐标"""
    rect1, rect2 = data.get('rect1'), data.get('rect2')
    if rect1 and rect2:
        slon, slat = _rect_center(rect1)
        elon, elat = _rect_center(rect2)
    else:
        slon = float(data['start_lon']) if data.get('start_lon') is not None else None
        slat = float(data['start_lat']) if data.get('start_lat') is not None else None
        elon = float(data['end_lon']) if data.get('end_lon') is not None else None
        elat = float(data['end_lat']) if data.get('end_lat') is not None else None
    return slon, slat, elon, elat


def _graph_path_to_grid_sequence(node_path):
    seq = []
    for node_id in node_path:
        coords = graph.get_node_coords(node_id)
        if not coords:
            continue
        gid = lonlat_to_grid_id(coords[0], coords[1])
        if gid != -1 and (not seq or seq[-1] != gid):
            seq.append(gid)
    return seq


def _graph_node_path_coords(node_path):
    coords = []
    for node_id in node_path:
        pt = graph.get_node_coords(node_id)
        if pt:
            coords.append([float(pt[0]), float(pt[1])])
    return coords


def _simplify_path_coords(coords):
    """去掉共线中间点，减轻网格折线的锯齿感"""
    if len(coords) <= 2:
        return coords
    out = [coords[0]]
    for i in range(1, len(coords) - 1):
        ax, ay = out[-1]
        bx, by = coords[i]
        cx, cy = coords[i + 1]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if abs(cross) > 1e-12:
            out.append(coords[i])
    out.append(coords[-1])
    return out


def _resolve_graph_path_display(node_path):
    """F9: 展示与 Dijkstra 一致的路网折线（不做 GPS 替换，避免路线偏离/变慢）"""
    coords = _simplify_path_coords(_graph_node_path_coords(node_path))
    return coords, False


def _f9_resolve_endpoints(data, slon, slat, elon, elat):
    """在矩形区域内找起终点路网节点；无矩形时用全局最近节点"""
    rect1_bounds = rect2_bounds = None
    if data.get('rect1') and data.get('rect2'):
        rect1_bounds = _rect_bounds_from_dict(data['rect1'])
        rect2_bounds = _rect_bounds_from_dict(data['rect2'])
        start_node, _ = graph.find_nearest_node_in_rect(slon, slat, rect1_bounds)
        end_node, _ = graph.find_nearest_node_in_rect(elon, elat, rect2_bounds)
    else:
        start_node, _ = graph.find_nearest_node(slon, slat)
        end_node, _ = graph.find_nearest_node(elon, elat)
    return start_node, end_node, rect1_bounds, rect2_bounds


def _generate_shortest_path_map_payload(path_coords, color=None, layer_id='shortest-path'):
    if len(path_coords) < 2:
        return build_payload(DEFAULT_VIEW_STATE, [], '{type}', lock_viewport=True)
    slon, slat = path_coords[0]
    elon, elat = path_coords[-1]
    layers = [
        path_layer([{
            'path': path_coords,
            'color': color or [255, 100, 50, 255],
            'width': 5,
        }], layer_id),
    ]
    markers = [
        {'lon': slon, 'lat': slat, 'type': '起点', 'color': [50, 200, 50, 255]},
        {'lon': elon, 'lat': elat, 'type': '终点', 'color': [200, 50, 50, 255]},
    ]
    layers.append(scatter_layer(markers, layer_id='path-markers', radius=80))
    vs = view_state_for_coords(
        [p[0] for p in path_coords],
        [p[1] for p in path_coords],
    )
    return build_payload(vs, layers, '{type}', lock_viewport=True)


def _generate_f9_region_path_map_payload(
    path_coords, rect1_bounds, rect2_bounds, best_start=None, best_end=None,
):
    """F9 区域模式：A/B 高亮 + 最优起终点网格 + 路网折线"""
    x_min1, y_min1, x_max1, y_max1 = rect1_bounds
    x_min2, y_min2, x_max2, y_max2 = rect2_bounds
    layers = [
        polygon_layer([{
            'polygon': [[x_min1, y_min1], [x_max1, y_min1], [x_max1, y_max1], [x_min1, y_max1]],
            'fill_color': [0, 200, 0, 15],
            'line_color': [0, 255, 0, 255],
        }], layer_id='f9-region-a', line_width_min_pixels=2),
        polygon_layer([{
            'polygon': [[x_min2, y_min2], [x_max2, y_min2], [x_max2, y_max2], [x_min2, y_max2]],
            'fill_color': [255, 100, 0, 15],
            'line_color': [255, 100, 0, 255],
        }], layer_id='f9-region-b', line_width_min_pixels=2),
    ]
    markers = []
    if best_start:
        markers.append({
            'lon': best_start['lon'],
            'lat': best_start['lat'],
            'type': f"最优起点(网格{best_start['grid_id']})",
            'color': [0, 255, 0, 255],
        })
    if best_end:
        markers.append({
            'lon': best_end['lon'],
            'lat': best_end['lat'],
            'type': f"最优终点(网格{best_end['grid_id']})",
            'color': [255, 120, 0, 255],
        })
    if markers:
        layers.append(scatter_layer(markers, layer_id='f9-optimal-grids', radius=80))
    if path_coords and len(path_coords) >= 2:
        layers.append(path_layer([{
            'path': path_coords,
            'color': [255, 100, 50, 255],
            'width': 5,
        }], 'f9-shortest-path'))
    all_lons = [x_min1, x_max1, x_min2, x_max2]
    all_lats = [y_min1, y_max1, y_min2, y_max2]
    if path_coords:
        all_lons.extend(p[0] for p in path_coords)
        all_lats.extend(p[1] for p in path_coords)
    vs = view_state_for_coords(all_lons, all_lats)
    return build_payload(vs, layers, '{type}', lock_viewport=True)


def _generate_compare_paths_map_payload(path_items):
    """多条时段路径对比（不同颜色）；起终点取第一条有效路径端点"""
    records = []
    all_lons, all_lats = [], []
    marker_lon, marker_lat, end_lon, end_lat = None, None, None, None
    for coords, color, _label in path_items:
        if len(coords) < 2:
            continue
        if marker_lon is None:
            marker_lon, marker_lat = coords[0]
            end_lon, end_lat = coords[-1]
        all_lons.extend(p[0] for p in coords)
        all_lats.extend(p[1] for p in coords)
        records.append({'path': coords, 'color': color, 'width': 4})
    layers = [path_layer(records, 'compare-paths')] if records else []
    if marker_lon is not None:
        layers.append(scatter_layer([
            {'lon': marker_lon, 'lat': marker_lat, 'type': '起点', 'color': [50, 200, 50, 255]},
            {'lon': end_lon, 'lat': end_lat, 'type': '终点', 'color': [200, 50, 50, 255]},
        ], layer_id='path-markers', radius=80))
        all_lons.extend([marker_lon, end_lon])
        all_lats.extend([marker_lat, end_lat])
    vs = view_state_for_coords(all_lons, all_lats) if all_lons else DEFAULT_VIEW_STATE
    return build_payload(vs, layers, '{type}', lock_viewport=True)


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


def _f6_is_edge_region(rect_bounds):
    x_min, y_min, x_max, y_max = rect_bounds
    bx0, by0, bx1, by1 = BOUNDS
    mx = (bx1 - bx0) * F6_BALANCE_EDGE_MARGIN_RATIO
    my = (by1 - by0) * F6_BALANCE_EDGE_MARGIN_RATIO
    return (
        x_min <= bx0 + mx or x_max >= bx1 - mx
        or y_min <= by0 + my or y_max >= by1 - my
    )


def _f6_direction_sector_from_path(p):
    sx, sy = p['source'][0], p['source'][1]
    tx, ty = p['target'][0], p['target'][1]
    dx = tx - sx
    dy = ty - sy
    angle = math.atan2(dy, dx)
    sector = int(((angle + math.pi) / (2 * math.pi)) * 8) % 8
    quadrant = int(((angle + math.pi) / (2 * math.pi)) * 4) % 4
    return sector, quadrant


def _f6_balance_metrics(od_paths):
    if not od_paths:
        return {'imbalance': 0.0, 'dominant_ratio': 0.0, 'sector_counts': [0] * 8}
    sector_counts = [0] * 8
    weighted_total = 0
    for p in od_paths:
        s, _ = _f6_direction_sector_from_path(p)
        w = max(1, int(p.get('count', 1)))
        sector_counts[s] += w
        weighted_total += w
    if weighted_total <= 0:
        return {'imbalance': 0.0, 'dominant_ratio': 0.0, 'sector_counts': sector_counts}
    dominant = max(sector_counts)
    dominant_ratio = dominant / weighted_total
    even_ratio = 1.0 / 8.0
    imbalance = max(0.0, (dominant_ratio - even_ratio) / (1.0 - even_ratio))
    return {
        'imbalance': float(imbalance),
        'dominant_ratio': float(dominant_ratio),
        'sector_counts': sector_counts,
    }


def _f6_quota_sample_paths(od_paths, cap=OD_MAX_ARCS):
    if len(od_paths) <= cap:
        return od_paths

    buckets = {i: [] for i in range(8)}
    bucket_flow = {i: 0 for i in range(8)}
    for p in od_paths:
        s, _ = _f6_direction_sector_from_path(p)
        buckets[s].append(p)
        bucket_flow[s] += max(1, int(p.get('count', 1)))

    for i in range(8):
        buckets[i].sort(key=lambda x: x.get('count', 0), reverse=True)

    non_empty = [i for i in range(8) if buckets[i]]
    if not non_empty:
        return od_paths[:cap]

    min_keep = 1
    picked = []
    picked_ids = set()
    for i in non_empty:
        for p in buckets[i][:min_keep]:
            pid = (p.get('start_grid'), p.get('end_grid'), p.get('period'), p.get('direction'))
            if pid not in picked_ids:
                picked.append(p)
                picked_ids.add(pid)

    remain = max(0, cap - len(picked))
    if remain <= 0:
        return picked[:cap]

    total_flow = sum(bucket_flow[i] for i in non_empty)
    alloc = {i: 0 for i in non_empty}
    if total_flow > 0:
        raw_alloc = {i: remain * (bucket_flow[i] / total_flow) for i in non_empty}
        for i in non_empty:
            alloc[i] = int(raw_alloc[i])
        left = remain - sum(alloc.values())
        if left > 0:
            order = sorted(non_empty, key=lambda i: (raw_alloc[i] - alloc[i]), reverse=True)
            for i in order[:left]:
                alloc[i] += 1

    for i in non_empty:
        needed = alloc[i]
        if needed <= 0:
            continue
        for p in buckets[i][min_keep:min_keep + needed]:
            pid = (p.get('start_grid'), p.get('end_grid'), p.get('period'), p.get('direction'))
            if pid not in picked_ids:
                picked.append(p)
                picked_ids.add(pid)

    if len(picked) < cap:
        rest = []
        for i in non_empty:
            rest.extend(buckets[i][min_keep + alloc[i]:])
        rest.sort(key=lambda x: x.get('count', 0), reverse=True)
        for p in rest:
            if len(picked) >= cap:
                break
            pid = (p.get('start_grid'), p.get('end_grid'), p.get('period'), p.get('direction'))
            if pid not in picked_ids:
                picked.append(p)
                picked_ids.add(pid)

    return picked[:cap]


def _od_paths_to_arcs(od_paths, mode='f5', balance_mode=False):
    """每条网格 OD 路径对应一条拱形弧线。mode: f5 | f6"""
    total = len(od_paths)
    if total > OD_MAX_ARCS:
        if mode == 'f6' and balance_mode:
            od_paths = _f6_quota_sample_paths(od_paths, OD_MAX_ARCS)
        else:
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
        sector, quadrant = _f6_direction_sector_from_path(p)
        jitter = ((p['start_grid'] * 13 + p['end_grid'] * 7) % 15) * 0.015

        base_h = period_base_height.get(pname, 0.3)
        if mode == 'f6' and balance_mode:
            sector_height = (sector - 3.5) * 0.06
            quadrant_height = (quadrant - 1.5) * 0.09
            direction_boost = 0.10 if p['direction'] == dir_primary else 0.03
            height = max(0.10, base_h + sector_height + quadrant_height + direction_boost + jitter)
        else:
            height = base_h + jitter

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

    edge_region = _f6_is_edge_region(rect_bounds)
    balance_metrics = _f6_balance_metrics(od_paths)
    imbalance = balance_metrics['imbalance']
    auto_balance_mode = (
        len(od_paths) >= F6_BALANCE_MIN_TOTAL_PATHS
        and (edge_region or imbalance >= F6_BALANCE_DIRECTION_IMBALANCE_THRESHOLD)
    )

    arcs, total_paths = _od_paths_to_arcs(
        od_paths,
        mode='f6',
        balance_mode=auto_balance_mode,
    )

    layers = [polygon_layer(polygons)]
    if arcs:
        layers.append(arc_layer(arcs))
    vs = view_state_for_coords([x_min, x_max], [y_min, y_max])
    vs['zoom'] = max(vs.get('zoom', 11), 11)
    meta = {
        'arc_count': len(arcs),
        'total_paths': total_paths,
        'capped': total_paths > OD_MAX_ARCS,
        'balance_mode': auto_balance_mode,
        'edge_region': edge_region,
        'flow_imbalance': imbalance,
        'boundary_hint': (
            '区域贴近研究边界，已自动优化飞线展示均衡性'
            if edge_region else None
        ),
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


def generate_f4_density_map_payload(heatmap_period=None, expand_ratio=F4_DEFAULT_EXPAND_RATIO):
    """F4 专用：平面 2D 热力图 + 立体柱"""
    ratio = _normalize_f4_expand_ratio(expand_ratio)
    if heatmap_period is None:
        grid_data, max_count, total_points = get_all_day_density(ratio)
        period_name = '全天'
    else:
        grid_data, max_count, total_points = get_density_data_by_period(heatmap_period, ratio)
        period_name = TimePeriod.get_period_name(heatmap_period)

    for item in grid_data:
        item['color'] = get_color_from_intensity(item['intensity'])

    heatmap_data = [
        {
            'lon': item['lon'],
            'lat': item['lat'],
            'weight': float(item['count']),
            'count': item['count'],
        }
        for item in grid_data
    ]
    layers = [
        heatmap_layer(heatmap_data),
        column_layer(grid_data, max_count, period_name),
    ]
    vs = dict(DEFAULT_VIEW_STATE)
    vs['pitch'] = 45
    return build_payload(
        vs,
        layers,
        f'密度: {{count}} | 时段: {period_name}',
        meta={
            'period_name': period_name,
            'total_points': total_points,
            'layers': 'heatmap+column',
            'expand_ratio': ratio,
            'expand_ratio_options': F4_EXPAND_RATIOS,
        },
        lock_viewport=True,
    )


def _search_rect_polygon(rect):
    """F3 探查区域：浅绿填充 + 绿色描边"""
    x_min = float(rect['x_min'])
    y_min = float(rect['y_min'])
    x_max = float(rect['x_max'])
    y_max = float(rect['y_max'])
    return {
        'polygon': [
            [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max],
        ],
        'fill_color': [0, 200, 0, 20],
        'line_color': [0, 255, 0, 255],
    }, (x_min, y_min, x_max, y_max)


def generate_map_payload(points_data=None, show_heatmap=False, path_coords=None,
                       start_lon=None, start_lat=None, end_lon=None, end_lat=None,
                       heatmap_period=None, search_rect=None):
    layers = []
    tooltip_text = ''
    vs = dict(DEFAULT_VIEW_STATE)
    all_lons, all_lats = [], []

    if show_heatmap:
        return generate_f4_density_map_payload(heatmap_period)

    if search_rect and all(k in search_rect for k in ('x_min', 'y_min', 'x_max', 'y_max')):
        poly, bounds = _search_rect_polygon(search_rect)
        layers.append(polygon_layer(
            [poly], layer_id='f3-search-region', line_width_min_pixels=3,
        ))
        x_min, y_min, x_max, y_max = bounds
        all_lons.extend([x_min, x_max])
        all_lats.extend([y_min, y_max])

    if path_coords and len(path_coords) > 0:
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
        for p in points_data:
            all_lons.append(p['lon'])
            all_lats.append(p['lat'])
        tooltip_text = '车辆ID: {taxi_id}'

    if all_lons and all_lats:
        vs = view_state_for_coords(all_lons, all_lats)
    elif search_rect:
        vs = view_state_for_coords(all_lons, all_lats)

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

    snap_to_road = bool(data.get('snap_to_road', False)) and mode == 'single'
    paths, err, snap_used = build_trajectory_paths(
        mode, vehicle_id, max_vehicles, point_sample, snap_to_road=snap_to_road,
    )
    if err:
        return jsonify({'error': err}), 400

    total_points = sum(len(p['path']) for p in paths)
    elapsed = time.time() - start
    print(
        f"🛤️ F1 轨迹: mode={mode}, 车辆数={len(paths)}, 点数={total_points}, "
        f"路网贴合={'是' if snap_used else '否'}, 耗时={elapsed:.3f}s"
    )

    payload = {
        'success': True,
        'mode': mode,
        'vehicle_count': len(paths),
        'point_count': total_points,
        'query_time': elapsed,
        'vehicle_ids': [p['taxi_id'] for p in paths],
        'snap_to_road': snap_used,
        'map': generate_trajectory_map_payload(paths),
    }
    if snap_to_road and not snap_used:
        payload['display_note'] = '路网贴合失败，已显示原始 GPS 折线'
    elif snap_used:
        payload['display_note'] = '轨迹已沿学习路网贴合（近似马路，非真实导航路网）'
    return jsonify(payload)


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
    return jsonify({
        'count': total,
        'returned': len(result),
        'query_time': time.time() - start,
        'map': generate_map_payload(points_data=result, search_rect=rect),
    })


@app.route('/api/density_view', methods=['GET'])
def density_view():
    expand_ratio = _normalize_f4_expand_ratio(request.args.get('expand_ratio', F4_DEFAULT_EXPAND_RATIO))
    return jsonify({'map': generate_f4_density_map_payload(heatmap_period=None, expand_ratio=expand_ratio)})


@app.route('/api/density_by_period', methods=['POST'])
def density_by_period():
    data = request.json or {}
    period = data.get('period')
    expand_ratio = _normalize_f4_expand_ratio(data.get('expand_ratio', F4_DEFAULT_EXPAND_RATIO))
    if period == 'all' or period is None:
        period = None
    else:
        period = int(period)

    if period is None:
        _, _, total_points = get_all_day_density(expand_ratio)
        period_name = "全天"
    else:
        _, _, total_points = get_density_data_by_period(period, expand_ratio)
        period_name = TimePeriod.get_period_name(period)

    return jsonify({
        'period': period,
        'period_name': period_name,
        'total_points': total_points,
        'expand_ratio': expand_ratio,
        'expand_ratio_options': F4_EXPAND_RATIOS,
        'map': generate_f4_density_map_payload(heatmap_period=period, expand_ratio=expand_ratio),
    })


@app.route('/api/density_stats', methods=['GET'])
def density_stats():
    expand_ratio = _normalize_f4_expand_ratio(
        request.args.get('expand_ratio', F4_DEFAULT_EXPAND_RATIO)
    )
    stats = []
    _, _, total_points = get_all_day_density(expand_ratio)
    stats.append({'name': '全天', 'total_points': total_points})
    for period in TimePeriod.get_all_periods():
        _, _, total_points = get_density_data_by_period(period, expand_ratio)
        stats.append({'name': TimePeriod.get_period_name(period), 'total_points': total_points})
    return jsonify({
        'expand_ratio': expand_ratio,
        'stats': stats,
    })


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
    data = request.json or {}
    period = data.get('period', TimePeriod.OFF_PEAK)
    rect1, rect2 = data.get('rect1'), data.get('rect2')

    if rect1 and rect2:
        rect1_bounds = _rect_bounds_from_dict(rect1)
        rect2_bounds = _rect_bounds_from_dict(rect2)
        best_path, total_time, best_start, best_end, calc_count = (
            find_shortest_path_between_regions_optimized(rect1_bounds, rect2_bounds, period)
        )
        if best_path is None or total_time == float('inf'):
            return jsonify({'error': '未找到路径'}), 400

        coords, from_gps = _resolve_graph_path_display(best_path)
        elapsed = time.time() - start
        time_str = f"{total_time/60:.1f}分钟" if total_time > 60 else f"{total_time:.0f}秒"
        period_name = TimePeriod.get_period_name(period)
        print(
            f"🛣️ [F9区域优化] {time_str}, {period_name}, "
            f"Dijkstra×{calc_count}, 节点{len(best_path)}, 耗时{elapsed:.3f}s"
        )
        return jsonify({
            'success': True,
            'time_seconds': round(total_time),
            'time_str': time_str,
            'period': period_name,
            'node_count': len(best_path),
            'start_grid': best_start['grid_id'],
            'end_grid': best_end['grid_id'],
            'dijkstra_calls': calc_count,
            'display_from_gps': from_gps,
            'display_note': (
                f'区域边缘网格优化搜索（{calc_count} 次 Dijkstra）；'
                f'绿/橙大点为最优起终点网格 {best_start["grid_id"]}→{best_end["grid_id"]}'
            ),
            'query_time': elapsed,
            'map': _generate_f9_region_path_map_payload(
                coords, rect1_bounds, rect2_bounds, best_start, best_end,
            ),
        })

    slon, slat, elon, elat = _parse_f9_points(data)
    if None in [slon, slat, elon, elat]:
        return jsonify({'error': '缺少起点/终点（坐标或矩形区域）'}), 400

    start_node, end_node, _, _ = _f9_resolve_endpoints(data, slon, slat, elon, elat)
    if start_node is None or end_node is None:
        return jsonify({'error': '无法找到附近路网节点'}), 400

    path, total_time = graph.dijkstra_time(start_node, end_node, period)
    if path is None:
        return jsonify({'error': '未找到路径'}), 400

    coords, from_gps = _resolve_graph_path_display(path)
    elapsed = time.time() - start
    time_str = f"{total_time/60:.1f}分钟" if total_time > 60 else f"{total_time:.0f}秒"
    period_name = TimePeriod.get_period_name(period)
    print(
        f"🛣️ 路径: {len(path)}节点, {time_str}, {period_name}, "
        f"路网节点{len(path)}, 耗时{elapsed:.3f}s"
    )
    return jsonify({
        'success': True,
        'time_seconds': round(total_time),
        'time_str': time_str,
        'period': period_name,
        'node_count': len(path),
        'display_from_gps': from_gps,
        'display_note': '时间为学习路网上的最短时间（单点 Dijkstra）',
        'query_time': elapsed,
        'map': _generate_shortest_path_map_payload(coords),
    })


@app.route('/api/compare_paths', methods=['POST'])
def compare_paths():
    data = request.json or {}
    rect1, rect2 = data.get('rect1'), data.get('rect2')

    if rect1 and rect2:
        rect1_bounds = _rect_bounds_from_dict(rect1)
        rect2_bounds = _rect_bounds_from_dict(rect2)
        best_times = _compare_periods_between_regions_optimized(rect1_bounds, rect2_bounds)
        results = {}
        for period in TimePeriod.get_all_periods():
            t = best_times[period]
            if t != float('inf'):
                results[TimePeriod.get_period_name(period)] = {
                    'time_seconds': round(t),
                    'time_str': f"{t/60:.1f}分钟" if t > 60 else f"{t:.0f}秒",
                }
        if not results:
            return jsonify({'error': '未找到任一时段路径'}), 400
        return jsonify({
            'periods': results,
            'display_note': '各时段在区域边缘网格上的最短时间估计（快速对比）',
        })

    slon, slat, elon, elat = _parse_f9_points(data)
    if None in [slon, slat, elon, elat]:
        return jsonify({'error': '缺少起点/终点（坐标或矩形区域）'}), 400

    start_node, end_node, _, _ = _f9_resolve_endpoints(data, slon, slat, elon, elat)
    if start_node is None or end_node is None:
        return jsonify({'error': '无法找到附近路网节点'}), 400

    period_colors = [
        [255, 120, 80, 220],
        [100, 160, 255, 220],
        [80, 210, 120, 220],
    ]
    results = {}
    path_items = []
    gps_hits = 0

    for i, period in enumerate(TimePeriod.get_all_periods()):
        node_path, t = graph.dijkstra_time(start_node, end_node, period)
        name = TimePeriod.get_period_name(period)
        if node_path and t is not None:
            coords, from_gps = _resolve_graph_path_display(node_path)
            if from_gps:
                gps_hits += 1
            results[name] = {
                'time_seconds': round(t),
                'time_str': f"{t/60:.1f}分钟" if t > 60 else f"{t:.0f}秒",
                'node_count': len(node_path),
                'display_from_gps': from_gps,
            }
            path_items.append((coords, period_colors[i % len(period_colors)], name))

    payload = {
        'periods': results,
        'gps_display_count': gps_hits,
        'exemplar_index_size': len(path_exemplars),
    }
    if path_items:
        payload['map'] = _generate_compare_paths_map_payload(path_items)
        payload['display_note'] = '各时段最短时间路径（单点 Dijkstra）'
    return jsonify(payload)


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
    """F7: 全局 Top-k（网格统计 + GPS 样例展示）"""
    data = request.json or {}
    k = data.get('k', 10)
    min_distance = data.get('min_distance', 0)

    min_length = data.get('min_length', 2)
    staged = _get_global_frequent_candidates(k, min_distance, min_length=min_length)
    result_paths, gps_hits = _enrich_paths_with_gps_display(staged, path_exemplars)
    paths_for_map = [(p['display_coords'], p['count'], None) for p in result_paths]

    total = len(result_paths)
    return jsonify({
        'success': True,
        'k': k,
        'min_distance': min_distance,
        'total_paths_found': total,
        'gps_display_count': gps_hits,
        'display_hint': _gps_display_hint(gps_hits, total),
        'exemplar_index_size': len(path_exemplars),
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

    rect1_bounds = _rect_bounds_from_dict(rect1)
    rect2_bounds = _rect_bounds_from_dict(rect2)

    start_grids = set(rect_to_grid_ids(rect1_bounds))
    end_grids = set(rect_to_grid_ids(rect2_bounds))

    staged = _get_region_frequent_candidates(
        start_grids, end_grids, k, min_distance, min_length=2
    )
    result_paths, gps_hits = _enrich_paths_with_gps_display(
        staged,
        path_exemplars,
        start_grids=start_grids,
        end_grids=end_grids,
        rect1_bounds=rect1_bounds,
        rect2_bounds=rect2_bounds,
    )
    paths_for_map = [(p['display_coords'], p['count'], None) for p in result_paths]

    total = len(result_paths)
    return jsonify({
        'success': True,
        'k': k,
        'min_distance': min_distance,
        'total_paths_found': total,
        'gps_display_count': gps_hits,
        'display_hint': _gps_display_hint(gps_hits, total),
        'exemplar_index_size': len(path_exemplars),
        'top_paths': result_paths,
        'map': generate_paths_with_regions_map_payload(paths_for_map, rect1_bounds, rect2_bounds),
    })


@app.route('/api/performance_stats', methods=['GET'])
def performance_stats():
    """F7-F9 性能优化指标"""
    gps_hits = _opt_stats.get('distance_cache_hit', 0)
    gps_miss = _opt_stats.get('distance_computed', 0)
    dist_total = gps_hits + gps_miss
    return jsonify({
        'optimization': {
            'distance_cache': {
                'size': len(_distance_cache),
                'max_size': PATH_DISTANCE_CACHE_MAX,
                'hit_rate': f'{100 * gps_hits / dist_total:.1f}%' if dist_total else 'N/A',
            },
            'region_query_cache': {
                'size': len(_region_query_lru),
                'max_size': PATH_QUERY_CACHE_SIZE,
            },
            'region_gps_trip_cache': {'size': len(_region_gps_trip_cache)},
            'gps_exemplar_index': {'size': len(path_exemplars)},
            'trie': {
                'method': 'DFS剪枝(F8) + 流式迭代(F7) + TopK小顶堆',
                'region_max_depth': TRIE_REGION_MAX_DEPTH,
            },
            'counters': dict(_opt_stats),
        },
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
        'data_config': {
            'sample_rate': SAMPLE_RATE,
            'quadtree_point_stride': QUADTREE_POINT_STRIDE,
            'path_trie_sample_rate': PATH_TRIE_SAMPLE_RATE,
            'path_exemplar_max': PATH_EXEMPLAR_MAX_ENTRIES,
            'graph_sample_vehicles': GRAPH_SAMPLE_VEHICLES,
            'od_sample_rate': OD_SAMPLE_RATE,
        },
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
    print("   - F7/F8: 频繁路径（网格统计 + GPS 样例展示）")
    print("   - F9: 区域边缘网格优化 Dijkstra + A/B 高亮与最优网格起终点")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000, use_reloader=False)