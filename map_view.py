"""
地图 JSON 载荷（供前端 deck.gl 渲染）+ F2 视野 LOD
"""

import math

DEFAULT_VIEW_STATE = {
    'longitude': 116.51,
    'latitude': 39.92,
    'zoom': 11,
    'pitch': 0,
    'bearing': 0,
}

TRAJECTORY_COLORS = [
    [65, 150, 210, 220], [255, 100, 50, 220], [50, 200, 120, 220],
    [255, 200, 50, 220], [200, 80, 200, 220], [80, 180, 255, 220],
    [255, 120, 150, 220], [150, 120, 80, 220], [120, 200, 80, 220],
    [180, 100, 255, 220],
]


def lod_for_zoom(zoom):
    """F2: 缩放越远，采样越稀、点数越少（控制渲染量减轻卡顿）"""
    z = float(zoom)
    if z <= 10:
        return {'max_points': 1200, 'stride': 25, 'radius': 12}
    if z <= 11.5:
        return {'max_points': 3000, 'stride': 10, 'radius': 16}
    if z <= 13:
        return {'max_points': 6000, 'stride': 5, 'radius': 20}
    return {'max_points': 10000, 'stride': 3, 'radius': 24}


def bbox_from_view(longitude, latitude, zoom, width=1200, height=800):
    scale = 512 * (2 ** float(zoom))
    lon_delta = 360.0 / scale * width / 2
    lat_rad = math.radians(float(latitude))
    lat_delta = 360.0 / scale * height / 2 / max(math.cos(lat_rad), 0.01)
    return [
        longitude - lon_delta,
        latitude - lat_delta,
        longitude + lon_delta,
        latitude + lat_delta,
    ]


def thin_quadtree_points(raw_points, zoom, bbox=None):
    """在视野 bbox 内网格分层采样，保证四角也有点（避免只取遍历前 N 条）。"""
    lod = lod_for_zoom(zoom)
    max_points = lod['max_points']

    if not raw_points:
        return [], lod

    if bbox is not None and len(bbox) >= 4:
        x_min, y_min, x_max, y_max = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    else:
        x_min = min(p[0] for p in raw_points)
        y_min = min(p[1] for p in raw_points)
        x_max = max(p[0] for p in raw_points)
        y_max = max(p[1] for p in raw_points)

    span_lon = max(x_max - x_min, 1e-9)
    span_lat = max(y_max - y_min, 1e-9)

    pool = raw_points
    if len(pool) > max_points * 12:
        step = max(1, len(pool) // (max_points * 12))
        pool = pool[::step]

    aspect = span_lon / span_lat
    grid_side = max(4, int(math.sqrt(max_points)))
    grid_cols = max(4, min(72, int(grid_side * math.sqrt(aspect))))
    grid_rows = max(4, min(72, int(grid_side / math.sqrt(aspect))))

    cells = {}
    for p in pool:
        gx = min(grid_cols - 1, int((p[0] - x_min) / span_lon * grid_cols))
        gy = min(grid_rows - 1, int((p[1] - y_min) / span_lat * grid_rows))
        key = (gx, gy)
        if key not in cells:
            cells[key] = p

    selected = list(cells.values())
    if len(selected) > max_points:
        step = max(1, len(selected) // max_points)
        selected = selected[::step][:max_points]

    points = [
        {
            'lon': p[0],
            'lat': p[1],
            'taxi_id': p[2],
            'timestamp': str(p[3]),
        }
        for p in selected
    ]
    return points, lod


def build_payload(view_state, layers, tooltip, meta=None, lock_viewport=False):
    vs = dict(DEFAULT_VIEW_STATE)
    if view_state:
        vs.update(view_state)
    return {
        'viewState': vs,
        'layers': layers,
        'tooltip': tooltip or '',
        'meta': meta or {},
        'lockViewport': lock_viewport,
    }


def scatter_layer(points, layer_id='scatter', radius=28, color=None, opacity=0.65, pickable=True):
    return {
        'type': 'scatter',
        'id': layer_id,
        'data': points,
        'radius': radius,
        'color': color or [65, 150, 210, 180],
        'opacity': opacity,
        'pickable': pickable,
    }


def path_layer(paths, layer_id='paths'):
    """paths: [{path, color?, width?}]"""
    return {'type': 'path', 'id': layer_id, 'data': paths}


def column_layer(grid_data, max_count, period_name=''):
    return {
        'type': 'column',
        'id': 'density',
        'data': grid_data,
        'elevation_scale': 10000 / max_count if max_count > 0 else 1,
        'meta': {'period_name': period_name},
    }


def polygon_layer(polygons):
    return {'type': 'polygon', 'id': 'regions', 'data': polygons}


def arc_layer(arcs, layer_id='od-arcs'):
    """拱形 OD 连线: source/target 为 [lon, lat]"""
    return {'type': 'arc', 'id': layer_id, 'data': arcs}


def trajectory_paths_to_layers(trajectory_paths):
    paths = []
    for i, item in enumerate(trajectory_paths):
        paths.append({
            'path': item['path'],
            'taxi_id': item['taxi_id'],
            'color': TRAJECTORY_COLORS[i % len(TRAJECTORY_COLORS)],
            'width': 3,
        })
    layers = [path_layer(paths, 'trajectories')]
    if len(trajectory_paths) == 1:
        p = trajectory_paths[0]['path']
        layers.append(scatter_layer(
            [
                {'lon': p[0][0], 'lat': p[0][1], 'type': '起点', 'color': [50, 200, 50, 255]},
                {'lon': p[-1][0], 'lat': p[-1][1], 'type': '终点', 'color': [200, 50, 50, 255]},
            ],
            layer_id='markers',
            radius=80,
            color=[255, 255, 255, 255],
        ))
    return layers, '车辆ID: {taxi_id}' if len(trajectory_paths) > 1 else '{type}'


def view_state_for_coords(all_lons, all_lats, default_pitch=0):
    if not all_lons or not all_lats:
        return dict(DEFAULT_VIEW_STATE)
    center_lon = (min(all_lons) + max(all_lons)) / 2
    center_lat = (min(all_lats) + max(all_lats)) / 2
    span = max(max(all_lons) - min(all_lons), max(all_lats) - min(all_lats), 0.01)
    zoom = 11 if span > 0.15 else 12 if span > 0.06 else 13
    return {
        'longitude': center_lon,
        'latitude': center_lat,
        'zoom': zoom,
        'pitch': default_pitch,
        'bearing': 0,
    }
