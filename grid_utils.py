"""
网格工具函数 - 用于 OD 矩阵的区域划分
"""

LON_MIN = 116.0
LON_MAX = 117.0
LAT_MIN = 39.4
LAT_MAX = 40.1

GRID_SIZE = 100

LON_STEP = (LON_MAX - LON_MIN) / GRID_SIZE
LAT_STEP = (LAT_MAX - LAT_MIN) / GRID_SIZE


class TimePeriod:
    """时段定义"""
    MORNING_PEAK = 0
    EVENING_PEAK = 1
    OFF_PEAK = 2
    
    @classmethod
    def get_period(cls, timestamp):
        """支持 pandas Timestamp、datetime、numpy.datetime64、字符串"""
        from datetime import datetime
        import pandas as pd

        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        if not hasattr(timestamp, 'hour'):
            timestamp = pd.Timestamp(timestamp)
        hour = timestamp.hour
        if 7 <= hour < 9:
            return cls.MORNING_PEAK
        elif 17 <= hour < 19:
            return cls.EVENING_PEAK
        else:
            return cls.OFF_PEAK
    
    @classmethod
    def get_period_name(cls, period):
        names = {0: "早高峰", 1: "晚高峰", 2: "平峰"}
        return names.get(period, "未知")
    
    @classmethod
    def get_all_periods(cls):
        return [cls.MORNING_PEAK, cls.EVENING_PEAK, cls.OFF_PEAK]


def lonlat_to_grid_id(lon, lat):
    if lon < LON_MIN or lon >= LON_MAX or lat < LAT_MIN or lat >= LAT_MAX:
        return -1
    x = int((lon - LON_MIN) / LON_STEP)
    y = int((lat - LAT_MIN) / LAT_STEP)
    x = min(x, GRID_SIZE - 1)
    y = min(y, GRID_SIZE - 1)
    return y * GRID_SIZE + x


def grid_id_to_bounds(grid_id):
    if grid_id < 0 or grid_id >= GRID_SIZE * GRID_SIZE:
        return None
    y = grid_id // GRID_SIZE
    x = grid_id % GRID_SIZE
    x_min = LON_MIN + x * LON_STEP
    x_max = x_min + LON_STEP
    y_min = LAT_MIN + y * LAT_STEP
    y_max = y_min + LAT_STEP
    return (x_min, y_min, x_max, y_max)


def rect_to_grid_ids(rect):
    x_min, y_min, x_max, y_max = rect
    start_x = int((x_min - LON_MIN) / LON_STEP)
    end_x = int((x_max - LON_MIN) / LON_STEP)
    start_y = int((y_min - LAT_MIN) / LAT_STEP)
    end_y = int((y_max - LAT_MIN) / LAT_STEP)
    
    start_x = max(0, min(start_x, GRID_SIZE - 1))
    end_x = max(0, min(end_x, GRID_SIZE - 1))
    start_y = max(0, min(start_y, GRID_SIZE - 1))
    end_y = max(0, min(end_y, GRID_SIZE - 1))
    
    grid_ids = []
    for y in range(start_y, end_y + 1):
        for x in range(start_x, end_x + 1):
            grid_ids.append(y * GRID_SIZE + x)
    return grid_ids