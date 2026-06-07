"""
MIDI 解析与旋律曲线提取
"""
import os
import numpy as np
import music21 as m21


def parse_midi(path, melody_only=True):
    """读取一个 MIDI 文件，提取音符的 (时间, 音高, 音量) 三元组

    Args:
        path: .mid 文件路径
        melody_only: 如果为 True，同一时间戳只保留最高音（提取主旋律线）

    Returns:
        (n, 3) numpy 数组，每行为 (t, pitch, velocity)，失败返回 None
    """
    try:
        score = m21.converter.parse(path)
    except Exception:
        return None

    notes = []
    try:
        for part in score.parts:
            for n in part.flatten().notesAndRests:
                if n.isNote:
                    t = float(n.offset)
                    pitch = n.pitch.midi
                    vel = n.volume.velocity if n.volume.velocity else 64
                    notes.append([t, pitch, vel])
    except Exception:
        return None

    if len(notes) < 5:  # 过滤音符太少的数据
        return None

    arr = np.array(notes, dtype=np.float64)

    if melody_only:
        # 同一时间戳只保留最高音（主旋律通常在最上面）
        arr = extract_melody(arr)

    return arr


def extract_melody(arr):
    """从多声部点集中提取主旋律线：同一时间戳只保留最高音

    Args:
        arr: (n, 3) 数组，每行 (t, pitch, velocity)

    Returns:
        (m, 3) 数组，m <= n
    """
    # 按时间排序
    arr = arr[arr[:, 0].argsort()]
    # 对每个时间戳，只保留 pitch 最大的那个
    unique_t, indices = np.unique(arr[:, 0], return_index=True)
    # 对每个唯一时间，找 pitch 最大的行
    melody = []
    i = 0
    n = len(arr)
    while i < n:
        t = arr[i, 0]
        # 找出所有同一时间的点
        j = i
        while j < n and arr[j, 0] == t:
            j += 1
        # 取 pitch 最大的那个
        best = arr[i:j][arr[i:j, 1].argmax()]
        melody.append(best)
        i = j
    return np.array(melody, dtype=np.float64)


def resample_curve(points, n_samples=200):
    """沿旋律曲线均匀重采样 n_samples 个点

    先按时间排序，连接相邻点形成分段线性曲线，然后沿弧长均匀采样。

    Args:
        points: (n, 3) 数组 (t, pitch, vel)
        n_samples: 采样点数

    Returns:
        (n_samples, 3) 数组
    """
    # 按时间排序
    points = points[points[:, 0].argsort()]
    if len(points) < 2:
        return points

    # 计算每段的长度（弧长）
    diffs = np.diff(points, axis=0)
    seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    total_length = seg_lengths.sum()

    if total_length == 0:
        return np.tile(points[0], (n_samples, 1))

    # 计算累积弧长
    cum_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
    # 均匀采样点位置（弧长比例）
    target_lengths = np.linspace(0, total_length, n_samples)

    # 插值
    result = []
    for t_len in target_lengths:
        # 找到在哪一段
        idx = np.searchsorted(cum_lengths, t_len) - 1
        idx = max(0, min(idx, len(points) - 2))
        # 该段内的比例
        seg_start = cum_lengths[idx]
        seg_end = cum_lengths[idx + 1]
        if seg_end > seg_start:
            ratio = (t_len - seg_start) / (seg_end - seg_start)
        else:
            ratio = 0
        # 线性插值
        interpolated = points[idx] + ratio * (points[idx + 1] - points[idx])
        result.append(interpolated)

    return np.array(result, dtype=np.float64)


def min_max_normalize(points):
    """Min-Max 归一化每个维度到 [0,1]

    Args:
        points: (n, 3) 数组

    Returns:
        (n, 3) 归一化后的数组
    """
    normalized = np.zeros_like(points)
    for i in range(3):
        col = points[:, i]
        min_val, max_val = col.min(), col.max()
        if max_val > min_val:
            normalized[:, i] = (col - min_val) / (max_val - min_val)
        else:
            normalized[:, i] = 0.5
    return normalized


def load_adl_data(base_path, genres, max_per_genre=50, normalize=True, normalize_mode='minmax', resample=True, n_samples=200):
    """批量加载 ADL Piano MIDI 数据集

    Args:
        base_path: adl-piano-midi 文件夹路径
        genres: 要加载的曲风列表
        max_per_genre: 每种曲风最多加载多少首
        normalize: 是否做归一化
        normalize_mode: 'minmax' 或 'time_only'（仅时间归一化，音高和音量保留原始值）
        resample: 是否沿弧长重采样到固定点数（捕获曲线形状）
        n_samples: 重采样点数

    Returns:
        points_list: list of (n_i, 3) arrays
        labels: list of genre strings
    """
    points_list = []
    labels = []

    for genre in genres:
        genre_path = os.path.join(base_path, genre)
        if not os.path.isdir(genre_path):
            print(f"  [警告] {genre} 文件夹不存在，跳过")
            continue

        files = sorted([
            f for f in os.listdir(genre_path) if f.endswith('.mid')
        ])[:max_per_genre]

        count = 0
        for fname in files:
            path = os.path.join(genre_path, fname)
            points = parse_midi(path, melody_only=True)
            if points is None:
                continue

            # 先重采样（捕获曲线形状），再归一化
            if resample:
                points = resample_curve(points, n_samples)

            if normalize and normalize_mode == 'minmax':
                points = min_max_normalize(points)
            elif normalize and normalize_mode == 'time_only':
                norm = np.zeros_like(points)
                t_col = points[:, 0]
                if t_col.max() > t_col.min():
                    norm[:, 0] = (t_col - t_col.min()) / (t_col.max() - t_col.min())
                else:
                    norm[:, 0] = 0.5
                norm[:, 1] = points[:, 1]
                norm[:, 2] = points[:, 2]
                points = norm

            points_list.append(points)
            labels.append(genre)
            count += 1

        print(f"  {genre}: {count}/{len(files)} 首成功加载")

    return points_list, labels
