import numpy as np
import laspy
from sklearn.neighbors import KDTree
import pickle
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import warnings

# -------------------------------
# 配置（ Poisson 最强效果 ）
# -------------------------------
INPUT_LAS = "Ground points.las"
OUTPUT_LAS = "Ground points completed_poisson.las"
SELECTED_TRIAL_ID = 1
MODEL_DIR = "models"
MAX_HOLE_SIZE = 80.0
VISUALIZE = True
MAX_FILL_POINTS = 250000
CPU_WORKERS = 3

# Poisson 核心（最自然）
POISSON_RADIUS = 2.8

warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore', over='ignore')


# -------------------------------
# ✅ 最强 Poisson 采样（保留！）
# -------------------------------
def accelerated_poisson(x_min, x_max, y_min, y_max, radius):
    xs = np.arange(x_min, x_max, radius * 0.4)
    ys = np.arange(y_min, y_max, radius * 0.4)
    X, Y = np.meshgrid(xs, ys)
    candidates = np.column_stack([X.ravel(), Y.ravel()])
    np.random.shuffle(candidates)

    grid = {}
    cell_size = radius
    points = []
    s = radius / np.sqrt(2)

    for (x, y) in candidates:
        gx = int(x // cell_size)
        gy = int(y // cell_size)
        ok = True

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (gx + dx, gy + dy)
                if key in grid:
                    px, py = grid[key]
                    d = (x - px) ** 2 + (y - py) ** 2
                    if d < s * s:
                        ok = False
                        break
            if not ok:
                break
        if ok:
            grid[(gx, gy)] = (x, y)
            points.append([x, y])
            if len(points) > MAX_FILL_POINTS:
                break

    return np.array(points)


# -------------------------------
# 距离掩码（和你能用的版本完全一致）
# -------------------------------
def create_distance_mask(points, max_hole_size=30.0, resolution=1.0):
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    nx = int(np.ceil((x_max - x_min) / resolution))
    ny = int(np.ceil((y_max - y_min) / resolution))
    x_centers = x_min + (np.arange(nx) + 0.5) * resolution
    y_centers = y_min + (np.arange(ny) + 0.5) * resolution
    X, Y = np.meshgrid(x_centers, y_centers)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    tree = KDTree(points[:, :2])
    dists, _ = tree.query(grid_points, k=1)
    dist_map = dists.reshape(ny, nx)
    mask = dist_map <= max_hole_size
    return mask, (x_min, x_max, y_min, y_max), resolution


# -------------------------------
# 坡度坡向
# -------------------------------
def compute_slope_aspect(x, y, z):
    x = x + np.random.normal(0, 1e-6, x.shape)
    y = y + np.random.normal(0, 1e-6, y.shape)
    try:
        dx = np.gradient(z, x, axis=0)
        dy = np.gradient(z, y, axis=0)
    except:
        return 0.0, 0.0

    dx = np.nan_to_num(dx)
    dy = np.nan_to_num(dy)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(dy, dx)
    return np.nanmean(slope), np.nanmean(aspect)


# -------------------------------
# 坐标偏移缩放
# -------------------------------
def compute_optimal_offset_scale(points, dtype=np.int32):
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()

    offset_x = (x_min + x_max) / 2
    offset_y = (y_min + y_max) / 2
    offset_z = (z_min + z_max) / 2

    max_int = np.iinfo(dtype).max
    scale = max((x_max-x_min), (y_max-y_min), (z_max-z_min)) / (max_int * 0.9)
    scale = max(scale, 1e-6)
    return (offset_x, offset_y, offset_z), (scale, scale, scale)


# -------------------------------
# 主程序（100% 修复）
# -------------------------------
def main():
    print("✅ 1/7 读取点云...")
    las = laspy.read(INPUT_LAS)
    points = np.vstack((las.x, las.y, las.z)).T
    print(f"   点云数量：{len(points):,}")

    print("\n✅ 2/7 加载模型...")
    model_path = os.path.join(MODEL_DIR, f"model_trial_{SELECTED_TRIAL_ID}.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)["model"]

    print("\n✅ 3/7 估算点间距...")
    sample_n = min(10000, len(points))
    idx = np.random.choice(len(points), sample_n, replace=False)
    dists, _ = KDTree(points[idx,:2]).query(points[idx,:2], k=2)
    avg_dist = np.mean(dists[:,1])
    print(f"   平均点间距：{avg_dist:.2f} m")

    # -------------------------------
    # 🔥 最强 Poisson 采样
    # -------------------------------
    print("\n✅ 4/7 Poisson 采样...")
    x_min, x_max = points[:,0].min(), points[:,0].max()
    y_min, y_max = points[:,1].min(), points[:,1].max()
    hole_xy = accelerated_poisson(x_min, x_max, y_min, y_max, POISSON_RADIUS)
    print(f"   候选补点：{len(hole_xy):,}")

    # 限制最大点数
    if len(hole_xy) > MAX_FILL_POINTS:
        hole_xy = hole_xy[np.random.choice(len(hole_xy), MAX_FILL_POINTS, replace=False)]

    print("\n✅ 5/7 距离场过滤...")
    mask, bounds, res = create_distance_mask(points, MAX_HOLE_SIZE)
    xmn, xmx, ymn, ymx = bounds

    in_bounds = (
        (hole_xy[:,0]>=xmn) & (hole_xy[:,0]<=xmx) &
        (hole_xy[:,1]>=ymn) & (hole_xy[:,1]<=ymx)
    )

    # -------------------------
    # ✅ 【终极修复】这里完全对齐你能用的代码！
    # -------------------------
    ix = np.clip(((hole_xy[:,0] - xmn) / res).astype(int), 0, mask.shape[1]-1)
    iy = np.clip(((hole_xy[:,1] - ymn) / res).astype(int), 0, mask.shape[0]-1)
    hole_xy = hole_xy[in_bounds & mask[iy, ix]]

    print(f"   有效补点：{len(hole_xy):,}")

    # 预览
    if VISUALIZE and len(hole_xy):
        print("\n✅ 生成预览图...")
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        s_idx = np.random.choice(len(points), min(50000, len(points)), replace=False)
        plt.figure(figsize=(12,8))
        plt.scatter(points[s_idx,0], points[s_idx,1], s=0.1, c='lightgray')
        plt.scatter(hole_xy[:,0], hole_xy[:,1], s=0.4, c='red')
        plt.title(f'Poisson 补点预览  数量={len(hole_xy)}')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('preview_holes.png', dpi=150)
        plt.close()

    if len(hole_xy) == 0:
        print("\n⚠️ 无空洞可补")
        new_points = points
    else:
        print("\n✅ 6/7 特征计算...")
        tree = KDTree(points[:,:2])
        dists, idx = tree.query(hole_xy, k=min(20, len(points)))

        nz = points[idx, 2]
        nx = points[idx, 0]
        ny = points[idx, 1]

        z_mean = nz.mean(1)
        z_std  = nz.std(1)
        z_min  = nz.min(1)
        z_max  = nz.max(1)
        d_mean = dists.mean(1)
        d_std  = dists.std(1)

        def work(i):
            return compute_slope_aspect(nx[i], ny[i], nz[i])

        print(f"\n✅ 坡度坡向（{CPU_WORKERS}核）")
        with ThreadPoolExecutor(CPU_WORKERS) as pool:
            res = list(tqdm(pool.map(work, range(len(hole_xy))), total=len(hole_xy)))

        slopes, aspects = zip(*res)
        feats = np.column_stack([
            z_mean, z_std, z_min, z_max,
            d_mean, d_std, slopes, aspects,
            hole_xy[:,0], hole_xy[:,1]
        ])
        feats = np.nan_to_num(feats)

        print("\n✅ 7/7 预测高程...")
        z_pred = model.predict(feats)
        z_pred = np.nan_to_num(z_pred, nan=points[:,2].mean())
        new_points = np.vstack([points, np.column_stack([hole_xy, z_pred])])

    print("\n💾 保存 LAS...")
    off, sc = compute_optimal_offset_scale(new_points)
    header = las.header.copy()
    header.offsets = off
    header.scales = sc

    out = laspy.LasData(header)
    out.x = new_points[:,0]
    out.y = new_points[:,1]
    out.z = new_points[:,2]

    for dim in las.point_format.dimension_names:
        if dim in ['X','Y','Z']: continue
        v = getattr(las, dim)
        pad = np.full(len(new_points)-len(v), 2 if 'class' in dim.lower() else 0, dtype=v.dtype)
        setattr(out, dim, np.hstack([v, pad]))

    out.write(OUTPUT_LAS)

    print("\n🎉 Poisson 完美补洞完成！")
    print(f"原始：{len(points)}  补点：{len(new_points)-len(points)}")


if __name__ == "__main__":
    main()