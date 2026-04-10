import numpy as np
import laspy
import os
import pickle
import optuna
from sklearn.neighbors import KDTree
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error
from concurrent.futures import ThreadPoolExecutor
import warnings
from tqdm import tqdm

# ======================
# 关闭警告
# ======================
warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore', over='ignore')

# ======================
# 🎯 核心配置（所有电脑都在这里改）
# ======================
TRAIN_LAS = "Ground points.las"
MODEL_SAVE_DIR = "models"
N_TRIALS = 20
K_NEIGHBORS = 20
SEED = 42

# 🔥 在这里设置你想使用的 CPU 核心数！
# 所有电脑通用：填 1~你最大核心数 之间
CPU_WORKERS = 3  # 想改几核就改几

# ======================
# 自动创建目录
# ======================
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ======================
# 1. 读取点云
# ======================
print("✅ 1/5 读取点云...")
las = laspy.read(TRAIN_LAS)
points = np.vstack((las.x, las.y, las.z)).T
print(f"   点云数量：{len(points):,}")

# ======================
# 2. 构建特征
# ======================
print("\n✅ 2/5 构建特征...")
tree = KDTree(points[:, :2])
dists, indices = tree.query(points[:, :2], k=K_NEIGHBORS)

neigh_z = points[indices, 2]
neigh_x = points[indices, 0]
neigh_y = points[indices, 1]

z_mean = neigh_z.mean(axis=1)
z_std = neigh_z.std(axis=1)
z_min = neigh_z.min(axis=1)
z_max = neigh_z.max(axis=1)
d_mean = dists.mean(axis=1)
d_std = dists.std(axis=1)

# ======================
# 3. 坡度坡向计算
# ======================
def compute_single(i):
    x = neigh_x[i]
    y = neigh_y[i]
    z = neigh_z[i]
    x += np.random.normal(0, 1e-6, size=x.shape)
    y += np.random.normal(0, 1e-6, size=y.shape)
    try:
        dx = np.gradient(z, x, axis=0)
        dy = np.gradient(z, y, axis=0)
    except:
        return 0.0, 0.0
    dx = np.nan_to_num(dx)
    dy = np.nan_to_num(dy)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(dy, dx)
    return np.nanmean(slope), np.nanmean(aspect)

print(f"\n✅ 3/5 计算坡度坡向 | 使用CPU核心数：{CPU_WORKERS}")
with ThreadPoolExecutor(max_workers=CPU_WORKERS) as executor:
    results = list(tqdm(executor.map(compute_single, range(len(points))), total=len(points)))

slopes, aspects = zip(*results)
slopes = np.array(slopes)
aspects = np.array(aspects)

# ======================
# 4. 构建训练集
# ======================
print("\n✅ 4/5 构建训练集...")
X = np.column_stack([
    z_mean, z_std, z_min, z_max,
    d_mean, d_std,
    slopes, aspects,
    points[:,0], points[:,1]
])
y = points[:, 2]

mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
X = X[mask]
y = y[mask]
print(f"   训练样本：{len(X):,}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=SEED)

# ======================
# 5. Optuna 调参 + 训练
# ======================
def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 200, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "depth": trial.suggest_int("depth", 3, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "random_seed": SEED,
        "verbose": 0,
        "thread_count": CPU_WORKERS,  # 模型也用同样核心数
    }
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train)
    return mean_squared_error(y_test, model.predict(X_test))

print(f"\n✅ 5/5 Optuna 调参与训练 | 使用CPU核心数：{CPU_WORKERS}")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS)

best_model = CatBoostRegressor(**study.best_params, verbose=100, thread_count=CPU_WORKERS)
best_model.fit(X_train, y_train)

# 评估
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\n🎉 R2: {r2:.4f}   RMSE: {rmse:.4f}")

# 保存模型
model_path = os.path.join(MODEL_SAVE_DIR, "model_trial_cpu_1.pkl")
with open(model_path, "wb") as f:
    pickle.dump({"model": best_model}, f)

print(f"\n✅ 模型保存完成：{model_path}")