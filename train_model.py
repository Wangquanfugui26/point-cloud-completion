import numpy as np
import laspy
import os
import pickle
import optuna
from sklearn.neighbors import KDTree
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score, mean_squared_error

# ======================
# 关闭所有警告（干净运行）
# ======================
import warnings

warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore', over='ignore')

# ======================
# 训练配置
# ======================
TRAIN_LAS = "Ground points.las"
MODEL_SAVE_DIR = "models"
N_TRIALS = 20
K_NEIGHBORS = 20
SEED = 42

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ======================
# 1. 读取点云
# ======================
print("✅ 1/5 读取训练点云...")
las = laspy.read(TRAIN_LAS)
points = np.vstack((las.x, las.y, las.z)).T
print(f"   点云数量：{len(points):,}")

# ======================
# 2. 构建训练数据集
# ======================
print("\n✅ 2/5 构建训练特征...")
print("   构建 KDTree...")
tree = KDTree(points[:, :2])

print("   查询近邻点...")
dists, indices = tree.query(points[:, :2], k=K_NEIGHBORS)

neigh_z = points[indices, 2]
neigh_x = points[indices, 0]
neigh_y = points[indices, 1]

print("   计算高程特征...")
z_mean = neigh_z.mean(axis=1)
z_std = neigh_z.std(axis=1)
z_min = neigh_z.min(axis=1)
z_max = neigh_z.max(axis=1)
d_mean = dists.mean(axis=1)
d_std = dists.std(axis=1)


# 坡度坡向
def compute_slope_aspect(x, y, z):
    x += np.random.normal(0, 1e-6, size=x.shape)
    y += np.random.normal(0, 1e-6, size=y.shape)
    try:
        dx = np.gradient(z, x, axis=0)
        dy = np.gradient(z, y, axis=0)
    except:
        return 0.0, 0.0
    dx = np.nan_to_num(dx)
    dy = np.nan_to_num(dy)
    slope = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    aspect = np.arctan2(dy, dx)
    return np.nanmean(slope), np.nanmean(aspect)


print("   计算坡度坡向（可能稍慢，耐心等）...")
slopes = []
aspects = []
total = len(points)

for i in range(total):
    # 每 10% 打印一次进度
    if i % (total // 10) == 0:
        print(f"   坡度坡向进度：{i / total * 100:.0f}%")

    s, a = compute_slope_aspect(neigh_x[i], neigh_y[i], neigh_z[i])
    slopes.append(s)
    aspects.append(a)

print("   构建特征矩阵...")
X = np.column_stack([
    z_mean, z_std, z_min, z_max,
    d_mean, d_std,
    slopes, aspects,
    points[:, 0], points[:, 1]
])
y = points[:, 2]

mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
X = X[mask]
y = y[mask]

print(f"   训练样本数：{len(X):,}")

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=SEED)


# ======================
# 3. Optuna 自动调参
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
    }
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)


print("\n✅ 3/5 开始 Optuna 自动调参（共20轮）...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS)
best_params = study.best_params
print("   最优参数：", best_params)

# ======================
# 4. 训练最终模型
# ======================
print("\n✅ 4/5 训练最优 CatBoost 模型...")
best_model = CatBoostRegressor(**best_params, verbose=100)
best_model.fit(X_train, y_train)

# 评估
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"   R2:  {r2:.4f}")
print(f"   RMSE:{rmse:.4f}")

# ======================
# 5. 保存模型
# ======================
print("\n✅ 5/5 保存模型...")
model_path = os.path.join(MODEL_SAVE_DIR, "model_trial_1.pkl")
with open(model_path, "wb") as f:
    pickle.dump({"model": best_model}, f)

print(f"\n🎉 训练全部完成！模型已保存：{model_path}")
print("👉 现在可以直接运行点云补全程序！")