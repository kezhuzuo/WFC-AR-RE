import numpy as np
import pandas as pd
import time
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import fusionRules
import AR
import SOTA
import SPOTIS
import random
import similarityMeasure


def mean_std(values):
    return np.mean(values), np.std(values)


# ======== 选择 UCI 数据集 ID ========
uci_dataset_id = 59  # 修改此 ID 来选择不同的数据集
N = 4  # 属性划分的组数

# ======== 1. 加载 UCI 数据集 ========
dataset = fetch_ucirepo(id=uci_dataset_id)

# **自动识别特征和目标列**
feature_data = pd.DataFrame(dataset.data.features)
target_data = pd.DataFrame(dataset.data.targets)

# **丢弃所有字符型特征**
feature_data = feature_data.select_dtypes(include=[np.number])

# 如果数据集所有特征都是字符型，抛出错误
if feature_data.shape[1] == 0:
    raise ValueError("数据集中所有特征均为字符型，无法进行训练！")

# 如果有多个目标列，选择第一个作为分类目标
target_col = target_data.columns[0]
feature_data[target_col] = target_data[target_col]

print(f"已加载数据集: {dataset.metadata.name}")
print(f"样本数: {feature_data.shape[0]}, 特征数: {feature_data.shape[1] - 1}, 目标列: {target_col}")

# **检查目标变量是否是字符串类型**
if feature_data[target_col].dtype == "object":
    print("\n检测到字符型类别标签，正在转换为数值标签...")
    label_encoder = LabelEncoder()
    feature_data[target_col] = label_encoder.fit_transform(feature_data[target_col]) + 1  # 转换为 1,2,3,...

# 统计类别分布
class_counts = feature_data[target_col].value_counts().sort_index()
print("\n原始类别分布：\n", class_counts)

# ======== 2. 处理类别不均衡（确保 5 折交叉验证可行） ========
min_samples_per_class = 25  # 5 折交叉验证至少需要 5×5=25 个样本
valid_classes = class_counts[class_counts >= min_samples_per_class].index.tolist()
invalid_classes = class_counts[class_counts < min_samples_per_class].index.tolist()

# **合并稀有类别**
mapping = {}
for cls in invalid_classes:
    # 找到最近的主要类别
    closest_class = min(valid_classes, key=lambda x: abs(x - cls))
    mapping[cls] = closest_class  # 合并到最近的主要类别

# **更新类别标签**
feature_data[target_col] = feature_data[target_col].replace(mapping)

# **重新统计合并后的类别分布**
class_counts = feature_data[target_col].value_counts().sort_index()
print("\n合并后类别分布：\n", class_counts)

# ======== 3. 随机划分特征为N组 ========
np.random.seed(42)
attributes = feature_data.columns[:-1].tolist()
np.random.shuffle(attributes)
groups = np.array_split(attributes, N)

# 4. 准备特征和目标
X = feature_data[attributes]
y = feature_data[target_col]
unique_classes = np.sort(y.unique())

# 5. 5 折交叉验证
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

fold_acc_ds = []
fold_acc_ar_ds = []
fold_acc_re_ds = []
fold_acc_zuo = []

fold_time_ds = []
fold_time_ar_ds = []
fold_time_re_ds = []
fold_time_zuo = []

# 随机选择 K 个不重复的大写字母
K = class_counts.size
bayes_P = random.sample("ABCDEFGHIJKLMNOPQRSTUVWXYZ", K)
print(bayes_P)

for train_index, test_index in kf.split(X, y):
    # 划分数据
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 5. 标准化数据（每个分组分别标准化）
    scalers = [StandardScaler() for _ in range(N)]
    X_train_groups = [scalers[i].fit_transform(X_train[groups[i]]) for i in range(N)]
    X_test_groups = [scalers[i].transform(X_test[groups[i]]) for i in range(N)]

    # 6. 训练 N 个 SVM 分类器，并进行概率校准
    svms = [SVC(probability=True, C=1.0) for _ in range(N)]
    # svms = [RandomForestClassifier() for _ in range(N)]
    # svms = [MLPClassifier() for _ in range(N)]
    calibrated_svms = [CalibratedClassifierCV(svms[i], cv=5) for i in range(N)]

    for i in range(N):
        calibrated_svms[i].fit(X_train_groups[i], y_train)

    # 8. 计算融合后的概率分布，并记录融合时间
    t_ds = []
    t_ar_ds = []
    t_re_ds = []
    t_zuo = []
    fusion_ds = []
    fusion_ar_ds = []
    fusion_re_ds = []
    fusion_zuo = []

    for i in range(X_test.shape[0]):
        # 获取 3 个 SVM 的概率分布
        fine_boe = np.array([calibrated_svms[j].predict_proba(X_test_groups[j][i].reshape(1, -1)) for j in range(N)])

        # # --------------------DS直接融合-------------------
        start_ori = time.time()
        ds = fusionRules.multiSourceFusion(fine_boe, bayes_P)
        end_ori = time.time()
        t_ds.append((end_ori - start_ori) * 1000)
        ds_pre = unique_classes[np.argmax(ds)]  # 选取最大概率对应的类别
        fusion_ds.append(ds_pre)

        # -------------------DS + AR---------------------
        start_ar = time.time()
        bba, p1 = AR.BBA_approximation(fine_boe, bayes_P)
        ar_ds = fusionRules.multiSourceFusion(bba, p1)
        end_ar = time.time()
        t_ar_ds.append((end_ar - start_ar) * 1000)

        bet = fusionRules.BetP(ar_ds, p1)
        id = np.argmax(bet)
        ar_ds_pre = bayes_P.index(p1[id])  # 注意这里的BeP是例表不是numpy，所示是idx不是idx[0]
        _pre = unique_classes[ar_ds_pre]
        fusion_ar_ds.append(_pre)

        # -------------------DS + RE---------------------
        start_re = time.time()
        wc = [1] * N
        re_ds = SOTA.zuo(fine_boe, bayes_P, wc, N)
        end_re = time.time()
        t_re_ds.append((end_re - start_re) * 1000)

        re_ds_pre = unique_classes[np.argmax(re_ds)]  # 选取最大概率对应的类别
        fusion_re_ds.append(re_ds_pre)

    # 9. 计算融合后的分类精度
    acc_ds = accuracy_score(y_test, fusion_ds) * 100
    fold_acc_ds.append(acc_ds)
    fold_time_ds.extend(t_ds)  # 记录所有样本的融合时间

    acc_ar_ds = accuracy_score(y_test, fusion_ar_ds) * 100
    fold_acc_ar_ds.append(acc_ar_ds)
    fold_time_ar_ds.extend(t_ar_ds)  # 记录所有样本的融合时间

    acc_re_ds = accuracy_score(y_test, fusion_re_ds) * 100
    fold_acc_re_ds.append(acc_re_ds)
    fold_time_re_ds.extend(t_re_ds)  # 记录所有样本的融合时间

# 10. 计算均值和标准差
acc_mean_ds = mean_std(fold_acc_ds)
time_mean_ds = mean_std(fold_time_ds)

acc_mean_ar_ds = mean_std(fold_acc_ar_ds)
time_mean_ar_ds = mean_std(fold_time_ar_ds)

acc_mean_re_ds = mean_std(fold_acc_re_ds)
time_mean_re_ds = mean_std(fold_time_re_ds)

# 11. 打印最终结果
print("DS:", acc_mean_ds, time_mean_ds)
print(f"{acc_mean_ds[0]:.2f} ± {acc_mean_ds[1]:.2f}", "&", f"{time_mean_ds[0]:.3f} ± {time_mean_ds[1] / 10:.3f}")
print(f"{acc_mean_ar_ds[0]:.2f} ± {acc_mean_ar_ds[1]:.2f}", "&",
      f"{time_mean_ar_ds[0]:.3f} ± {time_mean_ar_ds[1] / 10:.3f}")
print(f"{acc_mean_re_ds[0]:.2f} ± {acc_mean_re_ds[1]:.2f}", "&",
      f"{time_mean_re_ds[0]:.3f} ± {time_mean_re_ds[1] / 10:.3f}")
