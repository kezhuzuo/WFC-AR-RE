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
import random
import similarityMeasure


def mean_std(values):
    return np.mean(values), np.std(values)


# ======== Select UCI dataset ID ========
uci_dataset_id = 59  # Modify this ID to select a different dataset
N = 4  # Number of attribute groups

# ======== 1. Load UCI dataset ========
dataset = fetch_ucirepo(id=uci_dataset_id)

# **Automatically identify feature and target columns**
feature_data = pd.DataFrame(dataset.data.features)
target_data = pd.DataFrame(dataset.data.targets)

# **Drop all categorical features**
feature_data = feature_data.select_dtypes(include=[np.number])

# If all features are categorical, raise an error
if feature_data.shape[1] == 0:
    raise ValueError("All features in the dataset are categorical, unable to train!")

# If there are multiple target columns, select the first one as the classification target
target_col = target_data.columns[0]
feature_data[target_col] = target_data[target_col]

print(f"Loaded dataset: {dataset.metadata.name}")
print(
    f"Number of samples: {feature_data.shape[0]}, Number of features: {feature_data.shape[1] - 1}, Target column: {target_col}")

# **Check if the target variable is categorical**
if feature_data[target_col].dtype == "object":
    print("\nDetected categorical class labels, converting to numerical labels...")
    label_encoder = LabelEncoder()
    feature_data[target_col] = label_encoder.fit_transform(feature_data[target_col]) + 1  # Convert to 1,2,3,...

# Class distribution statistics
class_counts = feature_data[target_col].value_counts().sort_index()
print("\nOriginal class distribution:\n", class_counts)

# ======== 2. Handle class imbalance (ensure 5-fold cross-validation is feasible) ========
min_samples_per_class = 25  # 5-fold CV requires at least 5×5=25 samples per class
valid_classes = class_counts[class_counts >= min_samples_per_class].index.tolist()
invalid_classes = class_counts[class_counts < min_samples_per_class].index.tolist()

# **Merge rare classes**
mapping = {}
for cls in invalid_classes:
    # Find the closest major class
    closest_class = min(valid_classes, key=lambda x: abs(x - cls))
    mapping[cls] = closest_class  # Merge into the nearest major class

# **Update class labels**
feature_data[target_col] = feature_data[target_col].replace(mapping)

# **Recalculate the class distribution after merging**
class_counts = feature_data[target_col].value_counts().sort_index()
print("\nClass distribution after merging:\n", class_counts)

# ======== 3. Randomly split features into N groups ========
np.random.seed(42)
attributes = feature_data.columns[:-1].tolist()
np.random.shuffle(attributes)
groups = np.array_split(attributes, N)

# 4. Prepare features and target
X = feature_data[attributes]
y = feature_data[target_col]
unique_classes = np.sort(y.unique())

# 5. 5-fold cross-validation
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

fold_acc_ds = []
fold_acc_ar_ds = []
fold_acc_re_ds = []
fold_acc_zuo = []

fold_time_ds = []
fold_time_ar_ds = []
fold_time_re_ds = []
fold_time_zuo = []

# Randomly select K unique uppercase letters
K = class_counts.size
bayes_P = random.sample("ABCDEFGHIJKLMNOPQRSTUVWXYZ", K)
print(bayes_P)

for train_index, test_index in kf.split(X, y):
    # Split data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 5. Standardize data (each group separately)
    scalers = [StandardScaler() for _ in range(N)]
    X_train_groups = [scalers[i].fit_transform(X_train[groups[i]]) for i in range(N)]
    X_test_groups = [scalers[i].transform(X_test[groups[i]]) for i in range(N)]

    # 6. Train N SVM classifiers and calibrate probabilities
    svms = [SVC(probability=True, C=1.0) for _ in range(N)]
    # svms = [RandomForestClassifier() for _ in range(N)]
    # svms = [MLPClassifier() for _ in range(N)]
    calibrated_svms = [CalibratedClassifierCV(svms[i], cv=5) for i in range(N)]

    for i in range(N):
        calibrated_svms[i].fit(X_train_groups[i], y_train)

    # 8. Compute fused probability distributions and record fusion time
    t_ds = []
    t_ar_ds = []
    fusion_ds = []
    fusion_ar_ds = []

    for i in range(X_test.shape[0]):
        # Get probability distributions from N SVMs
        fine_boe = np.array([calibrated_svms[j].predict_proba(X_test_groups[j][i].reshape(1, -1)) for j in range(N)])

        # -------------------- DS direct fusion -------------------
        start_ori = time.time()
        ds = fusionRules.multiSourceFusion(fine_boe, bayes_P)
        end_ori = time.time()
        t_ds.append((end_ori - start_ori) * 1000)
        ds_pre = unique_classes[np.argmax(ds)]  # Select the class with the highest probability
        fusion_ds.append(ds_pre)

        # ------------------- DS + AR ---------------------
        start_ar = time.time()
        bba, p1 = AR.BBA_approximation(fine_boe, bayes_P)
        ar_ds = fusionRules.multiSourceFusion(bba, p1)
        end_ar = time.time()
        t_ar_ds.append((end_ar - start_ar) * 1000)

        bet = fusionRules.BetP(ar_ds, p1)
        id = np.argmax(bet)
        ar_ds_pre = bayes_P.index(p1[id])  # BetP is a list, not numpy, so it's an index
        _pre = unique_classes[ar_ds_pre]
        fusion_ar_ds.append(_pre)

    # 9. Compute classification accuracy after fusion
    acc_ds = accuracy_score(y_test, fusion_ds) * 100
    fold_acc_ds.append(acc_ds)
    fold_time_ds.extend(t_ds)

    acc_ar_ds = accuracy_score(y_test, fusion_ar_ds) * 100
    fold_acc_ar_ds.append(acc_ar_ds)
    fold_time_ar_ds.extend(t_ar_ds)

# 10. Compute mean and standard deviation
acc_mean_ds = mean_std(fold_acc_ds)
time_mean_ds = mean_std(fold_time_ds)

acc_mean_ar_ds = mean_std(fold_acc_ar_ds)
time_mean_ar_ds = mean_std(fold_time_ar_ds)

# 11. Print final results
print("DS:", acc_mean_ds, time_mean_ds)
print(f"{acc_mean_ds[0]:.2f} ± {acc_mean_ds[1]:.2f}", "&", f"{time_mean_ds[0]:.3f} ± {time_mean_ds[1] / 10:.3f}")
print(f"{acc_mean_ar_ds[0]:.2f} ± {acc_mean_ar_ds[1]:.2f}", "&",
      f"{time_mean_ar_ds[0]:.3f} ± {time_mean_ar_ds[1] / 10:.3f}")
