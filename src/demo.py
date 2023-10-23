import pandas as pd
import copy
df = pd.read_csv('../data/car.data', names = ["buying", "maintenance", "doors", "persons", "lug_boot", "safety", "classification"])

# buying ->      0: v-high,   1: high,   2: med,    3: low
# maintenance -> 0: v-high,   1: high,   2: med,    3: low
# doors ->       0: 2,        1: 3,      2: 4,      3: 5 - more
# persons ->     0: 2,        1: 4,      2: more
# lug_boot ->    0: small,    1: med,    2: big
# safety ->      0: low,      1: med,    2: high
#classification  0:unacc,     1: acc,    2: vgood,  3:good,

for column in df.columns:
    if column != 'classification':
        df[column], _ = pd.factorize(df[column])

#classification  0:unacc,     1: acc,    2: good,  3:vgood,

cfc = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
df['classification'] = df['classification'].map(cfc)

def k_fold(target, k):
    # Số lượng mẫu trong target
    total_samples = len(target)

    # Kích thước của mỗi fold
    fold_size = total_samples // k

    # Danh sách chứa các fold
    folds = []

    # Tạo ra các fold
    for i in range(k):
        start = i * fold_size
        # Nếu đây là fold cuối cùng, kết thúc ở cuối tập dữ liệu
        if i == k - 1:
            end = total_samples
        else:
            end = start + fold_size
        # Thêm fold vào danh sách folds
        folds.append((start, end))

    # Danh sách chứa các tập huấn luyện và kiểm tra
    train_test_folds = []

    for i in range(k):
        # Fold hiện tại sẽ là tập kiểm tra
        test_start, test_end = folds[i]

        # Các fold còn lại sẽ là tập huấn luyện
        train_indices = list(range(0, test_start)) + list(range(test_end, total_samples))

        # Thêm vào danh sách train_test_folds
        train_test_folds.append((train_indices, list(range(test_start, test_end))))

    return train_test_folds

# Sử dụng hàm vừa viết
# target = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# k = 5
# folds = k_fold(target, k)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def simple_cross_validation(model, features, target, k):
    # Khởi tạo danh sách để lưu trữ các chỉ số đánh giá
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    # Tạo các fold sử dụng hàm k_fold
    folds = k_fold(target, k)

    # Lặp qua từng fold
    for train_indices, test_indices in folds:
        # Tạo một bản sao của mô hình
        classifier = copy.copy(model)

        # Tạo tập huấn luyện và tập kiểm tra
        X_train, X_test = features.iloc[train_indices], features.iloc[test_indices]
        y_train, y_test = target[train_indices], target[test_indices]

        # Huấn luyện mô hình
        classifier.fit(X_train, y_train)

        # Dự đoán kết quả trên tập kiểm tra
        y_pred = classifier.predict(X_test)

        # Tính toán các chỉ số đánh giá và thêm vào danh sách tương ứng
        accuracy_list.append(accuracy_score(y_test, y_pred))
        precision_list.append(precision_score(y_test, y_pred, average='weighted', zero_division=1))
        recall_list.append(recall_score(y_test, y_pred, average='weighted', zero_division=1))
        f1_list.append(f1_score(y_test, y_pred, average='weighted', zero_division=1))

    # Trả về các chỉ số đánh giá trung bình
    return {
        'accuracy': sum(accuracy_list) / len(accuracy_list),
        'precision': sum(precision_list) / len(precision_list),
        'recall': sum(recall_list) / len(recall_list),
        'f1': sum(f1_list) / len(f1_list)
    }


from sklearn.neighbors import KNeighborsClassifier

# Khởi tạo mô hình
model = KNeighborsClassifier()

# Chọn features và target
features = df.drop('classification', axis=1)
target = df['classification']

# Số lượng fold
k = 10


# Gọi hàm simple_cross_validation
results = simple_cross_validation(model, features, target, k)

# In kết quả
print(results)


# folds = k_fold(df['classification'], 5)
#
# # In ra kết quả
# for i, (train_indices, test_indices) in enumerate(folds):
#     print(f"Fold {i+1}:")
#     print("Train indices:", train_indices)
#     print("Test indices:", test_indices)
