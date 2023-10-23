import pandas as pd


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

X = df.drop('classification', axis=1)
y = df['classification']

def chia_k_fold(nhan,k):
    soluong = len(nhan)
    dodai = soluong // k
    mang_1 = []
    for i in range(k):
        batdau = i*dodai
        if i == k - 1:
            ketthuc = soluong
        else:
            ketthuc = batdau + dodai
        mang_1.append((batdau,ketthuc))
    mang_2 = []
    for i in range(k):
        batdau , ketthuc = mang_1[i]
        kiemtra = list(range(batdau,ketthuc))
        huanluyen = list(range(0,batdau))+list(range(ketthuc,soluong))
        mang_2.append((kiemtra,huanluyen))
    return mang_2

import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

def danh_gia_trung_binh(model, X , nhan , k):

    mang_acc = []
    mang_pre = []
    mang_recall = []
    mang_f1= []

    folds = chia_k_fold(nhan,k)

    for kiemtra , huanluyen in folds:
        clf = copy.copy(model)

        X_train, X_test = X.iloc[huanluyen], X.iloc[kiemtra]
        y_train, y_test = y[huanluyen], y[kiemtra]

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        mang_acc.append(accuracy_score(y_test, y_pred))
        mang_pre.append(precision_score(y_test, y_pred, average='weighted', zero_division=1))
        mang_recall.append(recall_score(y_test, y_pred, average='weighted', zero_division=1))
        mang_f1.append(f1_score(y_test, y_pred, average='weighted', zero_division=1))
    return {
        'accuracy': sum(mang_acc) / len(mang_acc),
        'precision': sum(mang_pre) / len(mang_pre),
        'recall': sum(mang_recall) / len(mang_recall),
        'f1': sum(mang_f1) / len(mang_f1)
    }

model = KNeighborsClassifier()
k = 5
kq = danh_gia_trung_binh(model,X,y,k)

print(kq)
