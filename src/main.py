import pandas as pd


df = pd.read_csv('../data/car.data', names = ["buying", "maintenance", "doors", "persons", "lug_boot", "safety", "classification"])

# buying ->      0: v-high,   1: high,   2: med,    3: low
# maintenance -> 0: v-high,   1: high,   2: med,    3: low
# doors ->       0: 2,        1: 3,      2: 4,      3: 5 - more
# persons ->     0: 2,        1: 4,      2: more
# lug_boot ->    0: small,    1: med,    2: big
# safety ->      0: low,      1: med,    2: high
#classification  0:unacc,     1: acc,    2: vgood,  3:good,

X = df.drop('classification', axis=1)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
one_hot = OneHotEncoder()

encoder = ColumnTransformer(transformers=[
    ('OneHot', one_hot, [0,1,2,3,4,5])
], remainder='passthrough')

encoder_result = encoder.fit_transform(X)

X = pd.DataFrame(encoder_result.toarray())

def dauvao(num):
    if num == 1 :
        X = df.drop('classification', axis=1)
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        one_hot = OneHotEncoder()

        encoder = ColumnTransformer(transformers=[
            ('OneHot', one_hot, [0,1,2,3,4,5])
        ], remainder='passthrough')

        encoder_result = encoder.fit_transform(X)

        X = pd.DataFrame(encoder_result.toarray())
        y = df['classification']
        return X , y
    else :
        for column in df.columns:
            if column != 'classification':
                df[column], _ = pd.factorize(df[column])

        #classification  0:unacc,     1: acc,    2: good,  3:vgood,

        cfc = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
        df['classification'] = df['classification'].map(cfc)

        X = df.drop('classification', axis=1)
        y = df['classification']
        return X , y

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

# k = 10
# print("KNN :",danh_gia_trung_binh(KNeighborsClassifier(),X,y,k))
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import GaussianNB
# print("Bayes :",danh_gia_trung_binh(GaussianNB(),X,y,k))
# from sklearn.tree import DecisionTreeClassifier
# print("Tress :",danh_gia_trung_binh( DecisionTreeClassifier(),X,y,k))


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from statistics import mean, stdev
models = {
    'Naive Bayes': GaussianNB(),
    'KNN': NearestCentroid(),
    'Decision Tree': DecisionTreeClassifier(),
}

metrics = ['accuracy', 'precision', 'recall', 'f1']

#1: là chuẩn hóa theo ông nội kia
X , y = dauvao(1)

def plot_metric(metric,sta,end):
    for model in models.keys():
        averages = []
        for k in range(sta, end+1):
          results = danh_gia_trung_binh(models[model], X, y, k)
          averages.append(results[metric])
        # print(model,":",mean(averages))

for metric in metrics:
  # print(metric)
  plot_metric(metric,5,20)

results = {}
for name, model in models.items():
    results[name] = danh_gia_trung_binh(model, X, y, 28)

for metric in metrics:
    print(f'-----< {metric} >-----')

    for model, result in results.items():
        # mean_value = mean(result[metric])
        # stdev_value = stdev(result[metric])
        #
        # print(f'# {model}\nME: {stdev_value}\nDES: {stdev_value}\n')

        value = result[metric]

        print(f'# {model}\n{metric.upper()}: {value}\n')
