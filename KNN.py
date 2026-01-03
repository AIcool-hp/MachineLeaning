from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 加载Iris数据集
iris = load_iris()
X, y = iris.data, iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 构建与填充KNN分类器
knn_model = KNeighborsClassifier(
        n_neighbors=5,
        metric='minkowski')

knn_model.fit(X_train, y_train)

# 预测与评估
y_pred = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("===== KNN 模型评估 =====")
print(f"测试集准确率：{accuracy:.4f}")
print(f"选用K值(邻近样本数):{knn_model.n_neighbors}")
print(f"距离度量方式：{knn_model.metric}")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 可视化（混淆矩阵）
plt.figure(figsize=(6, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('KNN分类器混淆矩阵')
plt.show()

# 可视化K值对准确率的影响
k_values = range(1, 50)
accuracies = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski')
    knn.fit(X_train, y_train)
    y_k_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_k_pred)
    accuracies.append(acc)

plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o')
plt.xticks(k_values)
plt.xlabel('K值(邻近样本数)')
plt.ylabel('准确率')
plt.title('K值对KNN分类器准确率的影响')
plt.grid()
plt.show()

