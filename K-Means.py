import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# 加载数据集
url = 'https://github.com/tanishq21/Mall-Customers/blob/main/Mall_Customers.csv?raw=true'
df = pd.read_csv(url)

print(df.head())

# 选择特征
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]


# 特征标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=['Annual Income (k$)', 'Spending Score (1-100)'])

print(X.head())

# 构建与训练K-Means模型
kmeans = KMeans(n_clusters=5, random_state=42)

kmeans.fit(X)

# 预测簇标签
y_kmeans = kmeans.predict(X)
data = pd.Series(y_kmeans, name='Cluster')

print(data.value_counts())

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 可视化聚类结果
plt.figure(figsize=(10, 6))
plt.scatter(X['Annual Income (k$)'],
             X['Spending Score (1-100)'], 
             c=y_kmeans, 
             s=50, 
             cmap='viridis')

plt.scatter(kmeans.cluster_centers_[:, 0],
             kmeans.cluster_centers_[:, 1],
                c='red',
                s=200,
                alpha=0.75,
                marker='*',
                label='Centroids')

plt.title('商场顾客类别分析')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# 选择最佳簇数的肘部法则
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_) 

plt.figure(figsize=(10, 6))
plt.plot(K, inertia, color='purple', marker='o')
plt.axvline(x=5, color='blue', linestyle='--', label='最优K值(肘部点):5')
plt.xlabel('簇数 k')
plt.ylabel('惯性值')
plt.title('肘部法则确定最佳簇数')
plt.legend()
plt.show()








