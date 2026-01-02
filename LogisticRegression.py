import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 加载数据集
iris= load_iris()
X, y = iris.data, iris.target

# 创建DataFrame以便查看
data = pd.DataFrame(data=X, columns= iris.feature_names)
target = pd.Series(y, name=iris.target_names[0])

print(data.head())
print(f'\n{target.head()}')

# 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\n标准化后的特征示例:")
print(pd.DataFrame(X_train, columns=iris.feature_names).head())

# 创建与训练逻辑回归模型
log_reg_model = LogisticRegression(
    max_iter=200, 
    random_state=42,
    penalty='l2',     # 正则化类型
    C = 1.0,
    solver='lbfgs',   # 优化算法
)

log_reg_model.fit(X_train, y_train)

# 模型预测
y_pred = log_reg_model.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'\n模型准确率: {accuracy:.2f}')
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print("混淆矩阵:")
print(conf_matrix)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 可视化混淆矩阵
plt.figure(figsize=(10,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('模型混淆矩阵')
plt.show()








