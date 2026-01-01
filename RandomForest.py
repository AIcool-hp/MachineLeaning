from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据集
iris = load_iris()

# 分割数据集
X, y = iris.data, iris.target

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
rf_classifier.fit(X_train, y_train)

# 进行预测
y_pred = rf_classifier.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)

print(f"模型准确率: {accuracy * 100:.2f}%")

# 输出特征重要性
feature_importances = rf_classifier.feature_importances_
feature_names = iris.feature_names

# 创建DataFrame以显示特征重要性
importance_df = pd.DataFrame({
    '特征': feature_names,
    '重要性': feature_importances
}).sort_values(by='重要性', ascending=False)

print("特征重要性:")
print(importance_df)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh(importance_df['特征'], importance_df['重要性'], color='skyblue')
plt.xlabel('特征重要性')
plt.title('随机森林——特征重要性')
plt.gca().invert_yaxis()
plt.show()

# 保存模型
# import joblib
# joblib.dump(rf_classifier, 'random_forest_model.pkl')


