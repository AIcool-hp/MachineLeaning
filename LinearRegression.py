from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

# 加载数据集
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# 将数据转换为DataFrame以便查看
data = pd.DataFrame(data=X, columns= diabetes.feature_names)
target = pd.Series(y, name= diabetes.target_filename[0])
print(data.head())
print(f'\n{target.head()}')

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建与训练线性回归模型
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 模型预测
y_pred = lr_model.predict(X_test)

# 回归模型评估
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
intercept = lr_model.intercept_
coefficients = lr_model.coef_

print(f"\n均方误差 (MSE): {mse:.2f}")
print(f"决定系数 (R²): {r2:.2f}")
print(f"平均绝对误差 (MAE): {mae:.2f}")
print(f"均方根误差 (RMSE): {rmse:.2f}")

print("\n截距与系数:")
print(f"截距: {intercept:.2f}")
print("系数:")
for feature, coef in zip(diabetes.feature_names, coefficients):
    print(f"{feature}: {coef:.4f}")


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 可视化预测结果与误差分布
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='完美预测线')
plt.xlabel('真实值')
plt.ylabel('预测值 ')
plt.title('线性回归模型预测结果')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
errors = y_test.flatten() - y_pred.flatten()
plt.hist(errors, bins=15, edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--', label='零误差线')
plt.xlabel('预测误差')
plt.ylabel('频数')
plt.title('预测误差分布')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

plt.tight_layout()

plt.show()






