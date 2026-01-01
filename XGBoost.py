from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据集
iris= load_iris()

# 分割数据集
X, y = iris.data, iris.target

# 划分训练集与测试集
X_tain, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建模型
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,  # 控制每棵树的贡献权重
    random_state=42,
    #use_label_encoder=False,  
    eval_metric='mlogloss' 
)

# 训练模型
xgb_model.fit(X_tain, y_train)

# 预测结果
y_pred= xgb_model.predict(X_test)

# 评估模型
accuracy= accuracy_score(y_test, y_pred)

print(f'模型准确率:, {accuracy:.2f}')

# 输出特征重要性
feature_importances = xgb_model.feature_importances_
feature_names = iris.feature_names

# 创建DataFrame以显示特征重要性
importance_df = pd.DataFrame(
    {'重要性': feature_importances,
     '特征': feature_names
         }
).sort_values('重要性', ascending=False)

print("特征重要性:")
print(importance_df)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 可视化
plt.figure(figsize=(10,6))
plt.barh(importance_df['特征'], importance_df['重要性'] )
plt.xlabel('特征重要性')
plt.title('XGBoost——特征重要性')
plt.gca().invert_yaxis()
plt.show()


