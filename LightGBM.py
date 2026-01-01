from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb  
import matplotlib.pyplot as plt 
import pandas as pd 

# 加载数据集
iris = load_iris()
X, y = iris.data, iris.target


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 构建与训练模型
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,         
    learning_rate=0.1,       
    num_leaves=31,            # LightGBM特有参数：每棵树的叶子节点数
    random_state=42,          
    verbose=-1                
)

lgb_model.fit(X_train, y_train)  

# 模型预测与评估
y_pred = lgb_model.predict(X_test)  
accuracy = accuracy_score(y_test, y_pred)  

print(f"模型准确率：{accuracy:.4f}")

# 输出特征重要性
feature_importances = lgb_model.feature_importances_
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
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

# 可视化
plt.figure(figsize=(10,6))
plt.barh(importance_df['特征'], importance_df['重要性'] )
plt.xlabel('特征重要性')
plt.title('LightGBM——特征重要性')
plt.gca().invert_yaxis()
plt.show()
