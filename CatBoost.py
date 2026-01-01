import pandas as pd
from sklearn.model_selection import train_test_split
import catboost as cb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# 数据加载
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic = pd.read_csv(url)

print(titanic.info())

# 选择特征和目标变量
X = titanic[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].copy()
y = titanic['Survived']

# 记录类别型特征
cat_feature_names = ['Pclass', 'Sex', 'Embarked']

# 处理类别特征的类型与缺失值
for col in cat_feature_names:
    X[col] = X[col].astype('str')        # catboost仅支持字符串与整数类型的类别特征
    
X['Age'] = X['Age'].fillna(X['Age'].median())

print(X.head())

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建与训练CatBoost分类器
cat_model = cb.CatBoostClassifier(
    iterations=100, 
    depth=6, 
    learning_rate=0.1, 
    loss_function='Logloss', 
    random_state=42,
    cat_features=cat_feature_names,  # 指定类别特征,无需手动编码
    verbose=False)

cat_model.fit(X_train, y_train, cat_features=cat_feature_names)

# 预测与评估
y_pred = cat_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"模型准确率: {accuracy * 100:.2f}%")

# 输出特征重要性
feature_importances = cat_model.get_feature_importance()
feature_names = X.columns

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
plt.barh(importance_df['特征'], importance_df['重要性'], color='lightgreen')
plt.xlabel('特征重要性')
plt.title('CatBoost——特征重要性')
plt.gca().invert_yaxis()
plt.show()