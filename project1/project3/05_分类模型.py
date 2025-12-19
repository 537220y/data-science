
import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt

# ==========================================
# 1. 数据加载与预处理 (Data Loading & Preprocessing)
# ==========================================
base_dir = Path(r'd:/个人/助教/HW2/提前测试')
filename = 'goog4_request&X-Goog-Date=20251208T125933Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=075296d13eb7db7c856a6daba2b771a21c2d7d4c82e8ded759ecd14797124a5d54488eef1ee7c50151f823cd2c6be9e5607540a3805e82980196a42.xlsx'
dataset_path = base_dir / filename
df = pd.read_excel(dataset_path)

# 列名规范化
def canonical_colname(name):
    s = str(name).strip().lower()
    s = s.replace('_', ' ').replace('-', ' ')
    s = re.sub(r'\s+', ' ', s)
    m = {
        'gender': 'gender',
        'race/ethnicity': 'race/ethnicity',
        'parental level of education': 'parental level of education',
        'lunch': 'lunch',
        'test preparation course': 'test preparation course',
        'math score': 'math score',
        'reading score': 'reading score',
        'writing score': 'writing score',
    }
    return m.get(s, str(name))

df = df.rename(columns=lambda c: canonical_colname(c))

# 转换数值列
for col in ['math score','reading score','writing score']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 创建目标变量：平均分 >= 75
df['average score'] = df[['math score','reading score','writing score']].mean(axis=1)
df['High Achiever'] = (df['average score'] >= 75).astype(int)

print(f"High Achiever 样本分布:\n{df['High Achiever'].value_counts(normalize=True)}")

# ==========================================
# 2. 特征工程 (Feature Engineering)
# ==========================================
# 准备特征矩阵 X。注意：必须排除成绩相关列，否则会发生数据泄露！
feature_cols = [
    'gender', 
    'race/ethnicity', 
    'parental level of education', 
    'lunch', 
    'test preparation course'
]

X = df[feature_cols].copy()
y = df['High Achiever']

# 2.1 有序编码：父母教育程度
edu_order = {
    "some high school": 1,
    "high school": 2,
    "some college": 3,
    "associate's degree": 4,
    "bachelor's degree": 5,
    "master's degree": 6
}
# 简单的清理函数以匹配键值
def clean_edu(x):
    s = str(x).strip().lower()
    if 'associate' in s: return "associate's degree"
    if 'bachelor' in s: return "bachelor's degree"
    if 'master' in s: return "master's degree"
    if 'some high' in s: return "some high school"
    if s == 'high school': return "high school"
    if 'some college' in s: return "some college"
    return s

X['parental level of education'] = X['parental level of education'].apply(clean_edu).map(edu_order).fillna(0)

# 2.2 二值编码：午餐、课程、性别
# Lunch: standard=1, free/reduced=0 (假设 standard 更好)
X['lunch'] = X['lunch'].apply(lambda x: 1 if 'standard' in str(x).lower() else 0)

# Test Prep: completed=1, none=0
X['test preparation course'] = X['test preparation course'].apply(lambda x: 1 if 'completed' in str(x).lower() else 0)

# Gender: female=1, male=0 (随意指定，决策树不敏感)
X['gender'] = X['gender'].apply(lambda x: 1 if 'female' in str(x).lower() else 0)

# 2.3 One-Hot编码：种族 (Race)
X = pd.get_dummies(X, columns=['race/ethnicity'], drop_first=True)

print(f"特征矩阵形状: {X.shape}")
print("特征列表:", X.columns.tolist())

# ==========================================
# 3. 模型训练与评估 (Model Training & Evaluation)
# ==========================================
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化并训练决策树
clf = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 计算指标
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

print("-" * 30)
print("模型评估结果 (Decision Tree):")
print(f"Accuracy  (准确率): {acc:.4f}")
print(f"Precision (精确率): {prec:.4f}")
print(f"Recall    (召回率): {rec:.4f}")
print("-" * 30)

# ==========================================
# 4. 特征重要性分析 (Feature Importance)
# ==========================================
importances = clf.feature_importances_
feature_names = X.columns
feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp = feat_imp.sort_values(by='Importance', ascending=False)

print("最重要的前三个影响因素:")
print(feat_imp.head(3))
