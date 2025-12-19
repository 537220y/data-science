from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris  # 以自带的鸢尾花数据集为例

# 1. 准备数据 (特征 X 和 标签 y)
iris = load_iris()
X, y = iris.data, iris.target

# 2. 实例化模型
# max_depth 是树的最大深度，对应你图中代码的参数
clf = DecisionTreeClassifier(max_depth=5)

# 3. 训练模型 (拟合)
clf.fit(X, y)

# 4. 可视化 (也就是你图片中的代码部分)
plt.figure(figsize=(18, 10))
plot_tree(clf, 
          filled=True,      # 给节点填充颜色
          rounded=True,     # 节点框设为圆角
          fontsize=10)      # 字体大小
plt.show()