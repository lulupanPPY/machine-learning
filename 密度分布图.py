import numpy as np
from sklearn.datasets import make_classification
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt

X,y=make_classification(1000,n_features=8,n_informative=2,n_redundant=2,n_classes=2,random_state=0)
#存为dataframe格式
df = DataFrame(np.hstack((X, y[:, None])),columns = list(range(8))+["class"])
print(df.columns)
#使用pairplot去看不同特征维度pair下数据的空间分布状况#
# vars表示把里面的特征两两做个可视化
_ = sns.pairplot(df[:50], vars=[0,1,2,3,4,5,6,7], hue="class", size=1.5)
plt.show()
