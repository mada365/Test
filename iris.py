# 鸢尾花数据加载
from sklearn import datasets
iris = datasets.load_iris()
#展示鸢尾花数据

print('鸢尾花数据内容：\n',iris.data )

#展示字头的含义 花萼长度,花萼宽度,花瓣长度,花瓣宽度
print('鸢尾花表头:\n',iris.feature_names)

#输出的结果

print('输出的结果：\n',iris.target)

#结果的含义

print('结果的含义:\n',iris.target_names)


# 输入数据赋值X，输出数据赋值y

X = iris.data
y = iris.target



# =============================================================================================
# 1模型调用
# 2模型初始化
# 3模型训练
# 4模型预测

# 模型调用
from sklearn.neighbors import KNeighborsClassifier

#创建实例

knn = KNeighborsClassifier(n_neighbors=14)

#模型训练

knn.fit(X,y)

# 测试数据
x_test = [[1,2,3,4],[2,4,1,2]]
#预测结果
print('展示出来的结果:\n',knn.predict(x_test))