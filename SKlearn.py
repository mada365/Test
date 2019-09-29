from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target

print (X.shape)

print (y.shape)



# 模型调用

from sklearn.neighbors import KNeighborsClassifier



# 创建实例


knn = KNeighborsClassifier(n_neighbors=1)

# print(knn)


knn.fit(X,y)


print(knn.predict([[1,2,3,4],[2,1,3,2]]))
