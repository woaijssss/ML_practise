
import 线性回归算法.regression as regression
import numpy as np
from imp import reload
# reload函数希望获得的参数是一个已经加载了的模块对象的名称，所以如果在重载之前,请确保已经成功地导入了这个模块。
#reload(regression)

xArr, yArr = regression.loadDataSet('ex0.txt')
'''可以对单点进行估计'''
print(yArr[0])
print(regression.lwlr(xArr[0], xArr, yArr, 1.0))
print(regression.lwlr(xArr[0], xArr, yArr, 0.001))

# 为了得到数据集里所有点的估计，可以调用lwlrTest()函数
yHat = regression.lwlrTest(xArr, xArr, yArr, 0.01)

# 绘出这些估计值和原始值，看看yHat的拟合效果。
# 1、首先对xArr排序
xMat = np.mat(xArr)
srtInd = xMat[:, 1].argsort(0)	# np.argsort：返回的是数组值从小到大，对应索引值
xSort = xMat[srtInd][:, 0, :]

# 2、用matplotlib绘图
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:, 1], yHat[srtInd])
'''a是个矩阵或者数组，a.flatten()就是把a降到一维，默认是按横的方向降
a.flatten().A其实这是因为此时的a是个矩阵，降维后还是个矩阵，矩阵.A（等效于矩阵.getA()）变成了数组
'''
ax.scatter(xMat[:, 1].flatten().A[0], np.mat(yArr).T.flatten().A[0], s=2, c='red')
plt.show()
