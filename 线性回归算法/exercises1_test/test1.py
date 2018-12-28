
import 线性回归算法.regression as regression
import numpy as np

'''ex0.txt文件中，第一个总等于1，即x0，是正规方程中才会用到的特征'''
# 一般用xi表示第i个样本;
# xij表示第i个样本中的第j个特征
# xj表示第j个特征
xArr, yArr = regression.loadDataSet('ex0.txt')

# 变量theta存放的是回归系数
'''最终得到的回归方程为：
	y = theta[0] + theta[1] * X1
'''
theta = regression.standRegres(xArr, yArr)
'''np.mat：可以将list转化为matrix'''
xMat = np.mat(xArr)
yMat = np.mat(yArr)

# 现在就可以绘出数据集散点图和最佳拟合直线图：
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
'''画图要注意的地方：
如果值线上的数据点次序混乱，绘图时将出现问题，所以首先要将点按照升序排列。
'''
xCopy = xMat.copy()
xCopy.sort(0)
yHat = xCopy*theta
ax.plot(xCopy[:, 1], yHat)
plt.show()
'''几乎任一数据集都可以用上述方法建立模型'''
# 比较模型的效果，即：计算预测值yHat序列和真实值y序列的匹配程度（计算两序列的相关系数）
yHat = xMat * theta
print(np.corrcoef(yHat.T, yMat))	# np.corrcoef：计算两个序列相关系数(两个序列的形状要一致)

