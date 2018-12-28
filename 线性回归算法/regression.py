
import numpy as np

def loadDataSet(fileName):
	numFeat = len(open(fileName).readline().split('\t')) - 1
	dataMat, labelMat = [], []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr = []
		curLine = line.strip().split('\t')
		for i in range(numFeat):
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	return dataMat, labelMat

'''计算最佳拟合值线'''
def standRegres(xArr, yArr):
	xMat, yMat = np.mat(xArr), np.mat(yArr).T
	xTx = xMat.T * xMat		# xTx：使用正规方程的方法求解回归系数
	'''linalg.det(A)：计算A的行列式'''
	if np.linalg.det(xTx) == 0.0:	# 判断行列式是否为0，如果为0，则矩阵求逆错误
		print('This matrix is singular, connot do inverse')
		return
	theta = xTx.I * (xMat.T*yMat)
	# 等效的
	#t heta = np.linalg.solve(xTx, xTx.T*yMat)
	return theta

'''局部加权线性归回函数'''
def lwlr(testPoint, xArr, yArr, k=1.0):
	# 1、创建对角矩阵(权重矩阵是一个方阵，阶数等于样本点个数)
	## 即：该矩阵为每个样本点初始化了一个权重
	xMat, yMat = np.mat(xArr), np.mat(yArr).T
	m = np.shape(xMat)[0]
	weights = np.mat(np.eye((m)))	# mat函数将一个列表转换成相应的矩阵类型。

	# 2、使用高斯核计算权重值矩阵
	for j in range(m):
		diffMat = testPoint - xMat[j, :]
		weights[j, j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))	# 参数k控制指数衰减的速度
	xTx = xMat.T * (weights*xMat)
	if np.linalg.det(xTx) == 0.0:
		print('This matrix is singular, connot do inverse')
		return
	theta = xTx.I * (xMat.T * (weights * yMat))
	return testPoint*theta

def lwlrTest(testArr, xArr, yArr, k=1.0):
	m = np.shape(testArr)[0]
	yHat = np.zeros(m)
	for i in range(m):
		yHat[i] = lwlr(testArr[i], xArr, yArr, k)
	return yHat