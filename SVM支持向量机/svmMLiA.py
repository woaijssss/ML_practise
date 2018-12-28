
import random
import numpy as np

def loadDataSet(filename):
    dataMat, labalMat = [], []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labalMat.append(float(lineArr[2]))
    return dataMat, labalMat

'''
:param i:第一个alpha的下标
:param m:所有alpha的数目
只要函数值不等于输入值i，函数就会进行随机选择
'''
def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j

'''
辅助函数
'''
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

'''
简化版SMO算法
可能是本书中最大的一个函数
:param dataMatIn:数据集
:param classLabels:类别标签
:param C:常数
:param toler:容错率
:param maxIter:取消前最大循环次数
'''
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    # 将多个列表和输入参数转换成numpy矩阵，可以简化很多数学处理操作。
    dataMatrix, labelMat = np.mat(dataMatIn), np.mat(classLabels).transpose()
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while(iter < maxIter):
        # 每次循环中，alphaPairsChanged设置为0，
        alphaPairsChanged = 0
        # 如果alpha可以更改，进入优化过程
        for i in range(m):
            fXi = float(np.multiply(alphas, labelMat).T*\
                        (dataMatrix*dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or \
                    ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                # 随机选择第二个alpha
                j = selectJrand(i, m)
                fxj = float(np.multiply(alphas, labelMat).T*\
                        (dataMatrix*dataMatrix[j, :].T)) + b
                Ej = fxj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 保证alpha在0和C之间
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j]-alphas[i])
                    H = min(C, C+alphas[j]-alphas[i])
                else:
                    L = max(0, alphas[j]+alphas[i]-C)
                    H = min(C, alphas[j]+alphas[i])
                if L == H:
                    print('L==H')
                    continue
                eta = 2.0*dataMatrix[i, :]*dataMatrix[j, :].T - \
                    dataMatrix[i, :]*dataMatrix[i, :].T - \
                    dataMatrix[j, :]*dataMatrix[j, :].T
                if eta >= 0:
                    print('eta >= 0')
                    continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j]-alphaJold) < 0.00001):
                    print('j not moving enough')
                    continue
                # 对i进行更改，修改量与j相同，但方向相反
                alphas[i] += labelMat[j]* labelMat[i] * \
                             (alphaJold - alphas[j])

                # 设置常数项
                b1 = b - Ei - labelMat[i]*(alphas[i]-alphaIold)*\
                    dataMatrix[i, :]*dataMatrix[i, :].T - \
                    labelMat[j]*(alphas[j]-alphaJold) * \
                    dataMatrix[i, :]*dataMatrix[j, :].T
                b2 = b - Ei - labelMat[i]*(alphas[i]-alphaIold)*\
                    dataMatrix[i, :]*dataMatrix[j, :].T - \
                    labelMat[j]*(alphas[j]-alphaJold) * \
                    dataMatrix[j, :]*dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif(0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print('iter:%d, i:%d, pairs changed %d' %
                      (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print('iteration number: %d' % iter)
    return b, alphas

dataArr, labelArr = loadDataSet('testSet.txt')
b, alpha = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
print('b---->: ', b)