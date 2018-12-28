
#2.2.1代码
##每个模块单独进行测试

import numpy as np
import matplotlib.pyplot as plt
import kNN算法.kNN as kNN  #kNN是自己实现的算法

datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
#绘制散点图直观的展示数据
import matplotlib.pyplot as plt

fig = plt.figure()
#添加子图
##表示总大小是一行一列，当前子图在第一个位置
ax = fig.add_subplot(1, 1, 1)
#散点图接口
##与plt.scatter等同
#ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
           np.array(datingLabels), np.array(datingLabels))
#plt.show()

normMat, ranges, minVals = kNN.autoNorm(datingDataMat)

kNN.datingClassTest()