# -*- coding: utf-8 -*-

'''k-近邻算法(kNN)的预测步骤:
对未知类别属性的数据集中的每个点依次执行以下操作：
(1)计算已知类别数据集中的点与当前点之间的距离；(使用欧氏距离公式：相同纬度距离的平方和开根号)
(2)按照距离递增次序排序；
(3)选取与当前点距离最小的走个点；
(4)确定前灸个点所在类别的出现频率；
(5)返回前女个点出现频率最高的类别作为当前点的预测分类。
'''
# kNN算法无需训练

import numpy as np
import operator #https://www.cnblogs.com/nju2014/p/5568139.html

'''----------------------exercises1---------------------------'''
#构造第一个分类器
def classify0(inX, dataSet, labels, k):
    # 取行数
    #因为dataSet的行数和labels的元素个数是相等的
    dataSetSize = dataSet.shape[0]
    #使用欧氏距离公式计算两个向量点的距离
    #np.tile的用法http://blog.csdn.net/xiahei_d/article/details/52749395
    #复数组inX来构建新的数组
    #步骤1------计算已知数据集中的点与当前输入数据点的距离(放在矩阵里算方便)
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1) #按照列相加（每行求和），形成一行
    distances = sqDistances ** 0.5

    #步骤2------按照距离递增排序
    # argsort返回数组值从小到大的索引值，给labels用
    sortedDistIndicies = distances.argsort()

    classCount = {} #用来对label中的每一类元素计数

    #步骤3------选取与当前距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        #dict.get(key, default=None):获取key的数量，如果key不存在，返回默认值（这里为0）
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #dict.items():返回字典dict的(key,value)元组对的列表
    #sorted函数：http://www.runoob.com/python/python-func-sorted.html
    #这里classCount是一个元祖组成的字典，可迭代对象为每个元祖，因此key中指定的以元祖下标1来排序
    sortedClassCount = sorted(classCount.items(),
                              key = operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]

#将文本记录到转换numpy的解析程序
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)    #得到文件行数
    returnMat = np.zeros((numberOfLines, 3))    #返回numpy数组（矩阵）
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip() #去掉每一行的空白符
        listFromLine = line.split('\t') #去掉每一行的\r符号
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

#归一化特征值（特征缩放）
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(dataSet.shape)
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet/np.tile(ranges, (m, 1))   #特征值相除
    return normDataSet, ranges, minVals

#测试test1的分类器效果
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt') #加载文件数据
    normMat, ranges, minVals = autoNorm(datingDataMat)  #特征归一化
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0 #错误次数计数器
    for i in range(numTestVecs):
        classifierResult = classify0(
            normMat[i, :], normMat[numTestVecs:m, :],\
            datingLabels[numTestVecs:m], 3)
        print('the classifier came back with: %d, the real answer is: %d'\
              % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1
    print('the total error rate is: %.3f' % (errorCount/numTestVecs))
'''-------------------------------------------------------'''

'''----------------------exercises2---------------------------'''
#为了使用kNN的分类器，必须将图像格式化处理为一个向量
##当前文件是一个32*32的二进制图像矩阵，转换为1*1024的向量
#将数据处理成分类器可以识别的格式
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    ''' 自己写的，也能实现功能
    li = fr.readlines()
    string = ''
    for l in li:
        l = l.strip('\n')
        string += l
    return np.array(list(string), dtype=np.float64).reshape((1, 1024))
    '''
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

import os
def handwritingClassTest():
    #获取目录内容
    hwLabels = []
    trainingFileList = os.listdir('trainingDigits') #获取指定目录下的文件名列表
    m = len(trainingFileList)   # m：表示目录中文件的个数
    # 根据文件个数创建训练矩阵，该矩阵的每行数据存储一个图像
    trainingMat = np.zeros((m ,1024))
    for i in range(m):  # 训练集：从文件名解析数字
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])    #从文件名中解析出分类数字
        hwLabels.append(classNumStr)
        ## img2vector：自己定义的图像载入函数
        trainingMat[i, :] = img2vector('trainingDigits/' + fileNameStr)
    testFileList = os.listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):  # 测试集：从文件名解析数字
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split(',')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/' + fileNameStr)
        ## 由于文件中的值已经在[0,1]之间，因此不需要特征缩放
        classifierResult = classify0(vectorUnderTest, trainingMat,
                                     hwLabels, 3)
        print('the classifier come back with: %d, the real answer is: %d'\
              % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1
        print('\nthe total number of errors is: ', errorCount)
        print('\nthe total error rate is: %f' % (errorCount/float(mTest)))

'''-------------------------------------------------------'''

'''
#创建数据集和标签
def createDataSet():
    group = np.array([
        [1.0, 1.1],
        [1.0, 1.0],
        [0, 0],
        [0, 0.1]
    ])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

group, labels = createDataSet()
print(classify0([1, 1], group, labels, 3))
'''
