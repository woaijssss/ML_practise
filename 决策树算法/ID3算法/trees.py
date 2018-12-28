
# 决策树的主要优点就是直观、易于理解
'''
度量数据集合无序程度的方法：
1、信息熵
2、Gini系数：从数据集中随机选取子项，度量其被错误分类到其他分组里的概率。
'''

from math import log

'''信息熵计算
信息熵越高，则混合的数据越多
myDat[0][-1] = 'maybe' 增加第三个名为maybe的分类，信息熵会增大
熵计算将会告诉我们如何划分数据集是最好的数据组织方式
'''
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)   # 获取数据集长度
    labelCounts = {}    # 保存每个标签的数量
    for featVec in dataSet:
        currentLabel = featVec[-1]  # 取到每一行最后一列的标签输出列
        # 为所有可能分类创建字典
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    '''
    计算信息熵
    依据香农的信息熵计算公式
    '''
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries   # 每个标签在所有样本中的占比
        shannonEnt -= prob * log(prob, 2)   # 以2为底数求对数
    return shannonEnt

# 创建数据集
def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

'''按照给定特征划分数据集
@param dataSet：待划分的数据集
@param axis：划分数据集的特征
@param value：特征的返回值
'''
def splitDataSet(dataSet, axis, value):
    retDataSet = [] # 创建要返回的list对象
    for featVec in dataSet: # 抽取数据
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

'''
选择最好的数据集划分方式
'''
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1   # 获取数据集中的特征数量
    baseEntropy = calcShannonEnt(dataSet)   # 初始节点的信息熵
    bestInfoGain, bestFeature = 0.0, -1 # 初始化信息增益和最好的特征
    for i in range(numFeatures):    # 创建唯一的分类标签列表
        featList = [example[i] for example in dataSet]
        '''集合set：集合类型中的每个值互不相同
        从列表中创建集合是python得到列表中唯一元素值最快的方法。
        '''
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:    # 计算每种划分方式的信息熵
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 信息增益是熵的减小或者数据无序度的减小。
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):   # 计算最好的信息增益
            bestInfoGain = infoGain
            bestFeature = i # 返回第i个特征用于划分
    return bestFeature

'''
myDat, labels = createDataSet()
print(myDat)
#输出为0，表示第0个特征是最好的用于划分数据集的特征
print(chooseBestFeatureToSplit(myDat))
print(myDat)
'''

import operator

# 采用投票法决定叶子节点的分类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] = 1
    sortedClassCount = sorted(classCount.items())
    key = operator.itemgetter(1, reversed = True)
    return sortedClassCount[0][0]

'''
:param dataSet：数据集
:param labels：标签列表(包含了数据集中所有的特征的标签)
算法本身并不需要labels变量，这里是为了给出数据的明确含义。
'''
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]    # 取到每一行最后一列的标签
    '''如果类别完全相同，则停止划分，直接返回该类标签'''
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    '''遍历完所有特征时返回出现次数最多的
    由于这里无法简单的返回唯一的类标签，因此挑选出现次数最多的类别作为返回值
    '''
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    '''选取当前数据集中最好的特征'''
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    '''myTree存储了树的所有信息'''
    myTree = {bestFeatLabel:{}} # 开始创建树
    del(labels[bestFeat])
    # 得到l列表包含的所有属性值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)

    '''遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用
    createTree函数
    '''
    for value in uniqueVals:
        '''因为python中函数传参列表是按照引用传递的。为了确保每次调用函数
        createTree时不改变原始列表的内容，使用新变量subLabels代替原始列表。
        '''
        subLabels = labels[:]   # 复制类标签
        myTree[bestFeatLabel][value] = createTree(splitDataSet(
            dataSet, bestFeat, value), subLabels
        )
    return myTree

'''使用决策树执行分类
程序比较测试数据与决策树上的数值，递归执行该过程直到进入叶子节点，
最后将测试数据定义为叶子节点所属的类型
'''
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    # 将标签字符串转换为索引
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key],
                                      featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


#import 决策树算法.ID3算法.treePlotter as treePlotter
#import 决策树算法.saveTree as saveTree
myDat, label = createDataSet()
labels = []
labels = label.copy()
myTree = createTree(myDat, label)
# myTree：包含了很多代表树结构信息的嵌套字典
#myTree = treePlotter.retrieveTree(0)
#myTree1 = saveTree.grabTree('classifierStorage.txt')
#print(classify(myTree, labels, [0, 1]))
#print(classify(myTree, labels, [1, 1]))
#saveTree.storeTree(myTree, 'classifierStorage.txt')