
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(
        nodeTxt, xy=parentPt, xycoords='axes fraction',
        xytext = centerPt, textcoords = 'axes fraction',
        va = 'center', ha = 'center', bbox = nodeType,
        arrowprops = arrow_args
    )

'''在父子节点间填充文本信息'''
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    '''计算宽和高'''
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff +
              (1.0 + float(numLeafs))/2.0/plotTree.totalW,
               plotTree.yOff)

    '''标记子节点属性值'''
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    '''减少y偏移'''
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff),
                                       cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff),
                        cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

# 绘制树
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')  # 创建一个新图形
    fig.clf()   # 清空绘图区
    axprops = dict(xticks = [], yticks = [])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


# 获取叶节点的数目
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

# 主要用于测试，返回预定义的树结构。
def retrieveTree(i):
    listofTrees = [
        {
            'no surfacing':{
                0:'no',
                1:{
                    'flippers':{
                        0:'no',
                        1:'yes'
                    }
                }
            }
        },
        {
            'no surfacing':{
                0:'no',
                1:{
                    'flippers':{
                        0:{
                            'head':{
                                0:'no',
                                1:'yes'
                            }
                        },
                        1:'no'
                    }
                }
            }
        }
    ]
    return listofTrees[i]


import 决策树算法.ID3算法.trees as trees
myDat, labels = trees.createDataSet()
myTree = trees.createTree(myDat, labels)
# 尝试向树中加入新的结构，看看图形的形状
myTree['no surfacing'][1]['flippers'][2] = 'maybe'
print(getNumLeafs(myTree))
print(getTreeDepth(myTree))
createPlot(myTree)
