
import kNN算法.kNN as kNN
import numpy as np

def classifyPerson():
    resultList = [
        'not at all',
        'in small doses',
        'in large doses'
    ]
    percentTats = float(input(
        '玩视频游戏所耗时间百分比?'
    ))
    ffMiles = float(input('每年获得的飞行常客里程数?'))
    iceCream = float(input('每周消费的冰淇淋公升数?'))
    datingDataMat, datingLabes = kNN.file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = kNN.classify0((inArr-minVals)/ranges,
                                     normMat, datingLabes, 3)
    print('You will probably like this person: ',
          resultList[classifierResult-1])

classifyPerson()