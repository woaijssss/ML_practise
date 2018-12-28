
import pickle

# 保存树
def storeTree(inputTree, filename):
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

# 加载树
def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)
