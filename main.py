# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:42:57 2020

@author: SageKwun
"""

import math

# 数据集
# dataSet = [[1, 1, 'yes'],
#            [1, 1, 'yes'],
#            [1, 0, 'no'],
#            [0, 1, 'no'],
#            [1, 1, 'yes'],
#            [1, 1, 'no']]
# 特征集
# labels = ['round', 'red']

# 阈值
e = 0.1

dataSet = [[1, "青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 1],
           [2, "乌黑", "蜷缩", "闷沉", "清晰", "凹陷", "硬滑", 1],
           [3, "乌黑", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 1],
           [4, "青绿", "蜷缩", "闷沉", "清晰", "凹陷", "硬滑", 1],
           [5, "浅白", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 1],
           [6, "青绿", "稍蜷", "浊响", "清晰", "稍凹", "软粘", 1],
           [7, "乌黑", "稍蜷", "浊响", "稍糊", "稍凹", "软粘", 1],
           [8, "乌黑", "稍蜷", "浊响", "清晰", "稍凹", "硬滑", 1],
           [9, "乌黑", "稍蜷", "闷沉", "稍糊", "稍凹", "硬滑", 0],
           [10, "青绿", "硬挺", "清脆", "清晰", "平坦", "软粘", 0],
           [11, "浅白", "硬挺", "清脆", "模糊", "平坦", "硬滑", 0],
           [12, "浅白", "蜷缩", "浊响", "模糊", "平坦", "软粘", 0],
           [13, "青绿", "稍蜷", "浊响", "稍糊", "凹陷", "硬滑", 0],
           [14, "浅白", "稍蜷", "沉闷", "稍糊", "凹陷", "硬滑", 0],
           [15, "乌黑", "稍蜷", "浊响", "清晰", "稍凹", "软粘", 0],
           [16, "浅白", "蜷缩", "浊响", "模糊", "平坦", "硬滑", 0],
           [17, "青绿", "蜷缩", "沉闷", "稍糊", "稍凹", "硬滑", 0]]

labels = ["编号", "颜色", "根蒂", "敲声", "纹理", "脐部", "触感"]


# 决策树类
class CreateTree:

    def __init__(self, dataSet, labels, attribute=None):
        self.dataSet = dataSet
        self.labels = labels
        self.outLabelNum = [data[-1] for data in self.dataSet]
        self.attribute = attribute
        self.tree = {}

    # 获得 dataSet 某一 label 的键值对
    # 已验证
    def getNum(self, index):
        keys = [data[index] for data in self.dataSet]
        keysList = list(set(keys))
        values = [keys.count(x) for x in keysList]
        return dict(zip(keysList, values))

    # 判断 数据集DataSet 中的所有实例是否为同一类
    # 如果是，则返回 类；不是，返回 ''
    def checkIsOneCateg(self):
        flag = ''
        if len(self.getNum(-1)) == 1:  # getNum(-1)?
            # print("fla: " + str(self.getNum(-1)))
            flag = list(self.getNum(-1).keys())[0]
        return flag

    # 判断 labels是否为空集 或者 dataSet在 labels上取值相同
    # 如果是，则返回 最多类；不是，返回 ''
    def checkIsSame(self):
        flag = ''
        # 判断是否为空集
        if len(self.labels) == 1:
            # 返回最多的类
            flag = sorted(self.getNum(-1))[0]
            return flag
        else:
            # 判断是否取值相同
            for i in range(1, len(self.labels)):
                # if len(set([data[i] for data in self.dataSet])) == 1:
                if len(self.getNum(i)) == 1:
                    flag = sorted(self.getNum(-1))[0]
                    return flag
        return flag

    # 计算信息熵
    def calcEntropy(self, dataSet):
        totalNum = len(dataSet)
        labelNum = {}  # 字典：储存各个标签的数量
        entropy = 0  # 信息熵
        '''
        取每一条数据
        取数据最后的判断标签
        如果在字典里，则数量++;否则创建这个字典对
        '''
        for data in dataSet:
            label = data[-1]
            if label in labelNum:
                labelNum[label] += 1
            else:
                labelNum[label] = 1
        '''
        遍历标签字典labelNum的每一个字典对
        计算信息熵
        '''
        for key in labelNum:
            p = labelNum[key] / totalNum
            entropy -= p * math.log2(p)
        # print('ent: ' + str(entropy))
        return entropy

    # 计算 某一属性 信息增益率
    def calcGainRadio(self, attr):
        gain = self.calcEntropy(self.dataSet)  # 总信息熵
        iv = 0
        attrIndex = self.labels.index(attr)  # 属性的列索引
        attrValueList = list(set([data[attrIndex] for data in self.dataSet]))  # 属性值
        newDataSets = []
        if len(attrValueList) != 1:
            for x in attrValueList:
                newDataSets.append([data for data in self.dataSet if data[attrIndex] == x])  # 按照attr划分的新的数据集的列表
            lenD = len(self.dataSet)  # 数据集的长度
            for i in range(len(newDataSets)):
                D_v = newDataSets[i]
                # 验证，正确
                # print("len(D_v): " + str(len(D_v)))
                # print("self.calcEntropy(D_v): " + str(self.calcEntropy(D_v)))
                gain -= len(D_v) / lenD * self.calcEntropy(D_v)
                iv -= len(D_v) / lenD * math.log2(len(D_v) / lenD)
            # 验证，正确
            # print("gain: " + str(gain))
            # print(self.dataSet)
            # print(D_v)
            # print("iv: " + str(iv))
            # print("gain_radio: " + str(gain / iv))
            return gain / iv
        else:
            return 0

    # 找到信息增益率最大的属性
    def getBestAttr(self):
        gainRadioMax = max([self.calcGainRadio(attr) for attr in self.labels[1:]])
        for i in range(1, len(self.labels)):
            if self.calcGainRadio(self.labels[i]) == gainRadioMax:
                bestGainRadioAtrr = self.labels[i]
                return bestGainRadioAtrr

    # 入口函数
    def run(self, e):
        # 输入数据集D， 特征集A， 阈值e
        # t = CreateTree(self.dataSet, self.labels)
        # 如果 D中所有实例属于同一类Ck，则讲该结点标记为C类叶结点
        if self.checkIsOneCateg() != '' or self.checkIsSame() != '':
            self.tree['name'] = self.checkIsOneCateg()
            self.tree['dataSet'] = self.dataSet
            # self.tree['attribute'] = self.attribute
        # 如果最大信息增益率比阈值小，则返回 实例数最大的类

        elif max([self.calcGainRadio(attr) for attr in self.labels[1:]]) < e:
            # print("max: " + str(max([self.calcGainRadio(attr) for attr in self.labels[1:]])))
            self.tree['name'] = self.checkIsOneCateg()
            self.tree['dataSet'] = self.dataSet
            # self.tree['attribute'] = self.attribute
        else:
            # 获得最优划分属性
            bestGainRadioAttr = self.getBestAttr()
            # print(self.dataSet)
            # print(self.labels)
            # print(bestGainRadioAttr)

            bestGainRadioAttrIndex = self.labels.index(bestGainRadioAttr)

            bestGainRadioAttrList = list(set([data[bestGainRadioAttrIndex] for data in self.dataSet]))  # 属性的值列表
            self.tree['attribute'] = bestGainRadioAttr
            for x in bestGainRadioAttrList:
                newDataSet = [data[:bestGainRadioAttrIndex] + data[bestGainRadioAttrIndex + 1:] for data in self.dataSet
                              if data[bestGainRadioAttrIndex] == x]
                newLabels = self.labels[:]
                newLabels.remove(bestGainRadioAttr)
                # print('la: ' + str(newLabels))
                # print('da: ' + str(newDataSet))
                newt = CreateTree(newDataSet, newLabels, attribute=bestGainRadioAttr)
                newt.run(e)
                self.tree[x] = newt.tree


def printDic(dic, layer, value=None):
    if value is not None:
        print(" " * layer + str(value) + "\n" + " " * (layer + 2) + str(dic['attribute']))
    else:
        print(" " * layer + str(dic['attribute']))
    # 该结点内部结点
    if 'name' not in list(dic.keys()):
        subList = list(dic.keys())
        subList.remove('attribute')
        for x in subList:
            # value是叶结点
            if 'name' in list(dic[x].keys()):
                print(" " * (layer + 2) + str(x) + " " + str(dic[x]['name']))
            # value内部结点
            else:
                printDic(dic[x], layer + 2, value=x)
    # 该结点叶结点
    else:
        print(" " * (layer + 2) + str(value) + "\n" + " " * (layer + 2) + str(dict['name']))


if __name__ == '__main__':
    t = CreateTree(dataSet, labels)
    # t.calcGainRadio('颜色')

    t.run(e)
    printDic(t.tree, 0)

    print('nice')
