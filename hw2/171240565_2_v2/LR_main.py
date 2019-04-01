from numpy import *
import numpy as np
import pandas as pd
import csv


def gradient(beta, X, Y):  # 计算二分类逻辑回归的梯度
    beta_X_t = np.dot(X, beta)
    a = -np.dot(X.T, (Y.T - 1 / (1 + np.exp(-beta_X_t))))
    return a


def sigmod_fit_matrix(X, beta):  # 输入矩阵，计算sigmod,返回矩阵(数组)
    params = np.matrix(- np.dot(X, beta.T))
    y = np.matrix((1.0 / (1.0 + np.exp(params))))
    return y


def sigmod_output_matrix(X, beta):  # 输入矩阵,计算sigmod,返回矩阵(数组)
    params = np.matrix(- np.dot(X, beta))
    y = []
    y = np.matrix((1.0 / (1.0 + np.exp(params))))
    return y


class LogisticClassfier(object):  # 逻辑回归二分类器
    alpha = 0.01  # 梯度下降用的步长
    loop = 1000  # 梯度下降的最多轮次
    m = 0  # 表示数据集的数量
    accuracy = 0.0001  # 表示代价函数的最小精度，用来在利用cost_function预测结果的情况下，防止过拟合
    beta = None

    def __init__(self, learning_rate, loop_num, accuracy_num):  # 一些初始化
        self.alpha = learning_rate
        self.loop = loop_num
        self.accuracy = accuracy_num

    def cost_function(self, y, beta, x_hat):  # 代价函数，书上的3.27
        result = 0.0
        for i in range(self.m):
            temp_matrix = np.dot(x_hat[i], beta)
            sum_pre = -y[i] * temp_matrix + log(1 + math.exp(temp_matrix))
            result += sum_pre
        return result

    def fit(self, X, Y):  # X是16个标签构成的矩阵,Y是用于二分类class矩阵，值为0或1
        data_num = X.shape[0]  # 记录有几组训练的数据
        self.m = data_num
        feature_num = X.shape[1]  # 记录有几个feature
        self.beta = np.full((feature_num + 1, 1), 0.5)  # 1*n+1 初始化beta为0.5,增广了一个元素作为b（把原来的wx+b变成了beta*x_hat）
        tempx = np.full((data_num, 1), 1)
        X_hat = np.column_stack((X, tempx))  # m*(n+1) X增广的结果是(x,1),最后一列都是1
        count = 1  # 调试用计数

        # 梯度下降法
        while True:
            # 用于调参,计算最后两轮的cost_function的更新前的值
            if count > 1 * self.loop +1: # 我在测试的时候 是 count> 1*self.loop-2
             #   print("The time is :", count)
                old = self.cost_function(Y, self.beta, X_hat)  # 前一个
                print("old", old)

            # 更新每一次的beta矩阵
            self.beta = self.beta - self.alpha / self.m * gradient(self.beta, X_hat, matrix(Y))

            # 用于调参,计算最后两轮的cost_function的更新后的值
            # if count > 1 * self.loop +1:# 我在测试的时候 是 count> 1*self.loop-2
            #     new = self.cost_function(Y, self.beta, X_hat)
            #    # print("new", new)
            #   #  print("The difference between old and new is :", new - old)
            #     if math.fabs(
            #             new - old) < self.accuracy:  # 退出方式1：新旧的cost_function小于accuracy参数,用来防止过拟合。如果要使用这种退出方式，需要修改前面的if条件
            #       #  print("Break in LogisticClassfier")
            #     #    print("The value of cost function now is :", new)
            #         break
            # if count % 1000 == 0:  # 用于计数
            #      print("The time is :", count)

            if count > self.loop:  # 退出方式2：计数大于loop
               # print("Break in LogisticClassfier because of loop_num")
                break
            count += 1

    # 用于利用已经算出来的beta矩阵预测结果
    def predict(self, X):
        data_num = X.shape[0]
        tempx = np.full((data_num, 1), 1)
        X_hat = np.column_stack((X, tempx))  # m*(n+1) X增广的结果是(x,1),最后一列都是1
        result = sigmod_output_matrix(X_hat, self.beta)
        return result


# OVR方法实现的多分类器

class OVRclassifier(object):
    classfier = []
    predict_result = []
    result = []
    alpha = 0.01  # 梯度下降用的步长
    loop = 1000  # 梯度下降的最多轮次
    accuracy = 0.0001  # 表示代价函数的最小精度，用来防止过拟合

    def __init__(self, alpha_num, loop_num, accuracy_num):
        self.alpha = alpha_num
        self.loop = loop_num
        self.accuracy = accuracy_num

    # 首先按顺序选择代比较的组(记作chosen_one),然后初始化class序列(记作temp_y),然后丢给二分类器训练
    def OVR_train(self, X_train, Y_train):
        temp_y = list(Y_train)
        for i in range(1, 26 + 1):
            print("It is the ",i,"th LogisticClassfier! Patience is a good virtue.")  # 告知目前比较到第几组了
            chosen_one = i
            self.classfier.append(LogisticClassfier(self.alpha, self.loop, self.accuracy))  # 初始化相应的二分类器
            for j in range(Y_train.shape[0]):  # 用来将原来的class转化成0-1的形式
                if int(Y_train[j]) == chosen_one:
                    temp_y[j] = 1
                else:
                    temp_y[j] = 0
            self.classfier[i - 1].fit(X_train, temp_y)  # 训练相应的二分类器

    def OVR_predict(self, X_test):  # 输入test矩阵,输出结果矩阵,运用OVR；
        data_num = X_test.shape[0]
        feature_num = 26
        temp_result = np.zeros(
            (feature_num, data_num))  # 我用m*n的矩阵记录每一次预测的结果,n是测试集条数,m是OVR的数量,因为是OVR所以是feature的数量；准备转置之后找每一行的最大值

        for i in range(feature_num):
            temp_result[i] = self.classfier[i].predict(X_test).T

        temp_result2 = np.array(temp_result.T)  # 转置矩阵,此时每一行都是对一个测试集的不同OVR的结果,比较出最大值出结果
        return temp_result2


# 用于输出评估结果,并且将预测的结果用csv存下来(想要存下来请解除注释)
# 主要方法:
# 1.将之前得到的分数化结果0-1化表示,并且将目标值0-1化表示,都表示成(数据大小*class数)的混淆矩阵,然后进行比较求出TP,FP,FN
# 2.求出之前分数化结果的26个分类器的最大值,用该最大值的标签作为最后的OVR预测值
def score(result, target_result, alpha, loop, accuracy):  # result: 6000*26(数据大小*class数) 的一个矩阵
    total = result.shape[0]

    my_result = np.zeros((total, 1))  # 用于记录最大预测结果,是一个数据大小*1的矩阵
    target = np.zeros([result.shape[0], result.shape[1]])  # 预处理初始化,大小与result矩阵一样大
    standard_result = np.zeros([result.shape[0], result.shape[1]])  # 用来把之前的分数矩阵0-1化表示,用来计算查准率查全率等

    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            if j == target_result[i] - 1:
                target[i][j] = 1
            else:
                target[i][j] = 0
    # print(target)

    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            if result[i][j] > 0.5:
                standard_result[i][j] = 1
            else:
                standard_result[i][j] = 0
    # print("result:", macro_result)
    confusion_TP = np.full([target.shape[1], 1], 0.00000001)
    confusion_FP = np.full([target.shape[1], 1], 0.00000001)
    confusion_FN = np.full([target.shape[1], 1], 0.00000001)

    # 计算混淆矩阵
    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            if standard_result[i][j] == 1 and target[i][j] == 1:
                confusion_TP[j] += 1
            if standard_result[i][j] == 1 and target[i][j] == 0:
                confusion_FP[j] += 1
            if standard_result[i][j] == 0 and target[i][j] == 1:
                confusion_FN[j] += 1
    # print("confusion:", confusion_TP)
    macro_P = 0
    macro_R = 0

    micro_TP = np.mean(confusion_TP)
    micro_FP = np.mean(confusion_FP)
    micro_FN = np.mean(confusion_FN)

    for i in range(target.shape[1]):
        macro_P += float(confusion_TP[i]) / float((confusion_TP[i] + confusion_FP[i]))
        macro_R += float(confusion_TP[i]) / float((confusion_TP[i] + confusion_FN[i]))
    macro_P = macro_P / target.shape[1]
    macro_R = macro_R / target.shape[1]
    macro_F1 = 2 * macro_P * macro_R / (macro_P + macro_R)
    micro_P = micro_TP / (micro_TP + micro_FP)
    micro_R = micro_TP / (micro_TP + micro_FN)
    micro_F1 = 2 * micro_P * micro_R / (micro_P + micro_R)



    # 将预测结果保存为csv文件
    # dataframe = pd.DataFrame({'my_result': list(my_result), 'target': list(target_result)})
    # dataframe.to_csv("0.09_test_50000.csv", index=False, sep=',')

    # 求出之前分数化结果的26个分类器的最大值,用该最大值的标签作为最后的OVR预测值
    for i in range(total):
        my_result[i] = int(argmax(result[i])) + 1
    success = 0 # 记录命中次数
    for i in range(total):
        if my_result[i] == target_result[i]:
            success += 1

    print("========================")
    print("alpha:", alpha, "loop: ", loop)
    # print("Success counts: \t", success)
    print("accuracy: \t", '{:.2%}'.format(success / total))
    print("micro Precision: \t", '{:.2%}'.format(micro_P))
    print("micro Recall: \t", '{:.2%}'.format(micro_R))
    print("micro F1: \t", '{:.2%}'.format(micro_F1))
    print("macro Precision: ", '{:.2%}'.format(macro_P))
    print("macro Recall: \t", '{:.2%}'.format(macro_R))
    print("macro F1:  \t", '{:.2%}'.format(macro_F1))
    print("========================")


'''
导入训练集，测试集
'''

train_value = pd.read_csv('train_set.csv', header=0, sep=",").values
test_value = pd.read_csv('test_set.csv', header=0, sep=",").values

X_train = np.mat(train_value[:, 0:16])  # 保存0-15行数据并转换成矩阵
Y_train = np.mat(train_value[:, 16:17])  # 保存16行数据并转换成矩阵
X_test = np.mat(test_value[:, 0:16])  # 保存0-15行数据并转换成矩阵
Y_test = np.mat(test_value[:, 16:17])  # 保存16行数据并转换成矩阵
# 设定步长,轮次,cost_function最低精度值
print("这是我的多分类器，如果想要获得一个70%左右的精度,请把下面的alpha调成0.09,loop调成1000")
alpha = 0.09
loop = 1000
accuracy = 0.0001
myOVR = OVRclassifier(alpha, loop, accuracy)
myOVR.OVR_train(X_train, Y_train)
result1 = myOVR.OVR_predict(X_test)
# result2 = myOVR.OVR_predict(X_train)
score(result1, Y_test, alpha, loop, accuracy)
# score(result2, Y_train, alpha, loop, accuracy)
