from numpy import *
import numpy as np
import pandas as pd
import csv


def sigmod(Xi, beta):  # y=\frac{1}{1+e^{-\beta^{T}x}},返回的是一个浮点数,仅仅用于训练的时候用
    params = -float(np.dot(Xi, beta.T))
    y = (1.0 / (1.0 + math.exp(params)))
    return y


def sigmod_output_matrix(X, beta):  # 输入矩阵返回矩阵(数组)
    params = - np.dot(X, beta.T)
    r = np.zeros(params.shape[0])  # 返回一个np数组
    for i in range(len(r)):
        r[i] = float(1.0 / float((1.0 + math.exp(params[i]))))
    return r


class LogisticClassfier(object):  # 逻辑回归二分类器
    alpha = 0.01  # 梯度下降用的步长
    loop = 1000  # 梯度下降的最多轮次
    m = 0
    accuracy = 0.0001  # 表示代价函数的最小精度，用来防止过拟合
    beta = None

    def __init__(self, learning_rate, loop_num, accuracy_num):  # 一些初始化
        self.alpha = learning_rate
        self.loop = loop_num
        self.accuracy = accuracy_num
        # self.m=shape[0]

    def cost_function(self, y, beta, x_hat):  # 代价函数，书上的3.27
        result = 0.0
        for i in range(self.m):
            temp_matrix = np.dot(x_hat[i], beta.T)
            sum_pre = -y[i] * temp_matrix + log(1 + math.exp(temp_matrix))
            result += sum_pre  # 也许可以优化
        return result

    def fit(self, X, Y):  # X是16个标签构成的矩阵,Y是用于二分类class矩阵，值为0或1
        data_num = X.shape[0]  # 记录有几组训练的数据
        self.m = data_num
        # print("dere")
        feature_num = X.shape[1]  # 记录有几个feature\
        self.beta = np.full(feature_num + 1, 0.5)  # 1*n+1 初始化beta为0.5,增广了一个元素作为b（把原来的wx+b变成了beta*x_hat）

        tempx = np.full((data_num, 1), 1)
        X_hat = np.column_stack((X, tempx))  # m*(n+1) X增广的结果是(x,1),最后一列都是1

        count = 1  # 调试用计数

        # 梯度下降法
        while True:
            old = self.cost_function(Y, self.beta, X_hat)  # 前一个
            print("old", old)
            temp_sum = np.matrix(np.full(feature_num + 1, 0.5))
            for i in range(self.m):  # 也许可以优化

                temp_sum += np.array(np.dot(float((sigmod(X_hat[i], self.beta) - Y[i])), X_hat[i]))
            self.beta = self.beta - np.dot(self.alpha / self.m, temp_sum)
            new = self.cost_function(Y, self.beta, X_hat)  # 更新之后的
            print("new", new)

            if math.fabs(new - old) < self.accuracy:  # 退出方式1：新旧的cost_function小于accuracy参数,用来防止过拟合
                print("Break in LogisticClassfier")
                print("The value of cost function now is :", new)
                break
            print("The time is :", count)
            print("The difference between old and new is :", new - old)
            count += 1
            if count >= self.loop:  # 退出方式2：新旧的cost_function小于loop,用来防止出不去
                print("Break in LogisticClassfier because of loop_num")
                break

    def predict(self, X):
        data_num = X.shape[0]
        tempx = np.full((data_num, 1), 1)
        X_hat = np.column_stack((X, tempx))  # m*(n+1) X增广的结果是(x,1),最后一列都是1
        result = sigmod_output_matrix(X_hat, self.beta)
        print("in result: ")
        print(result)
        print("\n")
        return result


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

    def OVR_train(self, X_train, Y_train):
        chosen_one = 0  # OVR的被检验组
        #  print("in OVR train")
        temp_y = list(Y_train)

        #  print(temp_y)
        # self.classfier[1]=LogisticClassfier()
        feature_num = X_train.shape[1]
        for i in range(1, feature_num + 1):
            print(i)
            chosen_one = i
            # print(chosen_one)
            self.classfier.append(LogisticClassfier(self.alpha, self.loop, self.accuracy))
            # for j in range(Y_train.shape[0]):
            for j in range(Y_train.shape[0]):
                if int(Y_train[j]) == chosen_one:
                    # print("chose", chosen_one)
                    # print("temp", Y_train[j])
                    temp_y[j] = 1
                else:
                    # print("chose",chosen_one)
                    #  print("temp", Y_train[j])
                    temp_y[j] = 0
            # print(temp_y)
            self.classfier[i - 1].fit(X_train, temp_y)
        #  print(temp_y)
        #  print("next\n")
        # temp_y = Y_train

    def OVR_predict(self, X_test):  # 输入test矩阵,输出结果矩阵,运用OVR,用一个二维list记录,比较每一个行的最大值
        data_num = X_test.shape[0]
        feature_num = X_test.shape[1]
        temp_result = np.zeros(
            (feature_num, data_num))  # 我用m*n的矩阵记录每一次预测的结果,n是测试集条数,m是OVR的数量,因为是OVR所以是feature的数量；准备转置之后找每一行的最大值

        for i in range(feature_num):
            temp_result[i] = self.classfier[i].predict(X_test)

        temp_result2 = np.array(temp_result.T)  # 转置矩阵,此时每一行都是对一个测试集的不同OVR的结果,比较出最大值出结果
        result = np.zeros((data_num, 1))
        for i in range(data_num):
            result[i] = int(argmax(temp_result2[i])) + 1

        print("final result:", result)
        return result


def score(my_result, target_result, alpha, loop, accuracy):
    total = my_result.shape[0]
    success = 0
    data_params=[alpha,loop,accuracy]
    dataframe = pd.DataFrame({'my_result': list(my_result), 'target': list(target_result),'data_prams':list(data_params)})
    dataframe.to_csv("result6.csv", index=False, sep=',')
    print(my_result)
    print(target_result)
    for i in range(total):
        if my_result[i] == target_result[i]:
            success += 1
    print("========================\n")
    print("alpha:", alpha, "loop: ", loop, "accuracy: ", accuracy)
    print("Success counts: \n", success)
    print("Success rate: \n", success / total)
    print("========================")


'''
导入训练集，测试集
'''
# X_train=[]
# Y_train=[]

train_value = pd.read_csv('train_set.csv', header=0, sep=",").values
test_value = pd.read_csv('test_set.csv', header=0, sep=",").values
X_train = np.mat(train_value[:, 0:16])  # 保存0-15行数据并转换成矩阵
Y_train = np.mat(train_value[:, 16:17])  # 保存16行数据并转换成矩阵
X_test = np.mat(test_value[:, 0:16])  # 保存0-15行数据并转换成矩阵
Y_test = np.mat(test_value[:, 16:17])  # 保存16行数据并转换成矩阵
# print("here")
alpha = 0.05
loop = 500
accuracy = 0.001
myOVR = OVRclassifier(alpha, loop, accuracy)
# print("here1")
myOVR.OVR_train(X_train, Y_train)
result = myOVR.OVR_predict(X_test)
score(result, Y_test,alpha, loop, accuracy)
# m = X_train.shape[0]
# n = X_train.shape[1]
