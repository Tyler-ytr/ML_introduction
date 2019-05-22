#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

# from compiler.ast import flatten
from random import Random
from pandas import DataFrame
from numpy import log
from numpy import mat
from numpy import ones
from sklearn import metrics
from sklearn import tree
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as pl
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


# In[2]:


class randomforest(object):
    n_estimators=0 # 树的数量
    max_features=0 #每棵树的选用数据集的最大特征数
    min_samples_split=0 #每棵树最小分割数
    min_gain=0 #每一颗树到min_gain之后就停止
    max_depth=0 #每一颗树的最大层数
    trees=[] #森林
    trees_feature=[] #用来记录每一个树用了哪些特征
    
    def __init__(self,n_estimators=100,min_samples_split=3, min_gain=0,
                 max_depth=None,max_features=None):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.max_depth = max_depth
        self.max_features = max_features 
        
#         #建立森林(bulid forest)
#         for _ in range(self.n_estimators):
#             tree =DecisionTreeClassifier(min_samples_split=self.min_samples_split, min_impurity_split = self.min_gain,
#                                       max_depth=self.max_depth)
#             self.trees.append(tree)
            #self.trees_feature.append(0)
        
    def get_bootstrap_data(self,X,Y):
        # 用bookstarp的方法获得n_estimators组随机的数据
        
        m=X.shape[0]
        Y=Y.reshape(m,1)
        
        #合并X和Y
        X_Y=np.hstack((X,Y))
        np.random.shuffle(X_Y) #X_Y随机化
        
        result_sets=[]
        for _ in range(self.n_estimators):
            now=np.random.choice(m,m,replace=True) #有放回,随机序列顺序
            bootstrap_X_Y = X_Y[now,:]
            bootstrap_X =  bootstrap_X_Y[:,:-1]
            bootstrap_Y =  bootstrap_X_Y[:,-1:]
            result_sets.append([bootstrap_X,bootstrap_Y])
            
        return result_sets
    
    def fit(self,X_train,Y_train):
        # 每一颗树都通过get_bookstrap_data获得随机的数据集
        
        sub_sets=self.get_bootstrap_data(X_train,Y_train)
        n_features=X_train.shape[1]
        
        if self.max_features == None:
            self.max_features = int(np.sqrt(n_features))

        for i in range (self.n_estimators):
            # 现在为每一颗树选择随机的特征
            tree =DecisionTreeClassifier(min_samples_split=self.min_samples_split
                                         ,min_impurity_decrease = self.min_gain,max_depth=self.max_depth)
            
            sub_X,sub_Y=sub_sets[i]
            features=np.random.choice(n_features,self.max_features,replace=True)
            sub_X=sub_X[:,features]
            #print("X",sub_X)
            #print("X",sub_Y)
            tree.fit(sub_X,sub_Y)
            self.trees.append(tree)
            self.trees_feature.append(features)
    
    def predict(self,X):
        y_preds=[]
        for i in range(self.n_estimators):
            features=self.trees_feature[i]
            sub_X=X[:,features]
            y_pre=self.trees[i].predict(sub_X)
            y_preds.append(y_pre)
        
        y_preds=np.array(y_preds).T
        #print(y_preds)
        y_pred=[]
        
        for y_p in y_preds:
            #np.bincount()可以统计每个索引出现的次数,np.argmax()可以返回数组中最大值的索引
            #方案一 获得众数
            y_pred.append(np.bincount(y_p.astype('int')).argmax()) 
        #方案二 获得平均值
        y_pred=np.mean(y_preds,axis=1)
        #print("2",mat(y_pred).shape)
        return y_pred
        
        
            
            


# In[3]:


def get_data():
    adult_header=["age","workclass","fnlwgt","education","education-num",
              "marital-status","occupation","relationship","race","sex","capital-gain",
              "capital-loss","hours-per-week","native-country","label"]
    adult_data=pd.read_csv("./adult.data",index_col=False,names=adult_header)
    adult_test=pd.read_csv("./adult2.test",index_col=False,names=adult_header)
    #adult_data.shape
    #adult2.test的数据是adult的数据删去label的最后一个字符"."得到的;
    adult_data[adult_data==" ?"]=np.nan
    adult_test[adult_test==" ?"]=np.nan
    #print(adult_data["age"])
    adult_data.dropna(axis=0,how='any',inplace=True)
    adult_test.dropna(axis=0,how='any',inplace=True)
    discre_name=["workclass","education","marital-status",
             "occupation","relationship","race",
             "sex","native-country","label"]
    for name in discre_name:
        key=np.unique(adult_data[name])
        #print(key)
        le=preprocessing.LabelEncoder()
        le.fit(key)
        adult_test[name]=le.transform(adult_test[name])
        adult_data[name]=le.transform(adult_data[name])
    
    data = np.vstack((adult_data, adult_test))
    X = data[:, 0:-1]
    Y = data[:, -1]
    return X, Y
#print(adult_data)


# In[ ]:


# adult_header=["age","workclass","fnlwgt","education","education-num",
#               "marital-status","occupation","relationship","race","sex","capital-gain",
#               "capital-loss","hours-per-week","native-country","label"]
# adult_data=pd.read_csv("./adult.data",index_col=False,names=adult_header)
# adult_test=pd.read_csv("./adult2.test",index_col=False,names=adult_header)


# In[ ]:


# adult_data[adult_data==" ?"]=np.nan
# adult_test[adult_test==" ?"]=np.nan
# #print(adult_data["age"])
# adult_data.dropna(axis=0,how='any',inplace=True)
# adult_test.dropna(axis=0,how='any',inplace=True)


# In[ ]:


# discre_name=["workclass","education","marital-status",
#              "occupation","relationship","race",
#              "sex","native-country","label"]
# for name in discre_name:
#     key=np.unique(adult_data[name])
#     #print(key)
#     le=preprocessing.LabelEncoder()
#     le.fit(key)
#     adult_test[name]=le.transform(adult_test[name])
#     adult_data[name]=le.transform(adult_data[name])
# #print(adult_data)


# In[ ]:


# # X_data=adult_data.values[0:150,0:14]
# # #print(X_data[:,0:14])
# # Y_data=adult_data.values[0:150,14]
# # #print(Y_data)
# # X_test=adult_test.values[0:150,0:14]
# # #print(X_test[:,0:14])
# # Y_test=adult_test.values[0:150,14]
# # print(Y_test)
# # #记得删除
# X_data=np.array(adult_data.values[:,0:14])
# #print(X_data[:,0:14])0
# Y_data=np.array(adult_data.values[:,14])
# #print(Y_data)
# X_test=np.array(adult_test.values[:,0:14])
# #print(X_test[:,0:14])
# Y_test=np.array(adult_test.values[:,14])
# X=np.vstack((X_data,X_test))
# Y=np.hstack((Y_data,Y_test))
X,Y=get_data()
test_num=5
plot_T=[]
plot_auc=[]
for t in range(1,10):
    serand=t
    mean_auc=0.0
    mean_score=0.0
    for i in range(test_num):
        X_data,X_test,Y_data,Y_test=train_test_split(
                X, Y, test_size=.20, random_state=i*serand)
        Random_classfier=randomforest(t) #参数记得填
        Random_classfier.fit(X_data,Y_data)
        Y_pred=mat(Random_classfier.predict(X_test))
        #y_pred=mat(Adaboost_classfier.predict(X_test))
        Y_pred.astype(np.int)
        Y_pred=np.array(np.ravel(Y_pred))
        Y_true=np.array(Y_test)
        precision, recall, thresholds = precision_recall_curve(Y_true,Y_pred)
        pr_auc = auc(recall, precision)
        mean_auc+=pr_auc
#         score=accuracy_score(Y_test, np.array(Y_pred))
#         mean_score+=score
    plot_T.append(t)
    plot_auc.append(mean_auc/test_num)
    print("Randomforest: T=%d: auc=%f "%(t,(mean_auc/test_num)))
pl.plot(plot_T,plot_auc)
kf = KFold(n_splits=5,random_state=0)

for train_index, test_index in kf.split(X):
    #print(train_index,test_index)
    X_data=X[train_index]
    X_test=X[test_index]
    Y_data=Y[train_index]
    Y_test=Y[test_index]
    break;


# In[ ]:


# a1 = np.random.choice(9,1,replace=False, p=None)
# print(a1)


# In[ ]:



# Y=np.random.rand(5,1)
# print(Y)
# X=np.random.rand(5,9)
# m=X.shape[0]
# print(X)
# X_Y = np.hstack((X,Y))
# np.random.shuffle(X_Y)

# data_sets = []
# for _ in range(1):
#     idm = np.random.choice(m,m,replace=True)
#     bootstrap_X_Y = X_Y[idm,:]
#     bootstrap_X =  bootstrap_X_Y[:,:-1]
#     bootstrap_Y =  bootstrap_X_Y[:,-1:]
#     data_sets.append([bootstrap_X,bootstrap_Y])

# print(data_sets)
    


# In[ ]:


Random_classfier=randomforest() #参数记得填
# X_data.replace(np.nan, 0, inplace=True)
Random_classfier.fit(X_data,Y_data)
y_pred=Random_classfier.predict(X_test)
print(y_pred)
y_true=np.array(Y_test)
precision, recall, thresholds = precision_recall_curve( y_true,y_pred)
score=accuracy_score(y_true, y_pred)
print(score)
pr_auc = auc(recall, precision)
test_auc =metrics.roc_auc_score(y_true, y_pred)#验证集上的auc值
pl.plot(recall, precision)
print("auc",roc_auc_score(y_true, y_pred))
print("auc",test_auc)


# In[ ]:





# In[ ]:




