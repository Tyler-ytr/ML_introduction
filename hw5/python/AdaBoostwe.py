import random
import numpy as np
import pandas as pd
from sklearn import (metrics, tree, preprocessing)
from sklearn.model_selection import train_test_split


def extract_data():
    adult_header = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex",
                    "capital-gain", "capital-loss", "hours-per-week", "native-country", "label"]
    adult_data = pd.read_csv("./adult/adult.data",
                             index_col=False, names=adult_header)
    adult_test = pd.read_csv("./adult/adult.test",
                             index_col=False, names=adult_header)

    adult_data[adult_data == ' ?'] = np.nan
    adult_test[adult_test == ' ?'] = np.nan
    adult_data.dropna(axis=0, how='any', inplace=True)
    adult_test.dropna(axis=0, how='any', inplace=True)

    discre_name = ["workclass", "education", "education-num", "marital-status",
                   "occupation", "relationship", "race", "sex", "native-country", "label"]
    for name in discre_name:
        key = np.unique(adult_data[name])
        le = preprocessing.LabelEncoder()
        le.fit(key)
        adult_test[name] = le.transform(adult_test[name])
        adult_data[name] = le.transform(adult_data[name])

    adult_data.loc[adult_data.label == 0, 'label'] = -1
    adult_test.loc[adult_test.label == 0, 'label'] = -1

    data = np.vstack((adult_data, adult_test))
    X = data[:, 0:-1]
    y = data[:, -1]
    return X, y


"""
    X_train = adult_data.iloc[:, 0:-1]
    y_train = adult_data.iloc[:, -1]
    X_test = adult_test.iloc[:, 0:-1]
    y_test = adult_test.iloc[:, -1]
    return X_train, y_train, X_test, y_test
"""


class Boosting(object):

    def __init__(self, X_train, y_train, T=1500, min_samples_split=2):
        self.X_train = X_train
        self.N = self.X_train.shape[0]
        self.y_train = y_train
        self.weights = np.ones(self.N)/self.N
        self.epsilont = []
        self.alphas = []
        self.classifiers = []
        self.num_estimators = T
        self.min_samples_split = min_samples_split
        self.max_depth = self.X_train.shape[1]

    def doBoosting(self):
        errorThreshold = 0.0001
        epsilon = 0.001
        for t in range(self.num_estimators):
            clf = tree.DecisionTreeClassifier(
                criterion='gini', min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            clf = clf.fit(X_train, y_train,
                          sample_weight=np.array(self.weights))
            Y_pred = clf.predict(X_train)

            e_t = np.sum((Y_pred != self.y_train) * self.weights)

            if e_t > 0.5:
                print("T = %d >0.5 break" % t)
                break
            if e_t < errorThreshold:
                print("T = %d <errorThreshold break" % t)

            self.epsilont.append(e_t)

            alpha_t = 0.5 * np.log((1 - e_t + epsilon)/(e_t+epsilon))
            self.alphas.append(alpha_t)
            self.classifiers.append(clf)

            self.weights = self.weights * \
                np.exp(-alpha_t * Y_pred * self.y_train)
            self.weights = self.weights / np.sum(self.weights)

            #print("T_%d: " % t, end='')
            # print(self.alphas[t])

        #print("train finish!")

    def predict(self, X_test):
        Y_pred = np.zeros(X_test.shape[0])
        for i in range(len(self.alphas)):
            clf = self.classifiers[i]
            y_tmp = clf.predict(X_test)
            Y_pred += self.alphas[i] * y_tmp

        Y_pred[Y_pred >= 0] = 1
        Y_pred[Y_pred < 0] = -1
        return Y_pred


if __name__ == '__main__':
    # [X_train, y_train, X_test, y_test] = extract_data()
    X, y = extract_data()
    turns = 10
    # T_test = [27, 31, 40, 55, 70, 100, 130, 160, 200]

    for T in range(1, 51):
        SEED = T
        mean_auc = 0.0
        for i in range(turns):
            X_train, X_cv, y_train, y_cv = train_test_split(
                X, y, test_size=.20, random_state=i*SEED)
            Adab = Boosting(X_train, y_train, T)
            Adab.doBoosting()
            Y_pred = Adab.predict(X_cv)
            fpr, tpr, thresholds = metrics.roc_curve(y_cv, Y_pred)
            roc_auc = metrics.auc(fpr, tpr)
            mean_auc += roc_auc

        print("T = %d: %f" % (T, mean_auc / turns))
