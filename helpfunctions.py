#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 18:08:22 2020

@author: a810en
"""

import fire
import os
import statistics
import sys
import numpy as np
import pandas as pd

from fairness import results
from fairness.data.objects.list import DATASETS, get_dataset_names
from fairness.data.objects.ProcessedData import ProcessedData
from fairness.benchmark import run_alg
from sklearn.metrics import accuracy_score

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris

from fairness.algorithms.zafar.ZafarAlgorithm import ZafarAlgorithmBaseline, ZafarAlgorithmAccuracy, \
    ZafarAlgorithmFairness
from fairness.algorithms.kamishima.KamishimaAlgorithm import KamishimaAlgorithm
from fairness.algorithms.kamishima.CaldersAlgorithm import CaldersAlgorithm
from fairness.algorithms.feldman.FeldmanAlgorithm import FeldmanAlgorithm
from fairness.algorithms.baseline.SVM import SVM
from fairness.algorithms.baseline.DecisionTree import DecisionTree
from fairness.algorithms.baseline.GaussianNB import GaussianNB
from fairness.algorithms.baseline.LogisticRegression import LogisticRegression
from fairness.algorithms.ParamGridSearch import ParamGridSearch
from fairness.algorithms.Ben.SDBSVM import SDBSVM
import torch

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import shuffle
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow.compat.v1 as tf

from torch.autograd import Variable


def p_rule(y_pred, z_values, threshold=0.5):
    y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
    y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]
    odds = y_z_1.mean() / y_z_0.mean()
    return np.min([odds, 1 / odds]) * 100


class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))


def DispFNR(y_pred, y, z_values, threshold=0.5):
    ypred_z_1 = y_pred > threshold if threshold else y_pred[z_values == 1]
    ypred_z_0 = y_pred > threshold if threshold else y_pred[z_values == 0]
    result = abs(ypred_z_1[(y == 1) & (z_values == 0)].mean() - ypred_z_1[(y == 1) & (z_values == 1)].mean())
    return result


def DispFPR(y_pred, y, z_values, threshold=0.5):
    ypred_z_1 = y_pred > threshold if threshold else y_pred[z_values == 1]
    ypred_z_0 = y_pred > threshold if threshold else y_pred[z_values == 0]
    result = abs(ypred_z_1[(y == 0) & (z_values == 0)].mean() - ypred_z_1[(y == 0) & (z_values == 1)].mean())
    return result


def DI(y_pred, z_values, threshold=0.5):
    y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
    y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]
    odds = abs(y_z_1.mean() - y_z_0.mean())
    return odds


def display_results(y_pred, y, sensitive):
    y_pred2 = (y_pred>0.5).astype(int)
    accuracy = accuracy_score(y, np.squeeze(y_pred2))
    balanced_accuracy = balanced_accuracy_score(y, np.squeeze(y_pred2))
    roc_score = roc_auc_score(y, y_pred)
    print("Accuracy:", accuracy)
    print("Balanced Accuracy:", balanced_accuracy)
    print("ROC Score:", roc_score)
    print("PRULE : ", p_rule(y_pred,sensitive))
    print("DI : ", DI(y_pred, sensitive))
    print("DispFPR : ", DispFPR(y_pred2, y, sensitive))
    print("DispFNR : ", DispFNR(y_pred2, y, sensitive))
    return {'Accuracy': accuracy, 'Balanced Accuracy': balanced_accuracy, 'ROC Score': roc_score, 'PRULE': p_rule(y_pred,sensitive), 'DI': DI(y_pred, sensitive), 'DispFPR': DispFPR(y_pred2, y, sensitive)
            ,'DispFNR': DispFNR(y_pred2, y, sensitive)}


# def DATA_TRAIN_TEST(num, sens, y, columns_delete):
#     dataset = DATASETS[num]  # Adult data set
#     all_sensitive_attributes = dataset.get_sensitive_attributes_with_joint()
#     ProcessedData(dataset)
#     processed_dataset = ProcessedData(dataset)
#     train_test_splits = processed_dataset.create_train_test_splits(1)
#     train_test_splits.keys()
#     X_train, X_test = train_test_splits['numerical-binsensitive'][0]
#     sensitive = X_train[sens].values
#     sensitivet = X_test[sens].values
#     y_train = X_train[y]
#     y_test = X_test[y]
#
#     scaler = StandardScaler().fit(X_train)
#     s = X_train[sens]
#     st = X_test[sens]
#     t = X_train[y]
#     tt = X_test[y]
#     scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
#     X_train = X_train.pipe(scale_df, scaler)
#     X_test = X_test.pipe(scale_df, scaler)
#     X_train = X_train.drop([sens, y], axis=1)
#     X_train[sens] = s
#     X_train[y] = t
#     X_test = X_test.drop([sens, y], axis=1)
#     X_test[sens] = st
#     X_test[y] = tt
#
#     # X_train = X_train.drop(columns_delete,1)
#     X_train = X_train.drop(columns_delete, axis=1)
#     # X_test = X_test.drop(columns_delete,1)
#     X_test = X_test.drop(columns_delete, axis=1)
#
#     return X_train, X_test, y_train, y_test, sensitive, sensitivet


def p_rule(y_pred, z_values, threshold=0.5):
    y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
    y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]
    odds = y_z_1.mean() / y_z_0.mean()
    return np.min([odds, 1 / odds]) * 100


def lossgr(y, p):
    # Avoid division by zero
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return - y * np.log(p) - (1 - y) * np.log(1 - p)


class FAGTB(object):

    def __init__(self, n_estimators, learning_rate, min_samples_split,
                 min_impurity, max_depth, max_features, regression):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.regression = regression
        self.max_features = max_features

        # Initialize regression trees
        self.trees = []
        self.clfs = []
        self.lossfunction_adv = []
        #self.lfadv = []
        self.losstraining = []
        #self.lossglobal = []

        for _ in range(n_estimators):
            tree = DecisionTreeRegressor(criterion='friedman_mse', max_depth=9,
                                         max_features=self.max_features, max_leaf_nodes=None,
                                         min_impurity_decrease=0.0,  # min_impurity_split=None,
                                         min_samples_leaf=1, min_samples_split=2,
                                         min_weight_fraction_leaf=0.0
                                         , random_state=0)
            self.trees.append(tree)
            clf = LogisticRegression()
            self.clfs.append(clf)
            self.model = []

    def fit2(self, X, y, sensitive, LAMBDA):
        clf = LogisticRegression()
        clf._initialize_parameters(sensitive)
        # print(clf.param)

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)

    def build_graph(self, input_size, seed, learning_rate2):
        graph = tf.Graph()
        with graph.as_default():
            X_input = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name='X_input')
            y_input = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y_input')

            W1 = tf.Variable(tf.random_normal(shape=[input_size, 1], seed=seed), name='W1', trainable=True)
            new_W1 = tf.placeholder(tf.float32, shape=[input_size, 1])  # Define a placeholder for the new value
            assign_W1 = tf.assign(W1, new_W1)  # Create an additional assign operation

            b1 = tf.Variable(tf.random_normal(shape=[1], seed=seed), name='b1', trainable=True)
            new_b1 = tf.placeholder(tf.float32, shape=[1]) # Define a placeholder for the new value
            assign_b1 = tf.assign(b1, new_b1)  # Create an additional assign operation

            sigm = tf.nn.sigmoid(tf.add(tf.matmul(X_input, W1), b1), name='pred')
            logit = tf.add(tf.matmul(X_input, W1), b1)
            loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_input, logits=logit, name='loss'))
            train_steps = tf.train.GradientDescentOptimizer(learning_rate2).minimize(loss)

            sigm2 = tf.cast(sigm, tf.float32, name='sigm2')  # 1 if >= 0.5
            pred = tf.cast(tf.greater_equal(sigm, 0.5), tf.float32, name='pred')  # 1 if >= 0.5
            acc = tf.reduce_mean(tf.cast(tf.equal(pred, y_input), tf.float32), name='acc')

            init_var = tf.global_variables_initializer()
            var_grad = tf.gradients(loss, X_input)[0]

        return (graph, X_input, y_input, sigm, logit, loss, train_steps, sigm2, pred, acc, init_var, var_grad,
                W1, b1, assign_W1, assign_b1, new_W1, new_b1)

    def initialize_session(self, graph, init_var):
        sess = tf.Session(graph=graph)
        sess.run(init_var)
        return sess

    def fit_adversial(self, sess, graph, X_input, y_input, sigm2, var_grad, train_steps, y_pred, sensitive, W1, b1, loss, acc, sigm, logit):
        y2 = np.expand_dims(sensitive, axis=1)
        y_pred2 = np.expand_dims(1 / (1 + np.exp(-y_pred)), axis=1)
        train_feed_dict = {X_input: y_pred2, y_input: y2}

        inital_weight = sess.run(W1)
        inital_bias = sess.run(b1)
        #print(f'inital_weight {inital_weight}')
        #print(f'inital_bias {inital_bias}')

        sess.run(train_steps, feed_dict=train_feed_dict)  # Update the adversarial classifier

        first_weight = sess.run(W1)
        first_bias = sess.run(b1)
        #print(f'first_weight {first_weight}')
        #print(f'first_bias {first_bias}')

        cur_loss = sess.run(loss, feed_dict=train_feed_dict)
        #print(f'cur_loss {cur_loss}')
        train_acc = sess.run(acc, feed_dict=train_feed_dict)
        #print(f'train_acc {train_acc}')

        S_ADV = sess.run(sigm2, feed_dict=train_feed_dict)
        #print(f'S_ADV {S_ADV}')
        gradient_adv = sess.run(var_grad, feed_dict=train_feed_dict)
        #print(f'gradient_adv {gradient_adv}')

        if abs(np.sum(gradient_adv)) < 0.001:
            print('Gradient error')

        lfadv = gradient_adv * y_pred2 * (1 - y_pred2)
        #print(f'lfadv {lfadv}')

        return {'S_ADV': S_ADV, 'gradient_adv': gradient_adv, 'lfadv': lfadv}

    def update_adversial(self, sess, graph, X_input, y_input, sigm2, var_grad, train_steps, y_pred, sensitive, W1, b1, loss, acc):
        valueofw = sess.run(W1)
        valueofb = sess.run(b1)
        #print(f'value of W1 before update {valueofw}')
        #print(f'value of b1 before update {valueofb}')
        y2 = np.expand_dims(sensitive, axis=1)
        y_pred2 = np.expand_dims(1 / (1 + np.exp(-y_pred)), axis=1)
        train_feed_dict = {X_input: y_pred2, y_input: y2}

        sess.run(train_steps, feed_dict=train_feed_dict)  # Update the adversarial classifier
        updated_weights = sess.run(W1)
        updated_bias = sess.run(b1)
        #print(f'w1 {updated_weights}')
        #print(f'b1 {updated_bias}')

        valueofw = sess.run(W1)
        valueofb = sess.run(b1)
        #print(f'value of W1 after update {valueofw}')
        #print(f'value of b1 after update {valueofb}')

        cur_loss = sess.run(loss, feed_dict=train_feed_dict)
        #print(f'cur_loss {cur_loss}')
        train_acc = sess.run(acc, feed_dict=train_feed_dict)
        #print(f'train_acc {train_acc}')

        return {'sess': sess, 'W1_updated': updated_weights, 'b1_updated': updated_bias, 'W1': W1, 'b1': b1}

    def update_adv_gradient(self, sess, graph, X_input, y_input, sigm2, var_grad, train_steps, y_pred, sensitive, W1, b1, new_W, new_b, assign_W1, assign_b1, new_W1, new_b1, loss, acc):
        valueofw = sess.run(W1)
        valueofb = sess.run(b1)
        #print(f'value of W1 before change to global W1 {valueofw}')
        #print(f'value of b1 before change to global b1 {valueofb}')

        # Update W with a new value
        new_value_for_W1 = new_W
        sess.run(assign_W1, feed_dict={new_W1: new_value_for_W1})

        # Update b with a new value
        new_value_for_b1 = new_b
        sess.run(assign_b1, feed_dict={new_b1: new_value_for_b1})

        valueofw = sess.run(W1)
        valueofb = sess.run(b1)
        #print(f'value of W1 after change to global W1 {valueofw}')
        #print(f'value of b1 after change to global b1 {valueofb}')

        y2 = np.expand_dims(sensitive, axis=1)
        y_pred2 = np.expand_dims(1 / (1 + np.exp(-y_pred)), axis=1)
        train_feed_dict = {X_input: y_pred2, y_input: y2}

        cur_loss = sess.run(loss, feed_dict=train_feed_dict)
        #print(f'cur_loss {cur_loss}')
        train_acc = sess.run(acc, feed_dict=train_feed_dict)
        #print(f'train_acc {train_acc}')

        # sess.run(train_steps, feed_dict=train_feed_dict)  # Update the adversarial classifier
        S_ADV = sess.run(sigm2, feed_dict=train_feed_dict)
        gradient_adv = sess.run(var_grad, feed_dict=train_feed_dict)

        valueofw = sess.run(W1)
        valueofb = sess.run(b1)
        #print(f'value of W1 after everything else {valueofw}')
        #print(f'value of b1 after everything else {valueofb}')

        if abs(np.sum(gradient_adv)) < 0.001:
            print('Gradient error')

        lfadv = gradient_adv * y_pred2 * (1 - y_pred2)

        return {'S_ADV': S_ADV, 'gradient_adv': gradient_adv, 'lfadv': lfadv, 'W1': W1, 'b1': b1}


    def fit_initial(self, X, y, sensitive, LAMBDA, Xtest, yt, sensitivet):
        y2 = np.expand_dims(sensitive, axis=1)

        lfadv = 0

        self.Init = np.log(np.sum(y) / np.sum(1 - y))

        y_pred2 = np.full(np.shape(y), self.Init)
        y_pred = np.full(np.shape(y), self.Init)
        y_predt = np.full(np.shape(yt), self.Init)
        t = np.full(np.shape(y), 0)
        t2 = np.full(np.shape(yt), 0)
        self.LAMBDA = LAMBDA
        proj = 0
        table = [0, 0]
        # table = [0, 0, 0, 0]
        y_pred2 = np.expand_dims(1 / (1 + np.exp(-y_pred)), axis=1)

        return {'y_pred2': y_pred2, 'y_pred': y_pred, 'y_predt': y_predt} # 'lfadv': lfadv

    def fit_classifier(self, X, y, sensitive, LAMBDA, Xtest, yt, sensitivet, y_pred, y_predt, lfadv, i):
        table = [0, 0]

        t = -np.squeeze(lfadv.T) # (B) Berechnung Angreifer Residuen
        proj = 0
        gradient = y - 1 / (1 + np.exp(-y_pred)) - LAMBDA * t - proj #  (A): y - 1 / (1 + np.exp(-y_pred)) (B): t
                                                                    # (C) Ableitung Trainingsverlust
        self.trees[i].fit(X, gradient) # (D) Baumfitting
        local_model = self.trees[i].fit(X, gradient) # -> send local_model to  coordinator; calculate gobal model and retun gobal model for predication with local test data

        return {'local_model': local_model}

    def update_classifier(self, X, y, sensitive, LAMBDA, Xtest, yt, sensitivet, y_pred, lfadv, update, i): #y_predt, updatet
        table = [0,0]
        t = -np.squeeze(lfadv.T)  # (B) Berechnung Angreifer Residuen
        #update = self.trees[i].predict(X) #(F)
        y_pred += np.multiply(self.learning_rate, update) # (E) / (F) Aktualisierung Modell
        y_fin = 1 / (1 + np.exp(-y_pred)) # new predicted probabilty

        losstraining = lossgr(y, y_fin)
        lossglobal = losstraining - LAMBDA * t

        #Test
        # updatet = self.trees[i].predict(Xtest)
        #y_predt += np.multiply(self.learning_rate, updatet)
        #y_predt2 = 1 / (1 + np.exp(-y_predt))

        accuracy = accuracy_score(y, np.squeeze(y_fin) > 0.5)
        #accuracyt = accuracy_score(yt, np.squeeze(y_predt2) > 0.5)
        prule = p_rule(y_fin, sensitive) / 100
        #prulet = p_rule(y_predt2, sensitivet) / 100

        #if i % 5 == 0:
        print(i, np.sum(lfadv), np.sum(losstraining), np.sum(lossglobal), "Accuracy:", round(accuracy, 4),
               #" test : ", round(accuracyt, 4),
               " Prule Train : ", p_rule(y_fin, sensitive) / 100,
               #" Prule test : ", p_rule(y_predt2, sensitivet) / 100
              )
        table = np.vstack(
            [table, [accuracy,
                      #accuracyt,
                      p_rule(y_fin, sensitive) / 100,
                     #p_rule(y_predt2, sensitivet) / 100
                     ]])

        return {'y_pred2': y_fin, 'y_pred': y_pred, 'score': accuracy, 'prule': prule, #'scoret': accuracyt,
                #'prulet': prulet,
                'sum_lossglobal': np.sum(lossglobal)
                #'y_predt2': y_predt2, 'y_predt': y_predt
                }

    def predict(self, X, n_estimators):
        y_pred = np.full(np.shape(X)[0], self.Init, self.Init)

        for estimator in range(n_estimators):
            update = self.trees[estimator].predict(X)
            y_pred += np.multiply(self.learning_rate, update)
            y_fin = 1 / (1 + np.exp(-y_pred))
            # Set label to the value that maximizes probability
        return y_fin

tf.disable_v2_behavior()


class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))


class LogisticRegression():
    def __init__(self, learning_rate=.1):
        self.param = None
        self.learning_rate = learning_rate
        self.sigmoid = Sigmoid()

    def _initialize_parameters(self, X):
        n_features = np.shape(X)[1]
        limit = 1 / math.sqrt(n_features)
        self.param = np.random.uniform(-limit, limit, (n_features,))

    def fit(self, X, y, iteration):
        y_pred = self.sigmoid(X.dot(self.param))
        self.param -= self.learning_rate * -(y - y_pred).dot(X)
        return self.param

    def gradient_adv(self, X, y):
        y_pred = self.sigmoid(X.dot(self.param))
        gradient_adv = (y - y_pred) * self.param * X.T * (1 - X).T
        return gradient_adv

    def predict(self, X):
        y_pred = np.round(self.sigmoid(X.dot(self.param))).astype(int)
        return y_pred

    def lossfunction(self, X, y):
        y_pred = self.sigmoid(X.dot(self.param))
        return lossgr(y, y_pred)

    def lossfunction_adv(self, X, y):
        y_pred = self.sigmoid(X.dot(self.param))
        return y - y_pred

    def param2(self):
        return 2 * self.param

def aggregate_classifier_outputs(classifier_outputs):
    counter = len(classifier_outputs.shape)

    agg_classifier = []
    # Convert numpy arrays to lists
    # classifier_outputs = [list(arr) if isinstance(arr, np.ndarray) else arr for arr in classifier_outputs]
    if counter > 1:
        print('classifier_outputs_IF')
        for element in range(len(classifier_outputs[0].shape)):
            sum = 0
            for array in classifier_outputs:
                sum += array[element]
            agg_classifier.append((sum / counter))
    else:
        print('classifier_outputs_ELSE')
        agg_classifier = classifier_outputs

    #print(agg_classifier)
    return agg_classifier

