from FeatureCloud.app.engine.app import AppState, app_state, Role, LogLevel, State
from sklearn.model_selection import train_test_split, KFold

import time
import pandas as pd
import yaml
import bios
import shutil
import os
from functions import *
# from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score

from algo import Coordinator, Client

INPUT_DIR = '/mnt/input'
OUTPUT_DIR = '/mnt/output'

INITIAL_STATE = 'initial'
COMPUTE_STATE = 'compute'
AGGREGATE_STATE = 'aggregate'
COMPUTEADV_STATE = 'compute adversial'
WRITE_STATE = 'write'
TERMINAL_STATE = 'terminal'
# FeatureCloud requires that apps define the at least the 'initial' state.
# This state is executed after the app instance is started.

@app_state(INITIAL_STATE)
class InitialState(AppState):

    def register(self):
        self.register_transition(COMPUTE_STATE)  # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        #config = bios.read(f'{INPUT_DIR}/config.yml')
        try:
            self.log("[CLIENT] Read input and config")
            models = {}
            splits = {}
            test_splits = {}
            betas = {}

            config = bios.read(f'{INPUT_DIR}/config.yml')
            self.store('train', config['fc_fagtb']['input']['train'])
            self.store('test', config['fc_fagtb']['input']['test'])
            max_iterations = self.store('max_iterations', config['fc_fagtb']['algo']['max_iterations'])
            self.store('pred_output', config['fc_fagtb']['output']['pred'])
            self.store('test_output', config['fc_fagtb']['output']['test'])
            self.store('label_column', config['fc_fagtb']['parameters']['label_col'])
            self.store('sep', config['fc_fagtb']['format']['sep'])
            self.store('split_mode', config['fc_fagtb']['split']['mode'])
            self.store('split_dir', config['fc_fagtb']['split']['dir'])
            #self.store('smpc_used', config.get('use_smpc', False))
            self.log(f'Max iterations: {max_iterations}')
            self.log('Done reading Config File...')


            X_train, X_test, y_train, y_test, sensitive, sensitivet = DATA_TRAIN_TEST(1, 'sex', "income-per-year",
                                                                                      ['sex', 'income-per-year',
                                                                                       'race-sex'])  # ,'race','race-sex','sex'
            #X = pd.read_csv(train_path, sep=self.load('sep'))
            #y = X.loc[:, self.load('label_column')]
            #X = X.drop(self.load('label_column'), axis=1)

            X = X_train
            y = y_train

            #X_test = pd.read_csv(test_path, sep=self.load('sep'))
            #y_test = X_test.loc[:, self.load('label_column')]
            #X_test = X_test.drop(self.load('label_column'), axis=1)

            X_test = X_test
            y_test = y_test

            #y_test.to_csv(split.replace("/input", "/output") + "/" + self.load('test_output'), index=False)

            #splits[split] = [X, y]
            #test_splits[split] = [X_test, y_test]

            self.log('Preparing initial model...')
            # table = [0, 0, 0, 0]
            classifier = FAGTB(n_estimators=100, learning_rate=0.01, max_depth=10, min_samples_split=1.0,
                               min_impurity=False, max_features=20, regression=1)
            y_pred = classifier.fit(X_train.values, y_train.values, sensitive, LAMBDA=0.15, Xtest=X_test.values,
                                    yt=y_test,
                                    sensitivet=sensitivet)
            self.store('model', y_pred)

            self.store('iteration', 0)

            # y_pred = classifier.fit(X_train.values, y_train.values, sensitive, LAMBDA=0.15, Xtest=X_test.values, yt=y_test,sensitivet=sensitivet)
            # print(y_pred)

            if self.is_coordinator:
                self.log('Broadcasting initial model...')
                self.broadcast_data(y_pred)
                print(y_pred)
            return COMPUTE_STATE

        except Exception as e:
            self.log('no config file or missing fields', LogLevel.ERROR)
            self.update(message='no config file or missing fields', state=State.ERROR)
            print(e)
            return INITIAL_STATE

    def build_model(self, X_train, X_test, y_train, y_test, n_estimators, learning_rate, random_state, metric, sensitive, sensitivet):
        # Create GradientBoost object
        # gb = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
        # gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=3, max_features=90, random_state=0, min_impurity_decrease=0.0, min_samples_leaf=2, min_samples_split=2,  min_weight_fraction_leaf=0.0)
        model = FAGTB(n_estimators=100, learning_rate=0.01, max_depth=10, min_samples_split=1.0,
                      min_impurity=False, max_features=20, regression=1)
        y_pred = model.fit(X_train.values, y_train.values, sensitive, LAMBDA=0.15, Xtest=X_test.values,
                           yt=y_test,
                           sensitivet=sensitivet)
        # Train GradientBoost
        # model = y_pred.fit(X_train, y_train)

        if metric == "acc":
            score = accuracy_score(y_test, y_pred)
        elif metric == "matth":
            score = matthews_corrcoef(y_test, y_pred)
        elif metric == "roc" or metric == "auc":
            score = roc_auc_score(y_test, y_pred)
        else:
            score = accuracy_score(y_test, y_pred)

        return score, model


@app_state(COMPUTE_STATE)
class ComputeState(AppState):

    def register(self):
        self.register_transition(COMPUTE_STATE, role=Role.PARTICIPANT)
        self.register_transition(AGGREGATE_STATE, role=Role.COORDINATOR)
        self.register_transition(WRITE_STATE)

    def run(self):
        iteration = self.load('iteration')
        iteration += 1
        self.store('iteration', iteration)

        self.log(f'ITERATION {iteration}')

        # wenn
        # Warten auf glob. Klassifikatormodell und glob. Angreifermodell
        model, done = self.await_data()

        # (A) Residudenberechnung
        # (B) Faire Residuenberechnung
        # (C) Ableitung Trainingsverlust
        # (D) Baumfitting/ Hinzufügen eines neuen Baums (Training)
        # (E) Skalierung der Lernrate
        # (F) Aktualisierung des lokalen Modells -> send to coordinator -> switch to Aggregate State
        # lokalen Score berechnen

        if done:
            return WRITE_STATE

        self.send_data_to_coordinator(model, done)

        if self.is_coordinator:
            return AGGREGATE_STATE
        else:
            return COMPUTE_STATE


@app_state(AGGREGATE_STATE)
class AggregateState(AppState):

    def register(self):
        self.register_transition(COMPUTE_STATE)
        self.register_transition(COMPUTEADV_STATE, Role.COORDINATOR)

    def run(self):
        #   Aggregieren der lokalen Modellaktualiserungen zu einem glob. Modells
        #   globalen Score (Accuracy und prule) berechnen
        #   Broadcast glob. Modell an Clients (COMPUTE)
        #   Broadcast glob. Modell an COMPUTEADV

        if self.is_coordinator:
            # return COMPUTEADV_STATE
            return COMPUTE_STATE
        else:
            return COMPUTE_STATE


@app_state(COMPUTEADV_STATE)
class ComputeAdversialState(AppState):

    def register(self):
        self.register_transition(COMPUTE_STATE)  # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        #   Anpassen Angreifer-Klassifikator an neue glob. Modell
        #   Broadcast an Clients (COMPUTE)

        return COMPUTE_STATE


@app_state(WRITE_STATE)
class WriteState(AppState):

    def register(self):
        self.register_transition('terminal')  # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        self.log('Predicting data...')
        model = self.load('model')
        X_train = self.load('X_train')
        pred_output = self.load('pred_output')
        pred = model.predict(X_train)
        pd.DataFrame(data={'pred': pred}).to_csv(f'{OUTPUT_DIR}/{pred_output}')
        return TERMINAL_STATE
