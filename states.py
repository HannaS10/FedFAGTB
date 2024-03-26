import datetime
import pickle

import jsonpickle
import numpy as np
from FeatureCloud.app.engine.app import AppState, app_state, Role, LogLevel, State
from sklearn.model_selection import train_test_split, KFold

import time
import pandas as pd
import yaml
import bios
import shutil
import os
import tensorflow as tf
# from functions import *
# from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import StandardScaler

from helpfunctions import *

INPUT_DIR = '/mnt/input'
OUTPUT_DIR = '/mnt/output'

INITIAL_STATE = 'initial'
COMPUTE_LOCAL_MODEL_STATE = 'compute local model'
COMPUTE_GLOBAL_MODEL_STATE = 'compute global model'
AGGREGATE_CLASSIFIER_STATE = 'aggregate classifier'
WAIT_FOR_AGGREGATION_STATE = 'wait for aggregation'
UPDATE_LOCAL_MODEL_STATE = 'update local model'
COMPUTE_LOCAL_ADV_STATE = 'compute local adversarial'
COMPUTE_GLOBAL_ADV_STATE = 'compute global adversarial'
AGGREGATE_ADV_STATE = 'aggregate adversarial'
WAIT_FOR_AGGREGATION_ADV_STATE = 'wait for aggregation adversarial'
UPDATE_LOCAL_ADV_STATE = 'update local adversarial'
WRITE_STATE = 'write'
TERMINAL_STATE = 'terminal'

#n_clients
#LAMBDA

# FeatureCloud requires that apps define the at least the 'initial' state.
# This state is executed after the app instance is started.

@app_state(INITIAL_STATE)
class InitialState(AppState):

    def register(self):
        self.register_transition(
            COMPUTE_LOCAL_MODEL_STATE)  # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self): #, n_clients, LAMBDA
        try:
            self.log("[CLIENT] Read input and config")
            config = bios.read(f'{INPUT_DIR}/config.yml')
            self.store('train', config['fc_fagtb']['input']['train'])
            self.store('test', config['fc_fagtb']['input']['test'])
            self.store('pred_output', config['fc_fagtb']['output']['pred'])
            self.store('test_output', config['fc_fagtb']['output']['test'])
            self.store('log_output', config['fc_fagtb']['output']['log'])
            self.store('result_output', config['fc_fagtb']['output']['result'])
            self.store('label_column', config['fc_fagtb']['parameters']['label_col'])
            self.store('sens', config['fc_fagtb']['parameters']['sens'])
            self.store('columns_delete', config['fc_fagtb']['parameters']['columns_delete'])
            self.store('sep', config['fc_fagtb']['format']['sep'])
            self.store('split_mode', config['fc_fagtb']['split']['mode'])
            self.store('split_dir', config['fc_fagtb']['split']['dir'])
            self.log('Done reading Config File...')

            self.store('fold', 5) #1,2,3,4,5
            fold = self.load('fold')
            self.store('LAMBDA', 1.0) #lambdas
            LAMBDA = self.load('LAMBDA')
            print(f'RUNNING TEST WITH fold {fold} and LAMBDA {LAMBDA}')

            sens = self.load('sens')
            label_column = self.load('label_column')
            columns_delete = self.load('columns_delete')

            # IMPORT TRAINING AND TEST DATA
            X_train = pd.read_csv(f'{INPUT_DIR}/fold_{fold}_train.csv')
            X_test = pd.read_csv(f'{INPUT_DIR}/fold_{fold}_test.csv')

            #X_train = pd.read_csv(f'{INPUT_DIR}/train.csv')
            #X_test = pd.read_csv(f'{INPUT_DIR}/test.csv')

            sensitive = X_train[sens].values
            sensitivet = X_test[sens].values
            y_train = X_train[label_column]
            y_test = X_test[label_column]

            # X_train = X_train.drop(columns_delete,1)
            X_train = X_train.drop(columns_delete, axis=1)
            X_train = X_train.drop(sens, axis=1)
            # X_test = X_test.drop(columns_delete,1)
            X_test = X_test.drop(columns_delete, axis=1)
            X_test = X_test.drop(sens, axis=1)

            y_train = np.asarray(y_train)

            sensitivet = np.asarray(sensitivet)
            sensitive = np.asarray(sensitive)
            # X_train = X_train.drop(columns_delete,1)
            X_train = X_train.drop(label_column, axis=1)
            X_train = np.asarray(X_train)

            # X_test = X_test.drop(columns_delete,1)
            X_test = X_test.drop(label_column, axis=1)
            X_test = np.asarray(X_test)

            y_test.to_csv(f'{OUTPUT_DIR}/' + self.load('test_output'), index=False)
            y_test = np.asarray(y_test)
            #pd.DataFrame(data={'sum_pred': sum_pred}).to_csv(f'{OUTPUT_DIR}/{pred_output}')

            #print('Datasets:')
            self.store('X_train', X_train)
            #print(X_train)
            print(f'shape and len {X_train.shape} {len(X_train)}')
            self.store('y_train', y_train)
            #print(y_train)
            self.store('X_test', X_test)
            #print(X_test)
            print(f'shape and len {X_test.shape} {len(X_test)}')
            self.store('y_test', y_test)
            #print(f'y_test {y_test}')
            self.store('sensitive', sensitive)
            #print(sensitive)
            self.store('sensitivet', sensitivet)
            #print(sensitivet)

            # CREATE INITIAL CLASSIFIER WITH X TREES AND BROADCAST TO THE PARTICIPANTS
            self.store('n_estimators', 300)
            classifier = FAGTB(n_estimators=300, learning_rate=0.01, max_depth=10, min_samples_split=1.0,
                               min_impurity=False,
                               max_features=20, regression=1)
            print(f'classifier {classifier}')
            self.store('iteration', 0)
            self.store('i', 0)

            if self.is_coordinator:
                self.log('Broadcasting initial model...')
                self.broadcast_data(classifier)#, send_to_self=True
                #self._app.data_incoming.keys()
                #self._app.data_incoming.values()
                #print(f'self_data_incoming keys {self._app.data_incoming.keys()}')
                #print(f'self_data_incoming values {self._app.data_incoming.values()}')

                #time.sleep(20)
                #print(f'self_data_incoming keys {self._app.data_incoming.keys()}')
                #print(f'self_data_incoming values {self._app.data_incoming.values()}')
                #classifier = self.gather_data()
                #print(f'classifier {classifier}')
                #print(f'self_data_incoming keys {self._app.data_incoming.keys()}')
                #print(f'self_data_incoming values {self._app.data_incoming.values()}')

            return COMPUTE_LOCAL_MODEL_STATE

        except Exception as e:
            self.log('no config file or missing fields', LogLevel.ERROR)
            self.update(message='no config file or missing fields', state=State.ERROR)
            print(e)
            return INITIAL_STATE


@app_state(COMPUTE_LOCAL_MODEL_STATE)
class ComputeLocalModelState(AppState):

    def register(self):
        self.register_transition(COMPUTE_LOCAL_MODEL_STATE)
        self.register_transition(TERMINAL_STATE)
        self.register_transition(AGGREGATE_CLASSIFIER_STATE, role=Role.COORDINATOR)
        self.register_transition(WAIT_FOR_AGGREGATION_STATE, role=Role.PARTICIPANT)

    def run(self):
        iteration = self.load('iteration')
        iteration += 1
        self.store('iteration', iteration)
        self.log(f'ITERATION {iteration}')

        i = self.load('i')
        print(f'i {i} Perform COMPUTE_LOCAL_MODEL_STATE')

        #self.log("[CLIENT] Perform local computation")

        X_train = self.load('X_train')
        X_test = self.load('X_test')
        y_train = self.load('y_train')
        y_test = self.load('y_test')
        sensitive = self.load('sensitive')
        sensitivet = self.load('sensitivet')
        LAMBDA = self.load('LAMBDA')
        n_estimators = self.load('n_estimators')
        #for i in range(i, n_estimators):
        print('#####################################################################')
        #print(i)
        # Initialize the classifier
        if i == 0:
            print('initialization')
            #print(f'self_data_incoming {len(self._app.data_incoming)}')
            #print(f'len clients {len(self._app.clients)}')
            #print(f'self_data_incoming values {self._app.data_incoming.values()}')
            #if len(self._app.clients) == 1:
             #   print(f'VALUES {self._app.data_incoming.values()}')
                #classifier = self._app.data_incoming['None']
              #  classifier = pickle.loads(self._app.data_incoming['None'][0][0])

            #else:
            # RECEIVE INITIAL CLASSIFIER AND INITIALIZE Y_PRED
            classifier = self.await_data()
            print(f'classifier in compoute global {classifier}')
            init_result = classifier.fit_initial(X_train, y_train, sensitive, LAMBDA=LAMBDA, # .values wenn Daten in pd vorliegen
                                                 Xtest=X_test, yt=y_test, sensitivet=sensitivet)

            print('classifier.fit_initial DONE')#y_pred2, y_pred, y_predt, lfadv = init_result.values()
            y_pred = init_result['y_pred']
            #lfadv = init_result['lfadv']
            self.store('y_pred', y_pred)
            #self.store('lfadv', lfadv)
            #print(f' 00: y_pred {y_pred}') #, lfadv {lfadv}
            y_predt = init_result['y_predt']
            self.store('y_predt', y_predt)

            # INITIALIZE ADVERSARIAL
            (graph, X_input, y_input, sigm, logit, loss, train_steps, sigm2, pred, acc, init_var, var_grad,
             W1, b1, assign_W1, assign_b1, new_W1, new_b1) = classifier.build_graph(input_size=1, seed=7,
                                                                                    learning_rate2=0.01)
            print('classifier.build_graph DONE')
            self.store('graph', graph)
            self.store('X_input', X_input)
            self.store('y_input', y_input)
            self.store('sigm', sigm)
            self.store('logit', logit)
            self.store('loss', loss)
            self.store('train_steps', train_steps)
            self.store('sigm2', sigm2)
            self.store('init_var', init_var)
            self.store('pred', pred)
            self.store('acc', acc)
            self.store('init_var', init_var)
            self.store('var_grad', var_grad)
            self.store('W1', W1)
            self.store('b1', b1)
            self.store('assign_W1', assign_W1)
            self.store('assign_b1', assign_b1)
            self.store('new_W1', new_W1)
            self.store('new_b1', new_b1)
            sess = classifier.initialize_session(graph, init_var)
            print('classifier.initilaize_session DONE')
            self.store('sess', sess)

            # FIT ADVERSARIAL for i = 0
            result_adversarial_i = classifier.fit_adversial(sess, graph, X_input, y_input, sigm2, var_grad,
                                                           train_steps, y_pred, sensitive, W1, b1, loss, acc, sigm, logit)
            print('classifier.fit_adversial DONE')
            lfadv = result_adversarial_i['lfadv']
            self.store('lfadv', lfadv)
        else:
            # LOAD CLASSIFIER AND CURRENT Y_PRED
            classifier = self.load('local_classifier')
            y_pred = self.load('y_pred')
            #print(f'y_pred {y_pred}')
            y_predt = self.load('y_predt')
            #print(f'y_pred {y_predt}')

        # LOAD CURRENT ADVERSARIAL LOSS
        lfadv = self.load('lfadv')
        LAMBDA = self.load('LAMBDA')

        # FIT GTB WITH (NEW) Y_PRED ABD ADVERSARIAL LOSS
        result_classifier_i = classifier.fit_classifier(X_train, y_train, sensitive, LAMBDA=LAMBDA, #values
                                                        Xtest=X_test, yt=y_test, sensitivet=sensitivet,
                                                        y_pred=y_pred,
                                                        y_predt=y_predt,
                                                        lfadv=lfadv,
                                                        i=i)

        local_model_from_classifier = result_classifier_i['local_model']
        self.store('local_model_from_classifier', local_model_from_classifier)
        self.store('local_classifier', classifier)
        #print(local_model_from_classifier)

        self.send_data_to_coordinator(local_model_from_classifier, use_smpc=self.load('smpc_used'))

        if self.is_coordinator:
            return AGGREGATE_CLASSIFIER_STATE
        else:
            self.log(f'[CLIENT] Sending computation data to coordinator')
            return WAIT_FOR_AGGREGATION_STATE


@app_state(WAIT_FOR_AGGREGATION_STATE)
class WaitForAggregationState(AppState):
    """
    The participant waits until it receives the aggregation data from the coordinator.
    """

    def register(self):
        self.register_transition(COMPUTE_GLOBAL_MODEL_STATE, Role.PARTICIPANT)
        self.register_transition(WAIT_FOR_AGGREGATION_STATE, Role.PARTICIPANT)

    def run(self) -> str or None:
        try:
            self.log("[CLIENT] Wait for aggregation")
            #global_model = self.await_data()
            #self.store('global_model', global_model)
            #self.log(f'global_model {global_model}')

            self.log("[CLIENT] Received global model from coordinator.")

            # print(global_model)

            # print(done)
            # aggregated_result = aggregate_classifier_outputs(classifier_outputs)
            return COMPUTE_GLOBAL_MODEL_STATE

        except Exception as e:
            self.log('error wait for aggregation', LogLevel.ERROR)
            self.update(message='error wait for aggregation', state=State.ERROR)
            print(e)
            return WAIT_FOR_AGGREGATION_STATE


@app_state(AGGREGATE_CLASSIFIER_STATE)
class AggregateClassifierState(AppState):

    def register(self):
        self.register_transition(COMPUTE_GLOBAL_MODEL_STATE)
        self.register_transition(WRITE_STATE)

    def run(self):
        #self.log("[CLIENT] Perform aggregation of classifiers")
        i = self.load('i')
        print(f'i {i} Perform AGGREGATE_CLASSIFIER_STATE')

        n_estimators = self.load('n_estimators')

        models = self.gather_data()
        #print(f'models {models}')
        data_to_broadcast = []
        global_model = []

        # GATHER ALL LOCAL MODELS (TREES)
        for local_model in models:
            global_model.append(local_model)
            #print(f'global_model {global_model}')
        self.store('global_model', global_model)
        #self.log(f'global_model {global_model}')

        # SHARE TREES WITH CLIENTS
        self.broadcast_data(global_model) # data_to_broadcast
        self.log(f'[CLIENT] Broadcasting computation data to clients')

        return COMPUTE_GLOBAL_MODEL_STATE


@app_state(COMPUTE_GLOBAL_MODEL_STATE)
class ComputeGlobalModelState(AppState):
    def register(self):
        self.register_transition(UPDATE_LOCAL_MODEL_STATE)
        self.register_transition(TERMINAL_STATE)

    def run(self):
        #self.log("[CLIENT] Perform global computation")
        i = self.load('i')
        print(f'i {i} COMPUTE_GLOBAL_MODEL_STATE')
        n_estimators = self.load('n_estimators')
        #for i in range(n_estimators):
        self.log('Predicting data...')

        # RECEIVE TREES
        global_model = self.await_data()
        #self.log(f'global_model {global_model}')
        X_train = self.load('X_train')
        X_test = self.load('X_test')

        individual_predictions_train = []
        individual_predictions_test = []

        # Make predictions with each model
        for local_model in global_model:
            #print(type(local_model))
            if not isinstance(local_model, DecisionTreeRegressor):
                raise TypeError("All elements in global_model should be instances of DecisionTreeRegressor")
            individual_predictions_train.append(local_model.predict(X_train))
            #print(f'individual predictions train {individual_predictions_train}')
            #individual_predictions_test.append(local_model.predict(X_test))
            #print(f'individual predictions test {individual_predictions_test}')

        # Aggregate predictions
        individual_predictions_train = np.mean(individual_predictions_train, axis=0)
        #individual_predictions_test = np.mean(individual_predictions_test, axis=0)
        #print(individual_predictions_train)
        #print(individual_predictions_test)

        #score_train = accuracy_score(y_train, np.squeeze(individual_predictions_train) > 0.5)
        #score_test = accuracy_score(y_train, np.squeeze(individual_predictions_test) > 0.5)
        # score = accuracy_score(y_train, aggregated_predictions)

        self.store('predictions', individual_predictions_train)
        #self.store('predictions_test', individual_predictions_test)
        # self.store('score_combined', score)

        return UPDATE_LOCAL_MODEL_STATE


@app_state(UPDATE_LOCAL_MODEL_STATE)
class UpdateLocalModelState(AppState):
    def register(self):
        self.register_transition(COMPUTE_LOCAL_ADV_STATE)  # We declare that 'terminal' state is accessible from the 'initial' state.
        self.register_transition(TERMINAL_STATE)
        self.register_transition(WRITE_STATE)

    def run(self):
        #self.log("[CLIENT] Perform local model update")
        i = self.load('i')
        print(f'i {i} Perform UPDATE_LOCAL_MODEL_STATE')
        X_train = self.load('X_train')
        X_test = self.load('X_test')
        y_train = self.load('y_train')
        y_test = self.load('y_test')
        #print(f'y_test {y_test}')
        sensitive = self.load('sensitive')
        sensitivet = self.load('sensitivet')

        self.store('learning_rate', 0.01)
        learning_rate = self.load('learning_rate')

        y_pred = self.load('y_pred')
        #print(f'y_pred {y_pred}')
        #y_predt = self.load('y_predt')
        lfadv = self.load('lfadv')

        n_estimators = self.load('n_estimators')
        #for i in range(n_estimators):
        update = self.load('predictions')
        #print(f'update {update}')
        #updatet = self.load('predictions_test')
        local_model = self.load('local_model_from_classifier')
        local_classifier = self.load('local_classifier')
        print(f'local_classifier {local_classifier}')
        LAMBDA = self.load('LAMBDA')
        # UPDATE CLASSIFIER WITH NEW PREDICTIONS
        result_classifier_i = local_classifier.update_classifier(X_train, y_train, sensitive, LAMBDA=LAMBDA, #values
                                                        Xtest=X_test, yt=y_test, sensitivet=sensitivet,
                                                        y_pred=y_pred,
                                                        #y_predt=y_predt,
                                                        lfadv=lfadv,
                                                        update=update, #updatet=updatet,
                                                        i=i)
        self.store('y_pred2', result_classifier_i['y_pred2'])
        self.store('y_pred', result_classifier_i['y_pred'])
        y_pred = self.load('y_pred')

        score = result_classifier_i['score']
        prule = result_classifier_i['prule']
        #scoret = result_classifier_i['scoret']
        #prulet = result_classifier_i['prulet']

        self.store('local_classifier', local_classifier)

        if i < n_estimators-1:
            return COMPUTE_LOCAL_ADV_STATE
        else:
            return WRITE_STATE


@app_state(COMPUTE_LOCAL_ADV_STATE)
class ComputeLocalAdversarialState(AppState):

    def register(self):
        self.register_transition(COMPUTE_LOCAL_MODEL_STATE)
        self.register_transition(TERMINAL_STATE)
        self.register_transition(AGGREGATE_ADV_STATE, role=Role.COORDINATOR)
        self.register_transition(WAIT_FOR_AGGREGATION_ADV_STATE)

    def run(self):
        #self.log("[CLIENT] Perform local adversarial")
        i = self.load('i')
        print(f'i {i} Perform COMPUTE_LOCAL_ADV_STATE')
        #   Anpassen Angreifer-Klassifikator an neue glob. Modell
        #   Broadcast an Clients (COMPUTE)
        graph = self.load('graph')
        X_input = self.load('X_input')
        y_input = self.load('y_input')

        train_steps = self.load('train_steps')
        sigm2 = self.load('sigm2')

        var_grad = self.load('var_grad')
        W1 = self.load('W1')
        b1 = self.load('b1')
        sess = self.load('sess')
        loss = self.load('loss')
        acc = self.load('acc')

        sensitive = self.load('sensitive')
        y_pred = self.load('y_pred')
        #print(f'y_pred {y_pred}')

        local_classifier = self.load('local_classifier')
        #print(type(local_classifier))
        data_to_broadcast = []
        n_estimators = self.load('n_estimators')

        # UPDATE ADVERSARIAL WITH NEW PREDICTIONS
        result_adversarial_i = local_classifier.update_adversial(sess, graph, X_input, y_input, sigm2, var_grad,
                                                        train_steps, y_pred, sensitive, W1, b1, loss, acc)
        updated_weight = result_adversarial_i['W1_updated']
        updated_bias = result_adversarial_i['b1_updated']

        self.store('W1', W1) # zum Speichern der tf.Variablen
        self.store('b1', b1)

        self.store('W1_updated', updated_weight) # zum Speichern der Werte der  tf.Variablen
        self.store('b1_updated', updated_bias)
        #print(f'updated weight {updated_weight}, updated bias {updated_bias}')

        data_to_broadcast.append([updated_weight, updated_bias])

        self.send_data_to_coordinator(data_to_broadcast)
        #print(f'data_to broadcast {updated_weight, updated_bias}')

        if self.is_coordinator:
            return AGGREGATE_ADV_STATE
        else:
            self.log(f'[CLIENT] Sending computation data to coordinator')
            return WAIT_FOR_AGGREGATION_ADV_STATE


@app_state(AGGREGATE_ADV_STATE)
class AggregateAdversarialState(AppState):

    def register(self):
        self.register_transition(UPDATE_LOCAL_ADV_STATE)
        self.register_transition(TERMINAL_STATE)

    def run(self):
        #self.log("[CLIENT] Perform aggregation of adversarials")
        i = self.load('i')
        print(f'i {i} Perform AGGREGATE_ADV_STATE')

        n_estimators = self.load('n_estimators')
        # GATHER WEIGHTS AND BIASES
        updated_weight = self.gather_data()
        #print(f'updated_weight, updated_bias {updated_weight}')
        data_to_broadcast = []

        # AGGREGATE WEIGHTS AND BIASES
        weights_aggregated = [item[0][0] for item in updated_weight]
        weights_aggregated = np.concatenate(weights_aggregated, axis=0)

        biases_aggregated = [item[0][1] for item in updated_weight]
        biases_aggregated = np.concatenate(biases_aggregated, axis=0)

        mean_weight = np.mean(weights_aggregated)
        mean_bias = np.mean(biases_aggregated)

        # SHARE AGGREGATED WEIGHT AND BIAS
        data_to_broadcast.append([mean_weight, mean_bias])
        #done = self.load('iteration') >= self.load('max_iterations')
        self.broadcast_data(data_to_broadcast)
        #print(f'data_to_broadcast {data_to_broadcast}')
        self.log(f'[CLIENT] Broadcasting computation data to clients')

        return UPDATE_LOCAL_ADV_STATE


@app_state(WAIT_FOR_AGGREGATION_ADV_STATE)
class WaitForAggregationAdversarialState(AppState):
    """
    The participant waits until it receives the aggregation data from the coordinator.
    """

    def register(self):
        self.register_transition(COMPUTE_GLOBAL_MODEL_STATE, Role.PARTICIPANT)
        self.register_transition(WAIT_FOR_AGGREGATION_STATE, Role.PARTICIPANT)
        self.register_transition(UPDATE_LOCAL_ADV_STATE, Role.PARTICIPANT)

    def run(self) -> str or None:
        try:
            self.log("[CLIENT] Wait for aggregation")
            #global_adv_model = self.await_data()
            #self.store('global_model', global_adv_model)
            #self.log(f'global_model {global_adv_model}')

            self.log("[CLIENT] Received global model from coordinator.")

            # print(global_model)

            # print(done)
            # aggregated_result = aggregate_classifier_outputs(classifier_outputs)
            return UPDATE_LOCAL_ADV_STATE

        except Exception as e:
            self.log('error wait for aggregation', LogLevel.ERROR)
            self.update(message='error wait for aggregation', state=State.ERROR)
            print(e)
            return WAIT_FOR_AGGREGATION_ADV_STATE


@app_state(UPDATE_LOCAL_ADV_STATE)
class UpdateLocalAdversarialState(AppState):

    def register(self):
        self.register_transition(COMPUTE_LOCAL_MODEL_STATE)
        self.register_transition(TERMINAL_STATE)
        self.register_transition(AGGREGATE_ADV_STATE)

    def run(self):
        #self.log("[CLIENT] Perform local adversarial update")

        i = self.load('i')
        print(f'i {i} Perform UPDATE_LOCAL_ADV_STATE')

        mean_weight_bias = self.await_data()
        #self.log(f'global_model {mean_weight_bias}')

        new_W = [mean_weight_bias[0][0]]
        new_W = [new_W]
        #print(f'new_W1 {new_W}')
        new_b = [mean_weight_bias[0][1]]
        #print(f'new_b1 {new_b}')

        graph = self.load('graph')
        X_input = self.load('X_input')
        y_input = self.load('y_input')

        train_steps = self.load('train_steps')
        sigm2 = self.load('sigm2')

        var_grad = self.load('var_grad')
        W1 = self.load('W1')
        b1 = self.load('b1')
        assign_W1 = self.load('assign_W1')
        assign_b1 = self.load('assign_b1')
        new_W1 = self.load('new_W1')
        new_b1 = self.load('new_b1')
        sess = self.load('sess')
        loss = self.load('loss')
        acc = self.load('acc')

        sensitive = self.load('sensitive')
        y_pred = self.load('y_pred')
        #print(f'y_pred {y_pred}')

        local_classifier = self.load('local_classifier')

        n_estimators = self.load('n_estimators')
        # UPDATE ADVERSARIAL WITH NEW WEIGHT AND BIAS AND CALCULATE NEW LOSS
        result_adversarial_i = local_classifier.update_adv_gradient(sess, graph, X_input, y_input, sigm2, var_grad,
                                                                 train_steps, y_pred, sensitive,
                                                                W1, b1, new_W, new_b, assign_W1, assign_b1, new_W1, new_b1, loss, acc)
        new_lfadv = result_adversarial_i['lfadv']
        #print(f'new_lfadv {new_lfadv}')
        new_S_ADV = result_adversarial_i['S_ADV']
        #print(f'new_S_ADV {new_S_ADV}')

        self.store('lfadv', new_lfadv)
        self.store('S_ADV', new_S_ADV)

        self.store('W1', W1)  # zum Speichern der tf.Variablen
        self.store('b1', b1)

        i += 1
        self.store('i', i)
        #print(f' i {i}')

        return COMPUTE_LOCAL_MODEL_STATE


@app_state(WRITE_STATE)
class WriteState(AppState):

    def register(self):
        self.register_transition(TERMINAL_STATE)

    def run(self):
        i = self.load('i')
        print(f'i {i} Perform WRITE_STATE')
        #self.log('Predicting data...')
        local_classifier = self.load('local_classifier')
        print(f'local_classifier {local_classifier}')

        #local_model_from_classifier = self.load('local_model_from_classifier')
        #print(f'local_model_from_classifier {local_model_from_classifier}')

        X_test = self.load('X_test')
        y_test = self.load('y_test')
        sensitivet = self.load('sensitivet')
        pred_output = self.load('pred_output')
        log_output = self.load('log_output')
        table = [0, 0, 0, 0]

        n_estimators = self.load('n_estimators')

        # PREDICTION ON TEST DATA
        sum_pred = local_classifier.predict(X_test, n_estimators) #values
        print(f'y_predt2 {sum_pred}')
        #score = accuracy_score(y_test, sum_pred)
        score = accuracy_score(y_test, np.squeeze(sum_pred) > 0.5)

        self.log(
            f'[API] Combined FGTB classifier model score on local test data for [CLIENT]]: {score}')

        self.store('final_predictions', sum_pred)
        self.store('final_score', score)

        pd.DataFrame(data={'prediction': sum_pred}).to_csv(f'{OUTPUT_DIR}/{pred_output}', index=False)

        print('Results on test set :')
        Res = display_results(sum_pred, y_test, sensitivet) #values
        table = np.vstack([table, [Res['Accuracy'] * 100, Res['PRULE'], Res['DispFPR'], Res['DispFNR']]])

        np.savetxt(sys.stdout, np.mean(table[1:, ], axis=0).astype(float), '%5.2f')

        # WRITE RESULTS IN CSV FILE
        with open(f'{OUTPUT_DIR}/' + self.load('result_output'), 'a') as f:
            pd.DataFrame(data={'Accuracy': [Res['Accuracy'] * 100], 'Balanced Accuracy': [Res['Balanced Accuracy'] * 100],
                               'ROC Score': [Res['ROC Score'] * 100], 'Prule': [Res['PRULE']],
                              'DI': [Res['DI']], 'DispFPR': [Res['DispFPR']], 'DispFNR': [Res['DispFNR']]}).to_csv(f, index=False)


        X_train = self.load('X_train')
        y_train = self.load('y_train')
        sensitive = self.load('sensitive')
        y_pred_S1_train = local_classifier.predict((X_train)[sensitive == 1], n_estimators) #values
        score_S1_train = accuracy_score((y_train)[sensitive == 1], np.squeeze(y_pred_S1_train) > 0.5) #values
        score_balanced_S1_train = balanced_accuracy_score((y_train)[sensitive == 1], np.squeeze(y_pred_S1_train) > 0.5) #values
        print(f'score S1_train {score_S1_train}, score balanced_S1_train {score_balanced_S1_train}')
        y_pred_S0_train = local_classifier.predict((X_train)[sensitive == 0], n_estimators) #values
        score_S0_train = accuracy_score((y_train)[sensitive == 0], np.squeeze(y_pred_S0_train) > 0.5) #values
        score_balanced_S0_train = balanced_accuracy_score((y_train)[sensitive == 0], np.squeeze(y_pred_S0_train) > 0.5) #values
        print(f'score S0_train {score_S0_train}, score balanced_S0_train {score_balanced_S0_train}')

        # WRITE RESULTS IN CSV FILE
        with open(f'{OUTPUT_DIR}/' + self.load('result_output'), 'a') as f:
            pd.DataFrame(data={'score_S1_train': [score_S1_train], 'score_balanced_S1_train': [score_balanced_S1_train],
                               'score_S0_train': [score_S0_train], 'score_balanced_S0_train': [score_balanced_S0_train]
                                }).to_csv(f, index=False)

        return TERMINAL_STATE
