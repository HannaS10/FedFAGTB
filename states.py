import jsonpickle
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

from algo import Coordinator, Client
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


# FeatureCloud requires that apps define the at least the 'initial' state.
# This state is executed after the app instance is started.

@app_state(INITIAL_STATE)
class InitialState(AppState):

    def register(self):
        self.register_transition(
            COMPUTE_LOCAL_MODEL_STATE)  # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        # config = bios.read(f'{INPUT_DIR}/config.yml')
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
            self.store('sens', config['fc_fagtb']['parameters']['sens'])
            self.store('columns_delete', config['fc_fagtb']['parameters']['columns_delete'])
            self.store('sep', config['fc_fagtb']['format']['sep'])
            self.store('split_mode', config['fc_fagtb']['split']['mode'])
            self.store('split_dir', config['fc_fagtb']['split']['dir'])
            self.log(f'Max iterations: {max_iterations}')
            self.log('Done reading Config File...')

            train_path = '/' + self.load('train')
            test_path = '/' + self.load('test')
            sens = self.load('sens')
            self.log('sens')
            self.log(sens)
            label_column = self.load('label_column')
            columns_delete = self.load('columns_delete')
            X_train = pd.read_csv(f'{INPUT_DIR}/train.csv')
            X_test = pd.read_csv(f'{INPUT_DIR}/test.csv')
            sensitive = X_train[sens].values
            sensitivet = X_test[sens].values
            y_train = X_train[label_column]
            y_test = X_test[label_column]

            scaler = StandardScaler().fit(X_train)
            s = X_train[sens]
            st = X_test[sens]
            t = X_train[label_column]
            tt = X_test[label_column]
            scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
            X_train = X_train.pipe(scale_df, scaler)
            X_test = X_test.pipe(scale_df, scaler)
            X_train = X_train.drop(sens, axis=1)
            X_train[sens] = s
            X_train[label_column] = t
            X_test = X_test.drop(sens, axis=1)
            X_test[sens] = st
            X_test[label_column] = tt

            # X_train = X_train.drop(columns_delete,1)
            X_train = X_train.drop(columns_delete, axis=1)
            # X_test = X_test.drop(columns_delete,1)
            X_test = X_test.drop(columns_delete, axis=1)

            # X_train = X_train.drop(columns_delete,1)
            X_train = X_train.drop(label_column, axis=1)
            # X_test = X_test.drop(columns_delete,1)
            X_test = X_test.drop(label_column, axis=1)

            # y_test.to_csv(os.replace("/input", "/output") + "/" + self.load('test_output'), index=False)

            print('Datasets:')
            self.store('X_train', X_train)
            print(X_train)
            self.store('y_train', y_train)
            print(y_train)
            self.store('X_test', X_test)
            print(X_test)
            self.store('y_test', y_test)
            print(y_test)
            self.store('sensitive', sensitive)
            print(sensitive)
            self.store('sensitivet', sensitivet)
            print(sensitivet)
            table = [0, 0, 0, 0]

            # fagtb_instance1 = FAGTB(10, 0.01, 3, 90, X_train, y_train, regression=1)
            print('fit inital')
            self.store('n_estimators', 30)
            classifier = FAGTB(n_estimators=30, learning_rate=0.01, max_depth=10, min_samples_split=1.0,
                               min_impurity=False,
                               max_features=20, regression=1)

            self.store('iteration', 0)

            if self.is_coordinator:
                self.log('Broadcasting initial model...')
                self.broadcast_data(classifier)
                # print(classifier)
                # print(type(classifier))
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
        self.register_transition(AGGREGATE_CLASSIFIER_STATE, role=Role.COORDINATOR)
        self.register_transition(WAIT_FOR_AGGREGATION_STATE, role=Role.PARTICIPANT)

    def run(self):
        iteration = self.load('iteration')
        iteration += 1
        self.store('iteration', iteration)

        self.log(f'ITERATION {iteration}')
        self.log("[CLIENT] Perform local computation")
        data_to_send = []
        # self.log(f'splits: {splits}')
        classifier = self.await_data()
        # print(classifier)
        # print(type(classifier))

        X_train = self.load('X_train')
        X_test = self.load('X_test')
        y_train = self.load('y_train')
        y_test = self.load('y_test')
        sensitive = self.load('sensitive')
        sensitivet = self.load('sensitivet')

        print('#####################################################################')
        print('initialization')
        init_result = classifier.fit_initial(X_train.values, y_train.values, sensitive, LAMBDA=0.15,
                                             Xtest=X_test.values, yt=y_test, sensitivet=sensitivet)
        y_pred2, y_pred, y_predt = init_result.values()
        y_pred = init_result['y_pred']
        self.store('y_pred', y_pred)
        print(f'type of y_pred {type(y_pred)}')
        print(f'y_pred {y_pred}')
        y_predt = init_result['y_predt']
        self.store('y_predt', y_predt)

        graph, X_input, y_input, sigm, logit, loss, train_steps, sigm2, pred, acc, init_var, var_grad, W1, b1 = classifier.build_graph(
            input_size=1, seed=7, learning_rate2=0.01)
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
        sess = classifier.initialize_session(graph, init_var)
        self.store('sess', sess)


        n_estimators = self.load('n_estimators')
        # Initialize aggregated_result before the loop

        for i in range(n_estimators):
            print('#####################################################################')
            print(i)
            # Initialize the classifier
            if i == 0:
                # Fit adversarial for i = 0
                # self.log(f'sess {sess}, graph {graph}, X_input {X_input}, y_input {y_input}, sigm2 {sigm2}, var_grad {var_grad}, train_steps {train_steps}, y_pred {y_pred}, sensitive {sensitive}')
                result_adversarial_i = classifier.fit_adversial(sess, graph, X_input, y_input, sigm2, var_grad,
                                                                train_steps, y_pred, sensitive)
                gradient_adv_from_adversarial = result_adversarial_i['lfadv']
                self.log(f'lfadv {gradient_adv_from_adversarial}')
            else:
                aggregated_result = self.await_data()
                result_adversarial_i = classifier.fit_adversial(sess, graph, X_input, y_input, sigm2, var_grad,
                                                                train_steps, aggregated_result, sensitive)
                gradient_adv_from_adversarial = result_adversarial_i['lfadv']
            self.store('gradient_adv_from_adversarial', gradient_adv_from_adversarial)
            print('gradient_adv_from_adversarial')
            print(gradient_adv_from_adversarial)
            result_classifier_i = classifier.fit_classifier(X_train.values, y_train.values, sensitive, LAMBDA=0.15,
                                                            Xtest=X_test.values, yt=y_test, sensitivet=sensitivet,
                                                            y_pred=y_pred,
                                                            y_predt=y_predt,
                                                            lfadv=gradient_adv_from_adversarial,
                                                            i=i)

            print(result_classifier_i)

            local_model_from_classifier = result_classifier_i['local_model']
            self.store('local_model_from_classifier', local_model_from_classifier)
            self.store('local_classifier', classifier)
            print(local_model_from_classifier)

            self.send_data_to_coordinator(local_model_from_classifier, use_smpc=self.load('smpc_used'))

            if self.is_coordinator:
                return AGGREGATE_CLASSIFIER_STATE
            else:
                self.log(f'[CLIENT] Sending computation data to coordinator')
                return WAIT_FOR_AGGREGATION_STATE
        # wenn
        # Warten auf glob. Klassifikatormodell und glob. Angreifermodell
        # fagtb = self.await_data()

        # (A) Residudenberechnung
        # (B) Faire Residuenberechnung
        # (C) Ableitung Trainingsverlust
        # (D) Baumfitting/ Hinzufügen eines neuen Baums (Training)
        # (E) Skalierung der Lernrate
        # (F) Aktualisierung des lokalen Modells -> send to coordinator -> switch to Aggregate State
        # lokalen Score berechnen

        # if done:
        #   return WRITE_STATE

        # self.send_data_to_coordinator(model, done)

        if self.is_coordinator:
            return AGGREGATE_CLASSIFIER_STATE
        else:
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
            global_model = self.await_data()
            self.store('global_model', global_model)
            self.log(f'global_model {global_model}')

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
        #   Aggregieren der lokalen Modellaktualiserungen zu einem glob. Modells
        #   globalen Score (Accuracy und prule) berechnen
        #   Broadcast glob. Modell an Clients (COMPUTE)
        #   Broadcast glob. Modell an COMPUTEADV

        n_estimators = self.load('n_estimators')
        for i in range(n_estimators):
            print('#####################################################################')
            print(i)
            self.log("[CLIENT] Global computation")
            models = self.gather_data()
            print(f'models {models}')
            data_to_broadcast = []
            global_model = []
            print(f'subclassifiers {global_model}')

            for local_model in models:
                global_model.append(local_model)
                print(f'global_model {global_model}')
            self.store('global_model', global_model)
            self.log(f'global_model {global_model}')

            done = self.load('iteration') >= self.load('max_iterations')
            #data_to_broadcast.append(global_model)

            self.broadcast_data(global_model) # data_to_broadcast
            self.log(f'[CLIENT] Broadcasting computation data to clients')

            return COMPUTE_GLOBAL_MODEL_STATE


@app_state(COMPUTE_GLOBAL_MODEL_STATE)
class ComputeGlobalModelState(AppState):
    def register(self):
        self.register_transition(UPDATE_LOCAL_MODEL_STATE)
        self.register_transition(TERMINAL_STATE)

    def run(self):
        n_estimators = self.load('n_estimators')
        for i in range(n_estimators):
            self.log('Predicting data...')

            global_model = self.load('global_model')
            #global_model = self.await_data()
            self.log(f'global_model {global_model}')
            X_train = self.load('X_train')
            y_train = self.load('Y_train')
            pred_output = self.load('pred_output')

            # Initialize an empty list to store individual predictions
            individual_predictions = []

            # Make predictions with each model
            for local_model in global_model:
                print(type(local_model))
                if not isinstance(local_model, DecisionTreeRegressor):
                    raise TypeError("All elements in global_model should be instances of DecisionTreeRegressor")
                individual_predictions.append(local_model.predict(X_train))

            # Aggregate predictions (e.g., take the average)
            aggregated_predictions = np.mean(individual_predictions, axis=0)
            print(aggregated_predictions)

            # score = accuracy_score(y_train, np.squeeze(aggregated_predictions) > 0.5)
            # score = accuracy_score(y_train, aggregated_predictions)

            # self.log(f'[API] Combined FGTB classifier model score on local test data for [CLIENT]]: '
              #       f'{score}, Predictions: {aggregated_predictions}')
            self.store('predictions', aggregated_predictions)
            # self.store('score_combined', score)

            # pd.DataFrame(data={'sum_pred': aggregated_predictions}).to_csv(f'{OUTPUT_DIR}/{pred_output}')
            return UPDATE_LOCAL_MODEL_STATE


@app_state(UPDATE_LOCAL_MODEL_STATE)
class UpdateLocalModelState(AppState):
    def register(self):
        self.register_transition(COMPUTE_LOCAL_ADV_STATE)  # We declare that 'terminal' state is accessible from the 'initial' state.
        self.register_transition(TERMINAL_STATE)
    def run(self):
        X_train = self.load('X_train')
        X_test = self.load('X_test')
        y_train = self.load('y_train')
        y_test = self.load('y_test')
        sensitive = self.load('sensitive')
        sensitivet = self.load('sensitivet')

        self.store('learning_rate', 0.01)
        learning_rate = self.load('learning_rate')

        y_pred = self.load('y_pred')
        print(f'type of y_pred {type(y_pred)}')
        print(f'y_pred {y_pred}')
        y_predt = self.load('y_predt')
        gradient_adv_from_adversarial = self.load('gradient_adv_from_adversarial')

        n_estimators = self.load('n_estimators')
        for i in range(n_estimators):
            update = self.load('predictions')
            print(f'type of update {type(update)}')
            print(f'update {update}')
            local_model = self.load('local_model_from_classifier')
            local_classifier = self.load('local_classifier')
            result_classifier_i = local_classifier.update_classifier(X_train.values, y_train.values, sensitive, LAMBDA=0.15,
                                                            Xtest=X_test.values, yt=y_test, sensitivet=sensitivet,
                                                            y_pred=y_pred,
                                                            y_predt=y_predt,
                                                            lfadv=gradient_adv_from_adversarial,
                                                            update=update,
                                                            i=i)
            #y_pred2, y_predi, score = local_classifier.update_classifier(X_train.values, y_train.values, sensitive, LAMBDA=0.15,
                                                       #     Xtest=X_test.values, yt=y_test, sensitivet=sensitivet,
                                                        #    y_pred=y_pred,
                                                       #     y_predt=y_predt,
                                                       #     lfadv=gradient_adv_from_adversarial,
                                                        #    update=update,
                                                        #    i=i)
            # y_pred += np.multiply(learning_rate, update)  # (E) oder (F) Aktualisierung Modell
            # y_fin = 1 / (1 + np.exp(-y_pred))  # new predicted probabilty

            self.store('y_pred2', result_classifier_i['y_pred2'])
            self.store('y_pred', result_classifier_i['y_pred'])
            y_pred = self.load('y_pred')
            print(f'y_pred {y_pred}')
            self.store('score', result_classifier_i['score'])

            self.store('local_classifier', local_classifier)

            self.log(f'new y_pred: {y_pred}')
            return COMPUTE_LOCAL_ADV_STATE


@app_state(COMPUTE_LOCAL_ADV_STATE)
class ComputeLocalAdversarialState(AppState):

    def register(self):
        self.register_transition(COMPUTE_LOCAL_MODEL_STATE)
        self.register_transition(TERMINAL_STATE)
        self.register_transition(AGGREGATE_ADV_STATE, role=Role.COORDINATOR)
        self.register_transition(WAIT_FOR_AGGREGATION_ADV_STATE)

    def run(self):
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

        sensitive = self.load('sensitive')
        y_pred = self.load('y_pred')
        print(f'y_pred {y_pred}')

        local_classifier = self.load('local_classifier')
        print(type(local_classifier))
        data_to_broadcast = []
        n_estimators = self.load('n_estimators')
        for i in range(n_estimators):
            print('#####################################################################')
            print(i)
            # Initialize the classifier
            if i == 0:
                # Fit adversarial for i = 0
                self.log(f'sess {sess}, graph {graph}, X_input {X_input}, y_input {y_input}, sigm2 {sigm2}, var_grad {var_grad}, train_steps {train_steps}, y_pred {y_pred}, sensitive {sensitive}')
                # result_adversarial_i = local_classifier.fit_adversial(sess, graph, X_input, y_input, sigm2, var_grad,
                                                                # train_steps, y_pred, sensitive)
                result_adversarial_i = local_classifier.update_adversial(sess, graph, X_input, y_input, sigm2, var_grad,
                                                                train_steps, y_pred, sensitive, W1, b1)
                updated_weight = result_adversarial_i['W1']
                updated_bias = result_adversarial_i['b1']

                self.store('W1', updated_weight)
                self.store('b1', updated_bias)
                print(f'updated weight {updated_weight}, updated bias {updated_bias}')

                data_to_broadcast.append(updated_weight)
                data_to_broadcast.append(updated_bias)
                self.send_data_to_coordinator(data_to_broadcast, use_smpc=self.load('smpc_used'), memo='adv')
                print(f'data_to broadcast {data_to_broadcast}')

                if self.is_coordinator:
                    return AGGREGATE_ADV_STATE
                else:
                    self.log(f'[CLIENT] Sending computation data to coordinator')
                    return WAIT_FOR_AGGREGATION_ADV_STATE

                #return TERMINAL_STATE


@app_state(AGGREGATE_ADV_STATE)
class AggregateAdversarialState(AppState):

    def register(self):
        #self.register_transition(UPDATE_LOCAL_ADV_STATE)
        self.register_transition(WRITE_STATE)
        self.register_transition(TERMINAL_STATE)

    def run(self):
        #   Aggregieren der lokalen Modellaktualiserungen zu einem glob. Modells
        #   globalen Score (Accuracy und prule) berechnen
        #   Broadcast glob. Modell an Clients (COMPUTE)
        #   Broadcast glob. Modell an COMPUTEADV
        self.data_incoming = []
        n_estimators = self.load('n_estimators')
        for i in range(n_estimators):
            print('#####################################################################')
            print(i)
            self.log("[CLIENT] Global computation of adversarial weight and biases")

            updated_weights = self.gather_data(memo='adv')  # 0 sind die Decison Trees, 1 die weights -> besser lösen!
            print(f'updated_weight, updated_bias {updated_weights}')
            data_to_broadcast = []
            global_adversial = []
            print(f'global_adversial {global_adversial}')

            #for local_adversarial in adversarial_data:
             #   global_adversial.append(local_adversarial)
              #  print(f'global_adversial {global_adversial}')
            #self.store('global_adversial', global_adversial)
            #self.log(f'global_adversial {global_adversial}')

            done = self.load('iteration') >= self.load('max_iterations')
            #data_to_broadcast.append(global_model)

            #self.broadcast_data(global_adversial) # data_to_broadcast
            self.log(f'[CLIENT] Broadcasting computation data to clients')

            return TERMINAL_STATE


@app_state(WAIT_FOR_AGGREGATION_ADV_STATE)
class WaitForAggregationAdversarialState(AppState):
    """
    The participant waits until it receives the aggregation data from the coordinator.
    """

    def register(self):
        self.register_transition(COMPUTE_GLOBAL_MODEL_STATE, Role.PARTICIPANT)
        self.register_transition(WAIT_FOR_AGGREGATION_STATE, Role.PARTICIPANT)

    def run(self) -> str or None:
        try:
            self.log("[CLIENT] Wait for aggregation")
            global_model = self.await_data()
            self.store('global_model', global_model)
            self.log(f'global_model {global_model}')

            self.log("[CLIENT] Received global model from coordinator.")

            # print(global_model)

            # print(done)
            # aggregated_result = aggregate_classifier_outputs(classifier_outputs)
            return TERMINAL_STATE

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
        #   Anpassen Angreifer-Klassifikator an neue glob. Modell
        #   Broadcast an Clients (COMPUTE)
        graph = self.load('graph')
        X_input = self.load('X_input')
        y_input = self.load('y_input')
        sigm = self.load('sigm')
        logit = self.load('logit')
        loss = self.load('loss')
        train_steps = self.load('train_steps')
        sigm2 = self.load('sigm2')
        init_var = self.load('init_var')
        pred = self.load('pred')
        acc = self.load('acc')
        init_var = self.load('init_var')
        var_grad = self.load('var_grad')
        W1 = self.load('W1')
        b1 = self.load('b1')
        sess = self.load('sess')
        saver = self.load('saver')

        sensitive = self.load('sensitive')
        y_pred = self.load('y_pred')
        print(f'y_pred {y_pred}')

        local_classifier = self.load('local_classifier')
        print(type(local_classifier))

        n_estimators = self.load('n_estimators')
        for i in range(n_estimators):
            print('#####################################################################')
            print(i)
            # Initialize the classifier
            if i == 0:
                # Fit adversarial for i = 0
                self.log(f'sess {sess}, graph {graph}, X_input {X_input}, y_input {y_input}, sigm2 {sigm2}, var_grad {var_grad}, train_steps {train_steps}, y_pred {y_pred}, sensitive {sensitive}')
                # result_adversarial_i = local_classifier.fit_adversial(sess, graph, X_input, y_input, sigm2, var_grad,
                                                                # train_steps, y_pred, sensitive)
                result_adversarial_i = local_classifier.update_adversial(sess, graph, X_input, y_input, sigm2, var_grad,
                                                                train_steps, y_pred, sensitive, W1, b1)
                # gradient_adv_from_adversarial = result_adversarial_i['sess']
                updated_weight = result_adversarial_i['W1']
                updated_bias = result_adversarial_i['b1']
                # print(f'gradient_adv_from_adversarial {gradient_adv_from_adversarial}')
                #self.log(f'result {gradient_adv_from_adversarial}')
                #gradient_adv_from_adversarial = tf.saved_model.save(gradient_adv_from_adversarial)
                #serialized_model = tf.io.serialize_tensor(gradient_adv_from_adversarial)
                #stored_model = serialized_model.numpy()
                self.store('W1', updated_weight)
                self.store('b1', updated_bias)
                print(f'updated weight {updated_weight}, updated bias {updated_bias}')

                self.send_data_to_coordinator([updated_weight, updated_bias], use_smpc=self.load('smpc_used'))

                if self.is_coordinator:
                    return AGGREGATE_ADV_STATE
                else:
                    self.log(f'[CLIENT] Sending computation data to coordinator')
                    return TERMINAL_STATE

                #return TERMINAL_STATE

@app_state(WRITE_STATE)
class WriteState(AppState):

    def register(self):
        self.register_transition(TERMINAL_STATE)

    def run(self):
        self.log('Predicting data...')
        #global_model = self.load('global_model')
        #X_test = self.load('X_test')
        #y_test = self.load('Y_test')
        #pred_output = self.load('pred_output')

        #clf = global_model
        #print(f' clf {clf}')
        #sum_pred = clf.predict(X_test)

        global_model = self.load('global_model')
        #global_model = self.await_data()
        self.log(f'global_model {global_model}')
        X_test = self.load('X_test')
        y_test = self.load('Y_test')
        pred_output = self.load('pred_output')

        # Initialize an empty list to store individual predictions
        individual_predictions = []

        # Make predictions with each model
        for local_model in global_model:
            print(type(local_model))
            if not isinstance(local_model, DecisionTreeRegressor):
                raise TypeError("All elements in global_model should be instances of DecisionTreeRegressor")
            individual_predictions.append(local_model.predict(X_test))

        # Aggregate predictions (e.g., take the average)
        aggregated_predictions = np.mean(individual_predictions, axis=0)
        print(aggregated_predictions)
        #score = accuracy_score(y_train, aggregated_predictions)

        #self.log(
         #   f'[API] Combined FGTB classifier model score on local test data for [CLIENT]]: {score}, Predictions: {aggregated_predictions}')
        self.store('predictions', aggregated_predictions)
        #self.store('score_combined', score)

        pd.DataFrame(data={'sum_pred': aggregated_predictions}).to_csv(f'{OUTPUT_DIR}/{pred_output}')
        return TERMINAL_STATE
