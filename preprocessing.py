import csv
import os

import pandas as pd

from fairness import results
from fairness.data.objects.list import DATASETS, get_dataset_names
#from fairness.data.objects.ProcessedData import ProcessedData
from ProcessData import *

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MaxAbsScaler


def DATA_TRAIN_TEST_adult_compas(num, base_folder):
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    dataset = DATASETS[num] # 1 für Adult data set, 3 für COMPAS
    print(f'dataset {dataset}')
    all_sensitive_attributes = dataset.get_sensitive_attributes_with_joint()
    ProcessedData(dataset)
    print(f'ProcessedData {ProcessedData}')
    processed_dataset = ProcessedData(dataset)
    train_test_splits = processed_dataset.create_train_test_splits(1)
    train_test_splits.keys()
    X_train, X_test = train_test_splits['numerical-binsensitive'][0]
    print(f'X_train {X_train}')

    getDataframe = processed_dataset.get_dataframe('numerical-binsensitive')
    Data = getDataframe
    print(f'Data {Data}')
    adult_df = Data.sample(frac=1, random_state=42)

    # DataFrame mit den ausgeschlossenen Spalten
    df_binary = adult_df[['two_year_recid', 'race']] #income-per-year', 'sex'
    df_scaled = adult_df.drop(columns=['two_year_recid', 'race']) #income-per-year', 'sex'

    scaler = StandardScaler().fit(df_scaled)
    scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
    df_scaled = df_scaled.pipe(scale_df, scaler)

    data = pd.concat([df_scaled, df_binary], axis=1)
    print(f'Data {data}')
    #data.to_csv(os.path.join(base_folder, 'census.csv'), index=False) #, index=False
    data.to_csv(os.path.join(base_folder, 'compas.csv'), index=False) #, index=False

    return data


data = DATA_TRAIN_TEST_adult_compas(3, 'Preprocessing_Test') # 1=Adult, 3=Compas
