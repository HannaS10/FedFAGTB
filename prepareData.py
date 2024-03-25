import os

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


#def stratified_k_fold_cross_validation_for_clients(input_csv, output_dir, k, num_clients, target_column):
 #   dataset = pd.read_csv(input_csv)

    # Calculate the size of each client subset
    # subset_size = len(dataset) // num_clients

    # Create the output directory if it doesn't exist
    # os.makedirs(output_dir, exist_ok=True)

    # for client_id in range(num_clients):
        # Determine the indices for the current client's subset
        # start_index = client_id * subset_size
        # end_index = (client_id + 1) * subset_size

        # Extract the subset for the current client
        # client_subset = dataset.iloc[start_index:end_index]

        # Initialize stratified k-fold cross-validation for the current client's subset
        # skf = StratifiedKFold(n_splits=k, shuffle=True)

        # Create a directory for the current client if it doesn't exist
        # client_output_dir = os.path.join(output_dir, f'client_{client_id + 1}')
        # os.makedirs(client_output_dir, exist_ok=True)

        # Perform k-fold cross-validation and save train/validation splits for the current client
        # fold = 1
        # for train_indices, test_indices in skf.split(client_subset, client_subset[target_column]):
            # Use the first fold as the test set, rest as training set
            # test_data = client_subset.iloc[test_indices]
            # train_data = client_subset.iloc[train_indices]

            # train_data.to_csv(os.path.join(client_output_dir, f'fold_{fold}_train.csv'), index=False)
            # test_data.to_csv(os.path.join(client_output_dir, f'fold_{fold}_test.csv'), index=False)

            # fold += 1

# SPLIT DATA INTO N CLIENTS AND PERFORM K FOLD CROSS VALIDATION
# INPUT: 1 CSV FILE
# OUTPUT: k CSV FILES (TRAIN AND TEST) FOR N CLIENTS
def k_fold_cross_validation_for_clients(input_csv, output_dir, k, num_clients):
    dataset = pd.read_csv(input_csv)
    print(f'lenghh of dataset {len(dataset)}, shape {dataset.shape[1]}')

    # Calculate the size of each client subset
    subset_size = len(dataset) // num_clients
    print(f'subset size {subset_size}')

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for client_id in range(num_clients):
        # Determine the indices for the current client's subset
        start_index = client_id * subset_size
        end_index = (client_id + 1) * subset_size

        # Extract the subset for the current client
        client_subset = dataset.iloc[start_index:end_index]

        # Initialize k-fold cross-validation for the current client's subset
        kf = KFold(n_splits=k, shuffle=True)

        # Create a directory for the current client if it doesn't exist
        client_output_dir = os.path.join(output_dir, f'client_{client_id + 1}')
        os.makedirs(client_output_dir, exist_ok=True)

        # Perform k-fold cross-validation and save train/validation splits for the current client
        fold = 1
        for train_indices, test_indices in kf.split(client_subset):
            # Use the first fold as the test set, rest as training set
            test_data = client_subset.iloc[test_indices]
            train_data = client_subset.iloc[train_indices]

            train_data.to_csv(os.path.join(client_output_dir, f'fold_{fold}_train.csv'), index=False)
            test_data.to_csv(os.path.join(client_output_dir, f'fold_{fold}_test.csv'), index=False)

            fold += 1


# Example usage:
input_csv_path = '/home/hanna/Development/fagtb-test/Preprocessing_Test/compas.csv'
output_directory = '/home/hanna/Development/fagtb-test/Compas_5'
k_fold_cross_validation_for_clients(input_csv_path, output_directory, 5, 5)




