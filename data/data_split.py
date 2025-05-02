from torch.utils.data import random_split 
import torch
from data.dataset import MRIDataset
import pandas as pd 
from data.helper import preprocessing_labels, prepare_data, sampling_data, create_train_test, distributed_data_to_clients


ROOT_PATH = "dataset/not_skull_stripped"
LABEL_PATH = "dataset/label.csv"


def iid_client_split(dataset, num_client = 3,  val_ratio = 0.2):

    client_datasets = []
    sample_per_client = len(dataset) // num_client


    for i in range(num_client):
        start_idx = i * sample_per_client
        end_idx = (i + 1) * sample_per_client if i < num_client - 1 else len(dataset)
        indecies = list(range(start_idx, end_idx))

        client_dataset = torch.utils.data.Subset(dataset, indecies)
        train_dataset, val_dataset = random_split(client_dataset, [1 - val_ratio, val_ratio])

        client_datasets.append((train_dataset, val_dataset))
    return client_datasets





def same_distribution_client_split(dataset, num_client, val_ratio = 0.2, overlap_ratio = 0.2):
    """
    Split the dataset into clients with the same distribution of labels.
    """
    labels_df = dataset.labels_df
    labels_df = preprocessing_labels(labels_df)
    labels_df = prepare_data(labels_df)

    client_datasets = distributed_data_to_clients(labels_df, num_clients=num_client, overlap_ratio=overlap_ratio)

    client_datasets = create_train_test(client_datasets, val_ratio=val_ratio)

    return client_datasets