import pandas as pd
from torch.utils.data import random_split
from data.dataset import MRIDataset
import torch
import os 

ROOT_PATH = "dataset/not_skull_stripped"
LABEL_PATH = "dataset/label.csv"



# processing if using subset of labels 
def preprocessing_labels(df: pd.DataFrame):
    
    subject_list = []
    for root, dirs, files in os.walk(ROOT_PATH):
      for dir_name in dirs:
        if dir_name.startswith("sub-BrainAge"):
            subject_list.append(dir_name)


    return df[df['subject_id'].isin(subject_list)]


def prepare_data(data: pd.DataFrame):

  df = data.copy()
  df['age_group'] = pd.qcut(df['subject_age'], q = min(5, len(df)), labels = False)
  df['key'] = df.apply(lambda row : f"{row['age_group']}_{row['subject_sex']}", axis = 1)
  return df





def sampling_data(data, size, random_state ):

  samples = data.groupby('key', group_keys = False)


  samples = samples.apply(lambda x: x.sample(
      n = min(int(size / len(data['key'].unique())), len(x)),
      replace = len(x) < int(size / len(data['key'].unique())),
      random_state =  random_state
  ))


  if len(samples) < size:
    additional_samples = data.drop(samples.index).sample(
        n = min(size - len(samples), len(data) - len(samples)),
        replace = True,
        random_state = random_state
    )

    samples = pd.concat([samples, additional_samples])
  return samples


def create_train_test(sample_labels: list, val_ratio: float = 0.2):

  client_datasets = []
  for label_df in sample_labels:
    dataset = MRIDataset(root_dir=ROOT_PATH, label_path=LABEL_PATH, label_df = label_df)
    
    train_dataset, val_dataset = random_split(dataset, [1 - val_ratio, val_ratio])
    client_datasets.append((train_dataset, val_dataset))
  return client_datasets




def distributed_data_to_clients(data: pd.DataFrame, num_clients: int, overlap_ratio: float):

  df = prepare_data(data)

  n_samples = len(df)
  samples_per_client = int(n_samples / (num_clients * (1 - overlap_ratio) + overlap_ratio))

  client_datasets = []
  selected_samples = {}

  # Tạo các client datasets với sự phân bố cân bằng
  for client_idx in range(num_clients):

      if client_idx == 0:
          client_data = df.sample(n=samples_per_client, random_state=42+client_idx)
      else:
          # overlap size
          overlap_size = int(samples_per_client * overlap_ratio)
          non_overlap_size = samples_per_client - overlap_size

          # building overlap
          all_previous_samples = pd.DataFrame()
          for prev_client_idx in range(client_idx):
              all_previous_samples = pd.concat([all_previous_samples, selected_samples[prev_client_idx]])

          # sampling
          if len(all_previous_samples) > 0:
              overlap_samples = sampling_data(all_previous_samples, overlap_size, client_idx * 100 + 42)
          else:
              overlap_samples = pd.DataFrame(columns=df.columns)

          # Lấy mẫu mới (không overlap)
          remaining_indices = df.index.difference(all_previous_samples.index)
          if len(remaining_indices) > 0:
              remaining_df = df.loc[remaining_indices]
              non_overlap_samples = sampling_data(remaining_df, non_overlap_size, client_idx * 100 + 42)
          else:

              non_overlap_samples = df.sample(n=non_overlap_size, replace=True, random_state=42+client_idx*300)


          client_data = pd.concat([overlap_samples, non_overlap_samples])


      selected_samples[client_idx] = client_data
      client_datasets.append(client_data.drop(['age_group', 'key'], axis=1))

  return client_datasets







