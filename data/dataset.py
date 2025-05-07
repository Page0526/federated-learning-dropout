from torch.utils.data import Dataset
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


class MRIDataset(Dataset) :

    def __init__(self, root_dir: str, label_path: str = None, transform = None, label_df: pd.DataFrame = None ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        if label_df is None:
          self.labels_df = pd.read_csv(label_path)
        else :
          self.labels_df = label_df

        self.labels_df['subject_id'] = self.labels_df['subject_id'].astype(str)


        all_nii_files = list(self.root_dir.rglob("*.nii"))
        fail_paths = ["sub-BrainAge005600/anat/sub-BrainAge005600_T1w.nii/sub-BrainAge005600_T1w.nii"]
        self.file_paths = [fp for fp in all_nii_files if fp.is_file() and fp.name not in fail_paths ]

        valid_subjects = set(self.labels_df['subject_id'].values)

        self.file_paths = [fp for fp in self.file_paths if any(vs in str(fp) for vs in valid_subjects)]
        self.file_paths.sort()



    def __len__(self):
        return len(self.file_paths)


    def preprocessing_datapoint(self, img_data):

        mid_x = img_data.shape[0] // 2
        mid_y = img_data.shape[1] // 2
        mid_z = img_data.shape[2] // 2

        axial_slice = img_data[:, :, mid_z]
        coronal_slice = img_data[:, mid_y, :]
        sagittal_slice = img_data[mid_x, :, :]


        combined_data = np.stack([axial_slice, coronal_slice, sagittal_slice], axis=0)
        combined_data = torch.from_numpy(combined_data).float()

        if self.transform : combined_data = self.transform(combined_data)

        return combined_data




    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        file_path_str = str(img_path)

        subject_id = None
        valid_subjects_set = set(self.labels_df['subject_id'].values)


        for sid in valid_subjects_set:
            if sid in file_path_str:
                subject_id = sid
                break

        if subject_id is None:
            raise ValueError(f"Không tìm thấy subject_id cho file: {img_path}")

        metadata = self.labels_df.loc[self.labels_df['subject_id'] == subject_id].iloc[0].to_dict()

        img_data = nib.load(img_path).get_fdata()

        img_data = torch.from_numpy(img_data).float()

        label = 0
        if metadata['subject_sex'] == 'm' : label = 1

        return self.preprocessing_datapoint(img_data),  label



def visualize_sample(dataset, idx):
    mri_data, label = dataset[idx]
    title = f"Label: {label}\n"
    plt.close('all')
    fig = plt.figure(figsize = (18, 6))

    if isinstance(mri_data, torch.Tensor):
        data = mri_data.squeeze().numpy()
    else:
        data = mri_data


    ax1 = fig.add_subplot(1, 3, 1)
    plt.imshow(data[0, :, :].T, cmap='gray', origin='lower')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(data[1, :, :].T, cmap='gray', origin='lower')

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(data[2, :, :].T, cmap='gray', origin='lower')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()






if __name__ == "__main__":
 

    def test_dataset(): 
        print("Testing MRIDataset...")

        dataset = MRIDataset(root_dir="../dataset/not_skull_stripped", label_path="../dataset/label.csv")
        print(f"Number of samples: {len(dataset)}")

        for i in range(5):
            img, label = dataset[i]
            print(f"Sample {i}: Image shape: {img.shape}, Label: {label}")
        print("Test completed.")


    test_dataset()