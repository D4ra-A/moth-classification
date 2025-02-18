import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import deque
from torchvision import transforms
from torch.utils.data import Dataset


# Custom Dataset Class for handling insect and background images
class CustomDataset(Dataset):
    def __init__(
            self,
            path_dict: dict,
            run_type: str = "train",
            seed: int = 42
        ) -> None:
        """
        Initializes the CustomDataset with paths and configurations for different run types.

        Parameters:
        path_dict (dict): Dictionary containing paths for images, tensors, and moth information table.
        run_type (str): Specifies the dataset type - 'train', 'val', or 'test'. Default is 'train'.
        seed (int): Random seed for reproducibility. Default is 42.
        """

        # Determine dataset mode (train, validation, or test)
        if run_type == "train":
            train, val, test = True, False, False
        elif run_type == "val":
            train, val, test = False, True, False
        elif run_type == "test":
            train, val, test = False, False, True
        else:
            raise ValueError("Please check run_type parameter")

        # Load paths for the table, images, and tensors
        moth_table = path_dict["moth_table"]            # Path to the moth information table
        image_folder_path = path_dict["image_path"]     # Path to the raw image directory
        tensor_folder_path = path_dict["tensor_path"]   # Path to the converted tensor directory

        # Load and process the moth table, images, and tensors
        self._table_loading(moth_table, train, val, test, seed)
        self._image_2_tensor(image_folder_path, tensor_folder_path, train)
        self._tensor_loading(tensor_folder_path, seed)
        self._print_summary(train, val, test)


    def _table_loading(
            self, 
            moth_table: str, 
            train: bool, 
            val: bool, 
            test: bool, 
            seed: int
        ) -> None:
        """
        Processes the moth information table and splits it into train, validation, and test sets.

        Parameters:
        moth_table (str): Path to the moth information Excel file.
        train (bool): Flag indicating training mode.
        val (bool): Flag indicating validation mode.
        test (bool): Flag indicating testing mode.
        seed (int): Random seed for reproducibility.
        """
        
        assert os.path.isfile(moth_table), "Please check whether MothInfo.xlsx file exists in correct path."

        self.insect_table = pd.read_excel(moth_table)
        self.insect_table["label"] = self.insect_table.iloc[:, -2].map({"C": 0, "A": 1})

        self.table_info = {}
        cam_idx = np.where(self.insect_table.label == 0)[0]  # Camouflage indices
        apo_idx = np.where(self.insect_table.label == 1)[0]  # Aposematism indices

        RS = np.random.RandomState(seed)
        cam_idx = RS.permutation(cam_idx)
        apo_idx = RS.permutation(apo_idx)

        if train:
            self.table_info["Camouflage"] = self.insect_table.iloc[cam_idx[:-20], 1].values
            self.table_info["Aposematism"] = self.insect_table.iloc[apo_idx[:-20], 1].values
        
        elif val:
            self.table_info["Camouflage"] = self.insect_table.iloc[cam_idx[-20:-10], 1].values
            self.table_info["Aposematism"] = self.insect_table.iloc[apo_idx[-20:-10], 1].values
        
        elif test:
            self.table_info["Camouflage"] = self.insect_table.iloc[cam_idx[-10:], 1].values
            self.table_info["Aposematism"] = self.insect_table.iloc[apo_idx[-10:], 1].values


    def _image_2_tensor(
            self, 
            image_folder_path: str, 
            tensor_folder_path: str, 
            train: bool
        ) -> None:
        """
        Converts image files to tensor format if not already converted.

        Parameters:
        image_folder_path (str): Path to the raw images.
        tensor_folder_path (str): Path to save the converted tensors.
        train (bool): Flag to check if the process is for training.
        """

        if os.path.exists(tensor_folder_path) and len(os.listdir(tensor_folder_path + "\\insect")) == self.insect_table.shape[0]:
            if train:
                print("Image files are already converted to Tensor.\n")

        else:
            transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
            print("Converting image files to Tensor...\n")
            os.makedirs(tensor_folder_path)

            # Convert insect images
            ins_image_folder_path = os.path.join(image_folder_path, "insect")
            ins_tensor_folder_path = os.path.join(tensor_folder_path, "insect")
            os.makedirs(ins_tensor_folder_path)

            tqdm_bar = tqdm(
                os.listdir(ins_image_folder_path),
                desc="[Insect]\tConverting insect image file to tensor",
                leave=False
            )

            for ins_image_file in tqdm_bar:
                ins_image_path = os.path.join(ins_image_folder_path, ins_image_file)
                ins_tensor_path = os.path.join(ins_tensor_folder_path, ins_image_file.rstrip(".jpg") + ".pt")

                ins_tensor = Image.open(ins_image_path).convert("RGB")
                ins_tensor = transform(ins_tensor)
                torch.save(ins_tensor, ins_tensor_path)
            tqdm_bar.close()

            # Convert background images
            env_image_folder_path = os.path.join(image_folder_path, "background")
            env_tensor_folder_path = os.path.join(tensor_folder_path, "background")
            os.makedirs(env_tensor_folder_path)

            for env_class in os.listdir(env_image_folder_path):
                env_class_image_folder_path = os.path.join(env_image_folder_path, env_class)
                env_class_tensor_folder_path = os.path.join(env_tensor_folder_path, env_class)
                os.makedirs(env_class_tensor_folder_path)

                tqdm_bar = tqdm(
                    os.listdir(env_class_image_folder_path),
                    desc=f"[{env_class}]\tConverting background image file to tensor",
                    leave=False
                )

                for env_image_file in tqdm_bar:
                    env_class_image_path = os.path.join(env_class_image_folder_path, env_image_file)
                    env_class_tensor_path = os.path.join(env_class_tensor_folder_path, env_image_file.rstrip(".png") + ".pt")

                    env_tensor = Image.open(env_class_image_path).convert("RGB")
                    env_tensor = transform(env_tensor)
                    torch.save(env_tensor, env_class_tensor_path)
                tqdm_bar.close()


    def _tensor_loading(
            self, 
            tensor_folder_path: str, 
            seed: int
        ) -> None:
        """
        Loads and processes tensor data for both insects and backgrounds.

        Parameters:
        tensor_folder_path (str): Path to the tensor directory.
        seed (int): Random seed for reproducibility.
        """

        ins_tensor_folder_path = os.path.join(tensor_folder_path, "insect")
        env_tensor_folder_path = os.path.join(tensor_folder_path, "background")
        ins_image = deque([])
        env_image = {}

        for ins_tensor in os.listdir(ins_tensor_folder_path):
            if ins_tensor.rstrip(".pt") in self.table_info["Camouflage"]:
                ins_image.append((torch.load(os.path.join(ins_tensor_folder_path, ins_tensor)), 0))

            elif ins_tensor.rstrip(".pt") in self.table_info["Aposematism"]:
                ins_image.append((torch.load(os.path.join(ins_tensor_folder_path, ins_tensor)), 1))

        for env_class in os.listdir(env_tensor_folder_path):
            env_image[env_class] = deque([
                torch.load(os.path.join(env_tensor_folder_path, env_class, env_path))
                for env_path in os.listdir(os.path.join(env_tensor_folder_path, env_class))
            ])

        RS = np.random.RandomState(seed)
        self.data = deque([])
        env1, env2, env3, env4 = env_image.values()

        for insect, label in ins_image:
            for n_rotate in range(4):
                self.data.append((
                    torch.rot90(insect, k=n_rotate, dims=(1, 2)),
                    env1[RS.randint(0, len(env1))],
                    env2[RS.randint(0, len(env2))],
                    env3[RS.randint(0, len(env3))],
                    env4[RS.randint(0, len(env4))],
                    label
                ))


    def _print_summary(
            self, 
            train: bool, 
            val: bool, 
            test: bool
        ) -> None:
        """
        Prints a summary of the dataset.

        Parameters:
        train (bool): Flag indicating training mode.
        val (bool): Flag indicating validation mode.
        test (bool): Flag indicating testing mode.
        """

        if train:
            print("Summary of the train dataset")
        elif val:
            print("Summary of the validation dataset")
        elif test:
            print("Summary of the test dataset")

        print("===================================================")
        for k, v in self.table_info.items():
            print(f"(insect)\t{k:17s}\timages:\t{len(v)}")
        print("===================================================\n")


    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, index: int) -> tuple:
        return self.data[index]


    def collate_fn(self, data_samples: list) -> tuple:
        """
        Custom collate function for DataLoader.

        Parameters:
        data_samples (list): List of data samples.

        Returns:
        tuple: Batched insect images, background images, and labels.
        """

        b_ins, b_env1, b_env2, b_env3, b_env4, b_label = ([] for _ in range(6))
        rotation = transforms.RandomRotation(90)

        for sample in data_samples:
            for i, (data, batch) in enumerate(zip(sample, [b_ins, b_env1, b_env2, b_env3, b_env4, b_label])):
                if i == 5:
                    batch.append(torch.tensor([data]))
                    continue

                if np.random.random() <= 0.5:   # Apply random rotation augmentation
                    data = rotation(data)

                batch.append(data)

        b_ins = torch.stack(b_ins).float()
        b_env1 = torch.stack(b_env1).float()
        b_env2 = torch.stack(b_env2).float()
        b_env3 = torch.stack(b_env3).float()
        b_env4 = torch.stack(b_env4).float()
        b_label = torch.cat(b_label).long()

        return (b_ins, b_env1, b_env2, b_env3, b_env4, b_label)
