import os
from typing import Any, Callable, Optional, Tuple, Union
import itertools
from pathlib import Path

import pickle
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.transforms.v2 import Transform
from torchvision.datasets.utils import check_integrity, download_url, verify_str_arg

class BaseDataset():
    
    def __init__(self, train: bool, transforms: Transform):
        self.train = train
        self.transforms = transforms
    
    def split_labeled_unlabeled_data(self, X, y, labeled_size: Union[int, float], random_state:int = None):
        """_summary_

        Args:
            X (_type_): train_input
            y (_type_): train_output
            train_size (Union[int, float]): number of training images
            random_state (int, optional): the random training state, none mean random state

        Returns:
            _splitting : list, length=2 * len(arrays)
                List containing label-unlabels split of inputs.
        """
        # stratify ensure that it is an equal distribution split for each class
        return train_test_split(X, y, train_size=labeled_size, stratify=y, random_state=random_state)
    
    def get_dataloader(
        self, 
        labeled_batch_size: int, 
        labeled_num_worker: int,
        shuffle: bool, 
        unlabeled: bool = False, 
        labeled_size: int = None,
        unlabeled_batch_size: Union[int, float] = None, 
        unlabeled_num_worker: int = None,
        seed: int = None, 
        ):
        """return the dataloader for the dataset
        if the use both labeled and unlabeled set the flag unlabeled to true and other unlabeled input
        else don't fill the unlabled arguments

        Args:
            labeled_batch_size (int): _description_
            labeled_num_worker (int): _description_
            shuffle (bool): _description_
            unlabeled (bool, optional): _description_. Defaults to False.
            labeled_size (int, float): number of labeled image, if float divide by percentage 
            unlabeled_batch_size (int, optional): _description_. Defaults to None.
            unlabeled_num_worker (int, optional): _description_. Defaults to None.
            seed (int, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        
        raise NotImplementedError("Subclasses should implement this!")

    
class MTDataset(BaseDataset):
    def get_dataloader(
        self, 
        labeled_batch_size: int, 
        labeled_num_worker: int,
        shuffle: bool, 
        unlabeled: bool = False, 
        labeled_size: int = None,
        unlabeled_batch_size: Union[int, float] = None, 
        unlabeled_num_worker: int = None,
        seed: int = None, 
        ):
        """return the dataloader for the dataset
        if the use both labeled and unlabeled set the flag unlabeled to true and other unlabeled input
        else don't fill the unlabled arguments

        Args:
            labeled_batch_size (int): _description_
            labeled_num_worker (int): _description_
            shuffle (bool): _description_
            unlabeled (bool, optional): _description_. Defaults to False.
            labeled_size (int, float): number of labeled image, if float divide by percentage 
            unlabeled_batch_size (int, optional): _description_. Defaults to None.
            unlabeled_num_worker (int, optional): _description_. Defaults to None.
            seed (int, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        
        if self.train:
            if unlabeled:
                X_label, X_unlabeled, y_labeled ,y_unlabeled = self.split_labeled_unlabeled_data(X = self.data, y = self.targets, labeled_size = labeled_size, random_state = seed)
                labeled_ds = MTLabelDataset(X_label, y_labeled, self.transforms)
                unlabeled_ds = MTUnLabelDataset(X_unlabeled, self.transforms)
                unlabeled_dataloader = DataLoader(unlabeled_ds, unlabeled_batch_size, shuffle, num_workers=unlabeled_num_worker)
                labeled_dataloader = DataLoader(labeled_ds, labeled_batch_size, shuffle, num_workers=labeled_num_worker)
                return labeled_dataloader, unlabeled_dataloader
            
            ds = MTLabelDataset(self.data, self.targets, self.transforms)
            labeled_dataloader = DataLoader(ds, labeled_batch_size, shuffle, num_workers=labeled_num_worker)
            return labeled_dataloader
        
        ds = BasicLabelDataset(self.data, self.targets, self.transforms)
        test_dataloader = DataLoader(ds, labeled_batch_size, shuffle, num_workers=labeled_num_worker)
        
        return test_dataloader

class MySVHNMT(MTDataset):
    split_list = {
        "train": [
            "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
            "train_32x32.mat",
            "e26dedcc434d2e4c54c9b2d4a06d8373",
        ],
        "test": [
            "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
            "test_32x32.mat",
            "eb5a983be6a315427106f1b164d9cef3",
        ],
        "extra": [
            "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
            "extra_32x32.mat",
            "a93ce644f1a588dc4d68dda5feec44a7",
        ],
    }

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transforms: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        if isinstance(root, str):
            root = os.path.expanduser(root)
        self.root = root
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat["X"]
        # loading from the .mat file gives an np.ndarray of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.targets = loaded_mat["y"].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.targets, self.targets == 10, 0)
        # self.data = np.transpose(self.data, (3, 2, 0, 1))
        self.data = np.transpose(self.data, (3, 0, 1, 2))
        
        if split == "train" or split == "extra":
            self.train = True
        else:
            self.train = False
            
        self.transforms = transforms
        
    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)
    
    def download(self) -> None:
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

class MyCifar10MT(MTDataset):
    
    base_folder = "./cifar-10-batches-py"
    meta_data = "batches.meta"
    training_list = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
        ]
    test_list = [
        "test_batch",
        ]
    
    def __init__(self, train: bool, transforms: Transform):
        """

        Args:
            train (Bool): if True this will read train dataset else it will read test dataset
            transforms (Transform): the transformations that will be apply to the dataset
        """
        super().__init__(train, transforms)
        
        self.data = []
        self.targets = []
        
        if train:
            self.file_list = self.training_list
        else:
            self.file_list = self.test_list
        
        for file_name in self.file_list:
            fpath = os.path.join(os.path.curdir, self.base_folder, file_name)
            with open(fpath, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry["data"])
                self.targets.extend(entry["labels"])
        
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1)) 
        
    def _load_meta(self):
        path = os.path.join(os.path.curdir, self.base_folder, self.meta_data)
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta_data["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        
    
class BasicLabelDataset(Dataset):
    
    def __init__(self, data, targets, transforms):
        super().__init__()
        self.data = data
        self.targets = targets
        self.transforms = transforms
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        
        # return two different view at once
        if self.transforms is not None:
            return self.transforms(img), target
        return img, target

class BasicUnLabelDataset(Dataset):
    def __init__(self, data, transforms):
        super().__init__()
        self.data = data
        self.transforms = transforms
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        
        if self.transforms is not None:
            return self.transforms(img)
        
        return img
    
    
class MTLabelDataset(BasicLabelDataset):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        
        # return two different view at once
        if self.transforms is not None:
            return self.transforms(img), self.transforms(img) ,target
        return img, img, target
    
class MTUnLabelDataset(BasicUnLabelDataset):
    
    def __getitem__(self, index):
        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        
        if self.transforms is not None:
            return self.transforms(img), self.transforms(img)
        
        return img, img
    
# class CPS

