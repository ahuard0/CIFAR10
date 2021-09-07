# -*- coding: utf-8 -*-
r"""
Created on Mon Sep  6 19:55:13 2021

CIFAR-10 Dataset Implementation, Built from Scratch
    CIFAR10 dataset downloaded locally to A:\CIFAR\cifar-10-batches-py\
    from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    Dataset created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton

This dataset is an implementation of a PyTorch Dataset, which is intended to be
used with a PyTorch DataLoader.

@author: Andrew Huard
"""

from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class CIFAR10(Dataset):
    """
    The CIFAR-10 dataset
    https://www.cs.toronto.edu/~kriz/cifar.html
    
    The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes,
    with 6000 images per class. There are 50000 training images and 10000 test
    images.
    
    The dataset is divided into five training batches and one test batch, each
    with 10000 images. The test batch contains exactly 1000 randomly-selected
    images from each class. The training batches contain the remaining images
    in random order, but some training batches may contain more images from one
    class than another. Between them, the training batches contain exactly 5000
    images from each class.

    Instructions:
        Run this script directly to print debug information to the console.
        This PyTorch Dataset is compatible with PyTorch DataLoader.
        
    Note:
        The parent class requires overriding __getitem__() and __len__().
        These methods are the minimum required interfaces required by PyTorch
        Datasets.  PyTorch DataLoader will call both functions as they traverse
        the dataset.
    """
    
    def __init__(self, bool_train=False, bool_tensor=False):
        """
        Initialization function for the dataset.
        
        Parameters
        ----------
        bool_train : TYPE, optional
            A boolean switch used to load either the training dataset (True)
            or the validation dataset (False). The default is False.
        bool_tensor : TYPE, optional
            A boolean switch used to output either a PIL image (False) or a
            PyTorch Tensor (True) when retrieving an item from the dataset.
            The default is False.

        Returns
        -------
        None.

        """
        self.TRAIN_BOOL = bool_train
        self.PATH = r'A:\CIFAR\cifar-10-batches-py'
        self.bool_tensor = bool_tensor
        
        # Initialize Archives
        self.dict_batch_meta = self.unpickle(self.PATH+r"\batches.meta")
        if bool_train:
            self.dict_batch_1 = self.unpickle(self.PATH+r"\data_batch_1")
            self.dict_batch_2 = self.unpickle(self.PATH+r"\data_batch_2")
            self.dict_batch_3 = self.unpickle(self.PATH+r"\data_batch_3")
            self.dict_batch_4 = self.unpickle(self.PATH+r"\data_batch_4")
            self.dict_batch_5 = self.unpickle(self.PATH+r"\data_batch_5")
        else:
            self.dict_batch_test = self.unpickle(self.PATH+r"\test_batch")
        
        # Generate Meta
        self.label_dict = dict()
        labels = self.getMetaLabelsList()
        for idx, val in enumerate(labels):
            self.label_dict[idx] = val
        
    def __len__(self):
        """
        A required method by PyTorch Datasets, which is used to interface
        with a PyTorch DataLoader.
        
        Usage:
            length = len(dataset)

        Returns
        -------
        count : integer
            Returns the number of image files in the dataset.

        """
        if self.TRAIN_BOOL is True:
            count = len(self.dict_batch_1[b'data'])
            count += len(self.dict_batch_2[b'data'])
            count += len(self.dict_batch_3[b'data'])
            count += len(self.dict_batch_4[b'data'])
            count += len(self.dict_batch_5[b'data'])
        else:
            count = len(self.dict_batch_test[b'data'])
        return count
        
    def __getitem__(self, ndx):
        """
        A required method by PyTorch Datasets, which is used to interface
        with PyTorch Dataloader.
        
        Usage:
            img, labelID = dataset[ndx]

        Parameters
        ----------
        ndx : integer
            Index of the item to be returned by the dataset.

        Returns
        -------
        output : PIL Image or PyTorch Tensor
            Returns a PIL Image by default or a PyTorch Tensor if the dataset
            is configured with the boolean flag bool_tensor=True.
        labelID : integer
            The integer index of the image label, which is indexed from 0-9.
            Use getMetaLabelsList for a list of string labels ordered by ID
            index value.

        """
        if self.TRAIN_BOOL is True:
            if ndx < 10000:
                archive = self.dict_batch_1
            elif ndx < 20000:
                archive = self.dict_batch_2
            elif ndx < 30000:
                archive = self.dict_batch_3
            elif ndx < 40000:
                archive = self.dict_batch_4
            elif ndx < 50000:
                archive = self.dict_batch_5
            idx = ndx % 10000
        else:
            archive = self.dict_batch_test
            idx = ndx
            
        imgArray = archive[b'data'][idx]
        labelID = archive[b'labels'][idx]
        
        # Image is 3x32x32
        arrayRGB = np.zeros((32, 32, 3), dtype=np.uint8)
        arrayRGB[:, :, 0] = imgArray[:1024].reshape((32, 32))
        arrayRGB[:, :, 1] = imgArray[1024:1024*2].reshape((32, 32))
        arrayRGB[:, :, 2] = imgArray[1024*2:].reshape((32, 32))
        pil_image = Image.fromarray(arrayRGB, mode='RGB')
        
        # Apply Transform
        if self.bool_tensor:
            output = transforms.ToTensor()(pil_image)
        else:
            output = pil_image
            
        return output, labelID

    def getMetaLabelsList(self):
        """
        Label list provided by the dataset author in the meta file.
        This function is used to test and verify the dataset's integrity.
        
        Returns
        -------
        output : list
            Returns a list of labels in order of their corresponding integer
            ID values from 0-9.
        """
        return self.dict_batch_meta[b'label_names']
    
    def getMetaLabelsCount(self):
        """
        Helper function to provide the number of labels, which is always 10.
        This function is used to test and verify the dataset's integrity.

        Returns
        -------
        integer
            Number of Labels.

        """
        return len(self.label_dict)
    
    def getMetaLabel(self, idx):
        """
        Returns the string label corresponding to an integer label ID.

        Parameters
        ----------
        idx : integer
            Index of the CIFAR-10 image label.  Ranges from 0-9.

        Returns
        -------
        string
            The CIFAR-10 image label.

        """
        return self.label_dict[idx].decode("utf-8")
        
    def unpickle(self, file):
        """
        Returns a dictionary containing the contents of a Pickled archive.

        Parameters
        ----------
        file : string
            Filepath to the archive file to be extracted.

        Returns
        -------
        dictionary
            The archive file's contents.

        """
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict


def printMetaInfo(dataset):
    """
    Diagnostic function that prints a summary of the dataset's meta file
    contents.
    
    Used by main() for diagnostic testing only.

    Parameters
    ----------
    dataset : CIFAR10 Class object
        Dataset with meta information to be printed to the console.

    Returns
    -------
    None

    """
    print("\nMeta Info")
    print("Labels: " + str(dataset.getMetaLabelsCount()))
    for idx in range(dataset.getMetaLabelsCount()):
        print("Key: {}, Value: {}".format(str(idx), str(dataset.getMetaLabel(idx))))


def printDatasetInfo(dataset, verbose=False):
    """
    Diagnostic function that prints a summary of the dataset file contents.
    
    Used by main() for diagnostic testing only.

    Parameters
    ----------
    dataset : CIFAR10 Class object
        Dataset with content information to be printed to the console.
        
    verbose : bool
        Switch that determines if complete information will be printed to the
        console.

    Returns
    -------
    None

    """
    
    print("\nDataset Info")
    print("Dataset Mode: " + ("Training" if dataset.TRAIN_BOOL else "Validation"))
    if verbose:
        if dataset.TRAIN_BOOL:
            print("\nTraining Batch 1: ")
            for key, value in dataset.dict_batch_1.items():
                print("Key: {}, Length: {}".format(str(key), str(len(value))))
            
            print("\nTraining Batch 2: ")
            for key, value in dataset.dict_batch_2.items():
                print("Key: {}, Length: {}".format(str(key), str(len(value))))
                
            print("\nTraining Batch 3: ")
            for key, value in dataset.dict_batch_3.items():
                print("Key: {}, Length: {}".format(str(key), str(len(value))))
                
            print("\nTraining Batch 4: ")
            for key, value in dataset.dict_batch_4.items():
                print("Key: {}, Length: {}".format(str(key), str(len(value))))
                
            print("\nTraining Batch 5: ")
            for key, value in dataset.dict_batch_5.items():
                print("Key: {}, Length: {}".format(str(key), str(len(value))))
            
        else:
            print("\nTest Batch: ")
            for key, value in dataset.dict_batch_test.items():
                print("Key: {}, Length: {}".format(str(key), str(len(value))))

        print("\nMeta Batch: ")
        for key, value in dataset.dict_batch_meta.items():
            print("Key: {}, Value: {}".format(str(key), str(value)))

            
    print("Samples: " + str(len(dataset)))

    img, label = dataset[0]
    print("First Sample: Image Size " + str(np.asarray(img).shape) + ", Type " + str(type(img)) + ", Label: " + str(label))


def main():
    """
    Main method used for testing the CIFAR10 class. This function is used
    for testing purposes and to provide very basic examples of how to access
    data within the CIFAR-10 dataset.

    Returns
    -------
    None.

    """
    print("Begin Testing Main")
    
    print("\nInitialize Dataset Class")
    dataset_train = CIFAR10(bool_train=True, bool_tensor=True)
    dataset_val = CIFAR10(bool_train=False)
    
    printMetaInfo(dataset_train)
    
    printDatasetInfo(dataset_val, verbose=True)
    printDatasetInfo(dataset_train, verbose=True)
    
    img, label = dataset_train[2]
    label_str = dataset_train.getMetaLabel(label)
    print("\nShowing First Image in Training Set: " + label_str)
    
    if dataset_train.bool_tensor is False:
        img.show()


if __name__ == "__main__":
    """
    Main Function executes if python executes directly on this class file.
    This function is intended for diagnostic purposes only.  It is completely
    unused by PyTorch DataLoader.
    """
    main()
