# -*- coding: utf-8 -*-
r"""
Created on Mon Sep  6 19:55:13 2021

Test Bench Function for the CIFAR10 dataset class.

CIFAR-10 Dataset Implementation, Built from Scratch (Test Bench)
    CIFAR10 dataset downloaded locally to A:\CIFAR\cifar-10-batches-py\
    from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    Dataset created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton

This dataset is an implementation of a PyTorch Dataset, which is intended to be
used with a PyTorch DataLoader.

@author: Andrew Huard
"""
from CIFAR10 import CIFAR10
import numpy as np


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
