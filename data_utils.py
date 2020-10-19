'''
# PROGRAMMER: Junghwa C.
# DATE CREATED: 2020-10-16
# REVISED DATE: 2020-10-17

Dataset loading and preprocessing function before training a CNN.
'''

import os
from os import listdir
import torch
from torchvision import datasets, transforms

def data_utils(data_dir):
    data_dir = str(data_dir)
    train_dir = data_dir + '/train'  
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_files = listdir(train_dir)
    valid_files = listdir(valid_dir)
    test_files = listdir(test_dir)
    # print(len(train_files))
    
    # check out and remove hidden files in train dataset
    for idx in range(0, len(train_files)):
        class_dir = train_dir + '/' + str(idx+1)
        class_files = listdir(class_dir)
        try:
            if class_files[idx][0].startswith('.'):
                print("** Waring: hidden files exist in directory:", class_files[idx])
                os.remove(class_files[idx])
        except IndexError:
            pass
    
    # check out and remove hidden files in valid dataset
    for idx in range(0, len(valid_files)):
        class_dir = valid_dir + '/' + str(idx+1)
        class_files = listdir(class_dir)
        try: 
           if class_files[idx][0].startswith('.'):
                print("** Waring: hidden files exist in directory:", class_files[idx])
                os.remove(class_files[idx])
        except IndexError:
            pass
        
    # check out and remove hidden files in test dataset
    for idx in range(0, len(test_files)):
        class_dir = test_dir + '/' + str(idx+1)
        class_files = listdir(class_dir)
        # print('before', len(class_files))
        try: 
           if class_files[idx][0].startswith('.'):
                print("** Waring: hidden files exist in directory:", class_files[idx])
                os.remove(class_files[idx])
        except IndexError:
            pass
        # print('after', len(class_files))
        
        
    # loading and preprocessing datasets    
    train_transforms = transforms.Compose([transforms.RandomRotation(45), 
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.229, 0.224, 0.225], [0.485, 0.456, 0.406])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.229, 0.224, 0.225], [0.485, 0.456, 0.406])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.229, 0.224, 0.225], [0.485, 0.456, 0.406])])

    # load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms) 
    valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)

    # using the image datasets and the trainforms, define the dataloaders
    train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size = 63, shuffle=True) 
    valid_dataloaders = torch.utils.data.DataLoader(valid_datasets, batch_size = 63, shuffle=True)
    test_dataloaders = torch.utils.data.DataLoader(test_datasets, batch_size = 63, shuffle=True)

    return train_dataloaders, valid_dataloaders, train_datasets
  