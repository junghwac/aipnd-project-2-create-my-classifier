'''
# PROGRAMMER: Junghwa C.
# DATE CREATED: 2020-10-16
# REVISED DATE: 2020-10-17

A pretrained model will be downloaded and trained with a new dataset.
After training, checkpoint.pth will be saved for inference.
'''

from workspace_utils import active_session
from data_utils import data_utils

import torch 
from torch import nn, optim 
import torch.nn.functional as F
from torchvision import models

resnet50 = models.resnet50(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
models = {'resnet': resnet50, 'vgg': vgg16}
processors = {'cpu': 'cpu', 'gpu': 'cuda'}

def classifier(data_dir, device, arch, hidden, lr, epoch, checkpoint_pth):

    train_dataloaders, valid_dataloaders, train_datasets = data_utils(data_dir)
    processor = processors[device]
     
    if processor == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model = models[arch]
    for param in model.parameters():
        param.requires_grad = False
    
    hidden_unit = hidden
    if arch[0] == 'v': 
        model.classifier = nn.Sequential(nn.Linear(25088, hidden_unit),         
                                         nn.ReLU(),
                                         nn.Dropout(p = 0.2), 
                                         nn.Linear(hidden_unit, 102),
                                         nn.LogSoftmax(dim = 1))
        optimizer = optim.Adam(model.classifier.parameters(), lr)
        model.fc = None 
    else: 
        model.fc = nn.Sequential(nn.Linear(2048, hidden_unit),         
                                         nn.ReLU(),
                                         nn.Dropout(p = 0.2), 
                                         nn.Linear(hidden_unit, 102),
                                         nn.LogSoftmax(dim = 1))
        optimizer = optim.Adam(model.fc.parameters(), lr)
        model.classifier = None
        
    criterion = nn.NLLLoss()  
    model.to(device)

    epochs = epoch
    steps = 0
    train_loss = 0
    print_every = 5

    with active_session():
        for e in range(epochs):
            for inputs, labels in train_dataloaders:
                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                logps = model(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        
                if steps % print_every == 0:
                    model.eval()
                    valid_loss = 0
                    accuracy = 0
                
                    with torch.no_grad():
                        for inputs, labels in valid_dataloaders:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model(inputs)
                            loss = criterion(logps, labels)
                            valid_loss += loss.item()

                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim = 1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor))
                    print("Epoch: {}/{}..".format(e + 1, epochs),
                          "Training Loss: {:.3f}".format(train_loss/steps),
                          "Validation Loss: {:.3f}".format(valid_loss/len(valid_dataloaders)),
                          "Validation Accuracy: {:.3f}..".format(accuracy/len(valid_dataloaders)))
                
                    train_loss = 0
                    model.train()
                        
    checkpoint = {'model': arch,
                  'hidden_unit': hidden_unit,
                  'output_unit': 102,
                  'epochs': epoch,
                  'learnrate': lr,
                  'classifier': model.classifier,
                  'fc': model.fc, 
                  'state_dict': model.state_dict(), 
                  'optimizer_dict': optimizer.state_dict(),
                  'class_to_idx': train_datasets.class_to_idx}

    torch.save(checkpoint, checkpoint_pth)
    