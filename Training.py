#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import lr_scheduler
import os
from time import time
from copy import deepcopy

class Train:
    def __init__(self, epochs = 20, resnet = models.resnet18(pretrained = True), trained = 'frelon.pth'):
        self.epochs             = epochs
        self.device             = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.resnet             = resnet
        self.trained            = trained
        self.dir                = './imgs_frelon/'

        self.batch_size         = 10
        self.learning_rate      = 1e-3
        self.gamma              = 0.1
        self.momentum           = 0.9
        
        self.dataset            = {}
        self.loader             = {}
        self.dataset_sizes      = {}
        self.classes            = []
        
        self.train_loss         = []
        self.train_acc          = []
        self.val_loss           = []
        self.val_acc            = []
        self.learning_rates     = []
        
        self.data_transforms    = { 'imgs_train':   transforms.Compose([
                                                    transforms.Resize(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    #transforms.RandomVerticalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                                   
                                    'imgs_test':    transforms.Compose([
                                                    transforms.Resize(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
        
    def load_data(self):
        """ Loads data to train and test """
        
        self.dataset        = {x: datasets.ImageFolder(os.path.join(self.dir, x), self.data_transforms[x])
                                for x in ['imgs_train', 'imgs_test']}
        
        self.loader         = {x: DataLoader(self.dataset[x], batch_size = self.batch_size, shuffle = True, num_workers = 2)
                                for x in ['imgs_train', 'imgs_test']}

        self.dataset_sizes  = {x: len(self.dataset[x]) 
                                for x in ['imgs_train', 'imgs_test']}

        self.classes        = self.dataset['imgs_train'].classes


    def train(self, print_data = True):
        """ Trains the model over num_epoch """
        
        #saves elapsed time
        since = time()
        
        #loads data
        self.load_data()
        
        #optimizes the model
        self.resnet.fc  = nn.Linear(self.resnet.fc.in_features, 2)
        self.resnet     = self.resnet.to(self.device)
        criterion       = nn.CrossEntropyLoss()
        optimizer       = optim.SGD(self.resnet.parameters(), lr = self.learning_rate, momentum = self.momentum)
        scheduler       = lr_scheduler.StepLR(optimizer, step_size = 7, gamma = self.gamma)

        #saves best model weigths
        best_model_wts  = deepcopy(self.resnet.state_dict())
        best_acc        = 0.0

        for epoch in range(1, self.epochs + 1):
            print('Epoch {}/{}'.format(epoch, self.epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['imgs_train', 'imgs_test']:
                if phase == 'imgs_train':
                    self.resnet.train()  # Set model to training mode
                else:
                    self.resnet.eval()   # Set model to evaluate mode

                running_loss     = 0.0
                running_corrects = 0

                # Iterate over loader
                for inputs, labels in self.loader[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forwards
                    with torch.set_grad_enabled(phase == 'imgs_train'):
                        outputs     = self.resnet(inputs)
                        _, preds    = torch.max(outputs, 1)
                        loss        = criterion(outputs, labels)

                        # backwards and optimizes
                        if phase == 'imgs_train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss     += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                if phase == 'imgs_train':
                    scheduler.step()

                #Saves statistics
                epoch_loss  = running_loss / self.dataset_sizes[phase]
                epoch_acc   = running_corrects.double() / self.dataset_sizes[phase]

                if phase == 'imgs_train':
                    self.learning_rates.append(optimizer.param_groups[0]['lr'])
                    self.train_loss.append(epoch_loss)
                    self.train_acc.append(epoch_acc)
                else :
                    self.val_loss.append(epoch_loss)
                    self.val_acc.append(epoch_acc)

                print('{:14} Loss : {:1.4f}   Acc : {:1.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copies the best model
                if phase == 'imgs_test' and epoch_acc > best_acc:
                    best_acc        = epoch_acc
                    best_model_wts  = deepcopy(self.resnet.state_dict())

            print()

        time_elapsed = time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.resnet.load_state_dict(best_model_wts)

        #prints data if wanted
        if print_data is True:
            self.print_data()
        
        #saves and returns model
        torch.save(self.resnet, self.trained)
        return self.resnet

    def print_data(self):
        """ Prints data """
        x = [i for i in range(1, self.epochs + 1)]
        fig, axs = plt.subplots(2, 3)

        axs[0, 0].plot(x, self.train_loss, 'tab:red')
        axs[0, 0].set_title('Train Loss')

        axs[0, 1].plot(x, self.train_acc, 'tab:green')
        axs[0, 1].set_title('Train Accuracy')

        axs[0, 2].plot(x, self.learning_rates, 'tab:orange')
        axs[0, 2].set_title('Learning rate \nevolution')

        axs[1, 0].plot(x, self.val_loss, 'tab:red')
        axs[1, 0].set_title('\nTest Loss')

        axs[1, 1].plot(x, self.val_acc, 'tab:green')
        axs[1, 1].set_title('\nTest accuracy')

        axs[1, 2].plot(x, self.learning_rates, 'tab:blue')
        axs[1, 2].set_title('\nLearning rate \nevolution')

        for ax in axs.flat:
          ax.set(xlabel = 'epochs')

        fig.tight_layout()
        
if __name__ == '__main__':
    model = Train()
    model.train()