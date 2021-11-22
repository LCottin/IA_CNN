#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os

class Test:
    def __init__(self, model, num_img = 10):
        self.dir                = './imgs_frelon/'
        self.device             = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model              = model
        self.num_img            = num_img
        self.batch_size         = 64
        
        self.dataset            = {}
        self.loader             = {}
        self.dataset_sizes      = {}
        self.classes            = []
        
        self.data_transforms    = { 'imgs_train':   transforms.Compose([
                                                    transforms.Resize(224),
                                                    transforms.RandomHorizontalFlip(),
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
        
        self.loader         = {x: DataLoader(self.dataset[x], batch_size = self.batch_size, shuffle = True, num_workers = 4)
                                for x in ['imgs_train', 'imgs_test']}

        self.dataset_sizes  = {x: len(self.dataset[x]) 
                                for x in ['imgs_train', 'imgs_test']}

        self.classes        = self.dataset['imgs_train'].classes

        
    def show_images(self, image, title = None):
        """ Prints an image"""
        
        image   = image.numpy().transpose((1, 2, 0))
        mean    = np.array([0.485, 0.456, 0.406])
        std     = np.array([0.229, 0.224, 0.225])
        image   = std * image + mean
        image   = np.clip(image, 0, 1)
        
        plt.imshow(image)
        
        if title is not None:
            plt.title(title)
            
        plt.pause(0.001)  
        
    def test(self):
        """ Tests somes images"""
        was_training    = self.model.training
        images          = 0
        fig             = plt.figure()
        
        self.model = self.model.to(self.device)
        self.model.eval()
        self.load_data()

        with torch.no_grad():
            #
            for inputs, labels in self.loader['imgs_test']:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                #Computes the output
                outputs  = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                #Shows the images
                for j in range(inputs.size()[0]):
                    images += 1
                    ax      = plt.subplot(self.num_img//2, 2, images)
                    ax.axis('off')
                    ax.set_title('{}'.format(self.classes[preds[j]]))
                    self.show_images(inputs.cpu().data[j])

                    if images == self.num_img:
                        self.model.train(mode=was_training)
                        return
                    
            self.model.train(mode=was_training)

if __name__ == "__main__":
    test = Test('frelon.pth')
    test.test()