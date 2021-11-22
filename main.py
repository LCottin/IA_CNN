#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:40:16 2021

@author: tnieddu and lcottin
"""

from Training import Train
from Testing import Test
import torch

#uncomment to work on google colab
"""
from google.colab import drive
drive.mount("/content/gdrive")
!unzip gdrive/MyDrive/imgs_frelon.zip > /dev/null
"""

def main():
    """ main function """
    
    #parameters
    model       = './frelon.pth'
    training    = False
   
    if training is True:
        print("Training begins ...")
        train       = Train(epochs = 3, trained = model)
        new_model   = train.train()
        print("Training ends.")
    
    else :
        new_model = torch.load(model)

    print("Testing begins ...")
    test = Test(model = new_model, num_img = 20)
    test.test()
    print("Testing ends.")
    
if __name__ == '__main__':
    main()