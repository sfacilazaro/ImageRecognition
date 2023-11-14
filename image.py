#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 10:41:42 2023

@author: sergio
"""

import numpy as np
import tensorflow as tf
import os
import cv2

def acc(pred,targ):
    rows, cols = np.shape(targ-pred)
    a = 100 - sum(sum(np.abs(np.round(pred) - targ)))/(rows*cols) * 100
    return a

def WheatherMan(nIT,trainPath,testPath):
    """
    N --------- number of examples in the training set
    nIT ------- number of iterations for backprop
    trainPath - path to the training dataset
    testPath -- path to the testing dataset
    """
    
    width  = 128
    height = 128
    channs = 3
    
    #TRAIN
    # Initialize empty lists to store images and labels
    images = []
    labels = []
    
    # Create a mapping of class names to label indices
    classMapping = {folder: idx for idx, folder in enumerate(os.listdir(trainPath))}
    
    # Loop through subdirectories (classes) in the dataset folder
    for classFolder in os.listdir(trainPath):
        classPath = os.path.join(trainPath, classFolder)
        classLabel = classMapping[classFolder]
        
        # Loop through image files in the class folder
        for imageFile in os.listdir(classPath):
            imagePath = os.path.join(classPath, imageFile)
            
            # Read the image using OpenCV and convert it to grayscale or color as needed
            image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
            
            # Resize the image here if needed
            image = cv2.resize(image, (width, height))
            
            # Append the image and its corresponding label to the lists
            images.append(image)
            labels.append(classLabel)
    
    # Convert the image and label lists to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)
    nLabs = len(np.unique(labels))
    
    # shuffle
    shuffle = np.arange(len(images))
    np.random.shuffle(shuffle)
    X = images[shuffle]
    Y = labels[shuffle]
    
    L = tf.keras.utils.to_categorical(Y, num_classes=nLabs)
        
    myman = tf.keras.Sequential([        
        tf.keras.layers.Input(shape=(width, height, channs)),  # Input layer based on image shape
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'), # 32-64 and (3,3) first, more complex if needed 128,(5,5),(7,7)
        tf.keras.layers.MaxPooling2D((2, 2)),                  # downsample taking the max out of a 2x2 window
        tf.keras.layers.Flatten(),                             # unroll
        tf.keras.layers.Dense(128, activation='relu'),         # just another wednesday
        tf.keras.layers.Dense(nLabs, activation='softmax')     # just another thursday
    ])

    myman.compile(
        loss='categorical_crossentropy',                       
        optimizer='adam',                                      
        metrics=['accuracy']                                   
        
    )
    
    history = myman.fit(x=X, y=L, epochs=nIT, batch_size=len(Y))
    H = myman.predict(X)
    accTrain = acc(H, L)
    
    #TEST
    # Initialize empty lists to store images and labels
    images = []
    labels = []
    
    # Create a mapping of class names to label indices
    classMapping = {folder: idx for idx, folder in enumerate(os.listdir(testPath))}
    
    # Loop through subdirectories (classes) in the dataset folder
    for classFolder in os.listdir(testPath):
        classPath = os.path.join(testPath, classFolder)
        classLabel = classMapping[classFolder]
        
        # Loop through image files in the class folder
        for imageFile in os.listdir(classPath):
            imagePath = os.path.join(classPath, imageFile)
            
            # Read the image using OpenCV and convert it to grayscale or color as needed
            image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
            
            # Resize the image here if needed
            image = cv2.resize(image, (width, height))
            
            # Append the image and its corresponding label to the lists
            images.append(image)
            labels.append(classLabel)
    
    # Convert the image and label lists to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)
    nLabs = len(np.unique(labels))

    # shuffle
    shuffle = np.arange(len(images))
    np.random.shuffle(shuffle)
    X = images[shuffle]
    Y = labels[shuffle]
    
    L = tf.keras.utils.to_categorical(Y, num_classes=nLabs)
    
    loss, accu = myman.evaluate(X, L)
    
    H = myman.predict(X)
    accTest = acc(H, L)
    
    return loss, accu, accTrain, accTest
