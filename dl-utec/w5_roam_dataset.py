import pandas as pd
import numpy as np
import os

print(os.listdir('./data/kaggle-self-driving/'))

labels = { 1:'car', 2:'truck', 3:'person', 4:'bicycle', 5:'traffic light'}

labels_train = pd.read_csv('./data/kaggle-self-driving/labels_train.csv')
labels_trainval = pd.read_csv('./data/kaggle-self-driving/labels_trainval.csv')
labels_val = pd.read_csv('./data/kaggle-self-driving/labels_val.csv')

print(labels_train['class_id'].value_counts())


"""
https://www.kaggle.com/code/maryamnoroozi68/object-detection-by-using-yolov8#-Model
"""
