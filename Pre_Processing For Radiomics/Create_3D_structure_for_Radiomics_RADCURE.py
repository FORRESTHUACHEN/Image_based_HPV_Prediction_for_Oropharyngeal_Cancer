# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 03:56:43 2023

@author: chenj
"""
import pylidc as pl
from pylidc.utils import volume_viewer
import matplotlib.pyplot as plt
import numpy as np
import xlwt
import os
import pydicom
#from pydicom.dataset import Dataset, FileDataset
import SimpleITK as sitk
import pydicom_seg
from pydicom import dcmread
import cv2
import pandas as pd


Root_Path='D:/Datasets/RADCURE/RADCURE/'
Patient_list=os.listdir(Root_Path)
#Mask_Metadata_dir=pd.read_csv(ACRIN_Mask_Metadata)
number_of_imaging=[]
empty_list=[]
for i in range(len(Patient_list)):
    image_data_root_path=Root_Path+Patient_list[i]
    image_data_root_path_list=os.listdir(image_data_root_path)
    #number_of_imaging.append(len(image_data_root_path_list))
    for planningCT in range(len(image_data_root_path_list)):
        if "Planning CT" in image_data_root_path_list[planningCT]:
            image_data_planning_CT=image_data_root_path+'/'+image_data_root_path_list[planningCT]
            #print(image_data_planning_CT)
    
    if (len(os.listdir(image_data_planning_CT))<2):
        print(image_data_planning_CT)
        empty_list.append(image_data_planning_CT)