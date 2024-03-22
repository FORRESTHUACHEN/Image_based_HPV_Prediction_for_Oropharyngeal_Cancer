# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 14:41:38 2023

@author: chenj
"""

import os 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from itertools import chain
import numpy as np
import pandas as pd
from combat.pycombat import pycombat

label=pd.read_excel('F:/DataforExtract/RADCURE_HPV_Label.xlsx')

Radiomics_feature=pd.read_excel('F:/DataforExtract/Radiomics_summary_normalized.xlsx')

Deep_feature=pd.read_excel('F:/DataforExtract/RADCURE_3dnld_normalized_deep_feature_summary.xlsx')
n10l_Deep_feature=pd.read_excel('F:/DataforExtract/Deep_feature_summary_normalized.xlsx')
ircsn_Deep_feature=pd.read_csv('F:/DataforExtract/RADCURE_ircsn_normal_deep_feature_summary.csv')
# Radiomics_feature_group=Radiomics_feature.groupby(['Patient_ID'])

# Deep_feature_group=Deep_feature.groupby(['Patient_ID'])

# Radiomics_label_unique=list(Radiomics_feature.loc[:,'Patient_ID'].unique())
manu_label=pd.read_excel('F:/DataforExtract/Manufacturer_information_v3.xlsx')

label_available=[]


manu_list=list(manu_label.iloc[:,0])
HPV_list=list(label.iloc[:,0])
label_list=set(manu_list) & set(HPV_list)

for i in range(len(Deep_feature)):
    if Deep_feature.iloc[i,0] in label_list:
        
        label_available.append(Deep_feature.iloc[i,0])

available_radiomics=Radiomics_feature[Radiomics_feature['Patient_ID']==label_available[0]]
available_deep_feature=Deep_feature[Deep_feature['Patient_ID']==label_available[0]]
available_n10l_deep_feature=n10l_Deep_feature[n10l_Deep_feature['Patient_ID']==label_available[0]]
available_ircsn_deep_feature=ircsn_Deep_feature[ircsn_Deep_feature['Patient_ID']==label_available[0]]
available_manu_label=manu_label[manu_label['Patient_ID']==label_available[0]]
label_selected=label[label['Patient_ID']==label_available[0]]
for i in range(1,len(label_available)):
    available_radiomics=pd.concat([available_radiomics,Radiomics_feature[Radiomics_feature['Patient_ID']==label_available[i]]],axis=0)
    available_deep_feature=pd.concat([available_deep_feature,Deep_feature[Deep_feature['Patient_ID']==label_available[i]]],axis=0)
    
    available_n10l_deep_feature=pd.concat([available_n10l_deep_feature,n10l_Deep_feature[n10l_Deep_feature['Patient_ID']==label_available[i]]],axis=0)
    available_ircsn_deep_feature=pd.concat([available_ircsn_deep_feature,ircsn_Deep_feature[ircsn_Deep_feature['Patient_ID']==label_available[i]]],axis=0)

    available_manu_label=pd.concat([available_manu_label,manu_label[manu_label['Patient_ID']==label_available[i]]],axis=0)
    label_selected=pd.concat([label_selected,label[label['Patient_ID']==label_available[i]]],axis=0)


# deepfeature_value=available_deep_feature.iloc[:,1:]

# #deepfeautre_label_list=list(deepfeature_label)

# deepfeature_label=available_manu_label.iloc[:,1]
# deepfeature_corrected = pycombat(deepfeature_value.T,deepfeature_label.T)

# deepfeature_corrected =deepfeature_corrected.T

# available_deep_feature.iloc[:,1:]=deepfeature_corrected


# n10l_deep_feature_value=available_n10l_deep_feature.iloc[:,1:]

# #deepfeautre_label_list=list(deepfeature_label)

# deepfeature_label=available_manu_label.iloc[:,1]
# n10l_deep_feature_corrected = pycombat(n10l_deep_feature_value.T,deepfeature_label.T)

# n10l_deep_feature_corrected =n10l_deep_feature_corrected.T

# available_n10l_deep_feature.iloc[:,1:]=n10l_deep_feature_corrected 



# ircsn_deep_feature_value=available_ircsn_deep_feature.iloc[:,1:]

# #deepfeautre_label_list=list(deepfeature_label)

# deepfeature_label=available_manu_label.iloc[:,1]
# ircsn_deep_feature_corrected = pycombat(ircsn_deep_feature_value.T,deepfeature_label.T)

# ircsn_deep_feature_corrected =ircsn_deep_feature_corrected.T

# available_ircsn_deep_feature.iloc[:,1:]=ircsn_deep_feature_corrected


# # radiomics_label=np.zeros(len(Radiomics_feature))

# # for radio_label in range(len(Radiomics_feature)):
# #     radiomics_label[radio_label]=manu_label[manu_label['Patient ID']==Radiomics_feature.iloc[radio_label][0]].iloc[:,1]
# radiomics_value=available_radiomics.iloc[:,1:]

# radiomics_corrected=pycombat(radiomics_value.T,deepfeature_label.T)

# radiomics_corrected=radiomics_corrected.T

# available_radiomics.iloc[:,1:]=radiomics_corrected

# available_radiomics.to_csv('C:/Users/chenj/OneDrive/桌面/20th paper/combatcorrectedRadiomics.csv')
# available_deep_feature.to_csv('C:/Users/chenj/OneDrive/桌面/20th paper/combatcorrectedDeepfeatures.csv')

# writer1= tf.python_io.TFRecordWriter("D:/Datasets/Data Recovered/Rec Core Fle/SubtypesDecodingCore_featureTrain_HER2_combat10.tfrecords") 
# writer2= tf.python_io.TFRecordWriter("D:/Datasets/Data Recovered/Rec Core Fle/SubtypesDecodingCore_featureTest_HER2_combat10.tfrecords")
writer_root_path="F:/DataforExtract/slowfast+i3d_5_fold_files_TOSHIBA/"
#writer1= tf.python_io.TFRecordWriter("D:/Datasets/RADCURE/TFRecord/Train_combat4.tfrecords") 
#writer2= tf.python_io.TFRecordWriter("D:/Datasets/RADCURE/TFRecord/Test_combat4.tfrecords")
 

index=list(range(len(available_radiomics)))
np.random.shuffle(index)

count1=0
count2=0


# # # Radiomics_feature.to_csv('D:/Datasets/Data Recovered/DataSets/ACRIN-6698/Classifier-LSTM/combatcorrectedRadiomics.csv')
# # # Deep_feature.to_csv('D:/Datasets/Data Recovered/DataSets/ACRIN-6698/Classifier-LSTM/combatcorrectedDeepfeatures.csv')
for fold_number in range(5):
    writer1=tf.python_io.TFRecordWriter(writer_root_path+"Trainuncombatrunning2_"+str(fold_number)+".tfrecords")
    writer2=tf.python_io.TFRecordWriter(writer_root_path+"Testuncombatrunning2_"+str(fold_number)+".tfrecords")
    for i in range(len(available_radiomics)):
        Radiomics_features_value=np.array(available_radiomics.iloc[i,2:],dtype=np.float32)
        Deep_features_value=np.array(available_deep_feature.iloc[i,1:],dtype=np.float32)
        n10l_Deep_feature_value=np.array(available_n10l_deep_feature.iloc[i,1:],dtype=np.float32)
        ircsn_Deep_feature_value=np.array(available_ircsn_deep_feature.iloc[i,1:],dtype=np.float32)
        
        label_value=np.array(label_selected.iloc[i,1],dtype=np.int32)
        

    
        Radiomics_feature_group_sample_data_towrite=Radiomics_features_value.tobytes()
        Deep_feature_group_sample_data_towrite=Deep_features_value.tobytes()
        n10l_Deep_feature_group_sample_data_towrite=n10l_Deep_feature_value.tobytes()
        
        ircsn_Deep_feature_group_sample_data_towrite=ircsn_Deep_feature_value.tobytes()
        label_towrite=label_value.tobytes()
    
        example = tf.train.Example(features=tf.train.Features(feature={
           
            'RadiomicFeatures': tf.train.Feature(bytes_list=tf.train.BytesList(value=[Radiomics_feature_group_sample_data_towrite])),
            'DeepFeatures': tf.train.Feature(bytes_list=tf.train.BytesList(value=[Deep_feature_group_sample_data_towrite])),
            'n10lDeepFeatures': tf.train.Feature(bytes_list=tf.train.BytesList(value=[n10l_Deep_feature_group_sample_data_towrite])),
            'ircsnDeepFeatures': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ircsn_Deep_feature_group_sample_data_towrite])),
            'ValidatedPhases':tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_towrite])),
            'Label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_towrite]))
            }))
    

        # writer1=tf.python_io.TFRecordWriter(writer_root_path+"Traincombat"+str(fold_num)+".tfrecords")
        # writer2=tf.python_io.TFRecordWriter(writer_root_path+"Testcombat"+str(fold_num)+".tfrecords")
        if i in index[205*fold_number:205*fold_number+205]:
            count1=count1+1
            writer2.write(example.SerializeToString())  
        #print(Radiomics_features_value.shape)
        #print(Deep_features_value.shape)
            print(label_value)
        else:
            count2=count2+1
            writer1.write(example.SerializeToString())  
            print(label_value)
    writer1.close()
    writer2.close()
