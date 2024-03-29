# Image based HPV Prediction for Oropharyngeal Cancer

Here is the source codes, Radiomics and deep features, data for statistical analysis, and Supporting Information for manuscript '**Human Papillomavirus (HPV) Prediction for Oropharyngeal Cancer Based on CT by Using Training-needless Features: a Multi-dataset Study**'  

Pipeline of study is shown as follow: while A. means Radiomics features and its related analysis workflow; B. means Deep features and its related analysis workflow.
![image](https://github.com/FORRESTHUACHEN/Image_based_HPV_Prediction_for_Oropharyngeal_Cancer/blob/main/Figures/Figure2.png)

Several steps are needed to execute our method (1)radiomics feature extraction; (2)deep feature extraction; (3) tfrecord files making; (4) model training and validation.

## (1)Radiomics Feature extraction

3D mask of regoin of interest is needed in radiomics feature extraction, mask files in RADCURE dataset are stored in RTSTRUC file, you need to use ['Pre_Processing For Radiomics
/Create_3D_structure_for_Radiomics_RADCURE.py'](https://github.com/FORRESTHUACHEN/Image_based_HPV_Prediction_for_Oropharyngeal_Cancer/blob/main/Pre_Processing%20For%20Radiomics/Create_3D_structure_for_Radiomics_RADCURE.py) to reconstrucated 3D image data and corresonding mask. 3D mask of ROI available in H&N 1 dataset.

Pyradiomics is used to extracted radiomics features, instructions of using Pyradiomics to extracted radiomics features can be found [here](https://pyradiomics.readthedocs.io/en/latest/)

Extracted and normalized radiomics features for RADCUE dataset and H&N 1 dataset can be found [here](https://github.com/FORRESTHUACHEN/Image_based_HPV_Prediction_for_Oropharyngeal_Cancer/tree/main/Extracted%20Features)

## (2)Deep Feature Extraction

For extracted deep feature from action recognition networks, we need to transfer DICOM data to videos, You can use ['DICOMtoVIDEO
/GenerateOPCVideo.py'](https://github.com/FORRESTHUACHEN/Image_based_HPV_Prediction_for_Oropharyngeal_Cancer/blob/main/DICOMtoVIDEO/GenerateOPCVideo.py) to finish this transform. An example of transferred video of patient 'RADCURE-0005' can be found in [here](https://github.com/FORRESTHUACHEN/Image_based_HPV_Prediction_for_Oropharyngeal_Cancer/tree/main/DICOMtoVIDEO).

Deep features extraction based on pretrained 'i3d_inceptionv1_kinetics400' can be finished by this [file](https://github.com/FORRESTHUACHEN/Image_based_HPV_Prediction_for_Oropharyngeal_Cancer/blob/main/DICOMtoVIDEO/FeatureExtractor.py). Deep feature extraction based on pretrained 'r2plus1d_v2_resnet152_kinetics400' can be finished by this [file](https://github.com/FORRESTHUACHEN/Image_based_HPV_Prediction_for_Oropharyngeal_Cancer/blob/main/DICOMtoVIDEO/feat_extract_pytorch.py). Before ues mentioned files, you need to implement 'GluonCV' and 'Pytroch' platform.

Extracted and normalilzed deep features for RADCUE dataset and H&N 1 can be found [here](https://github.com/FORRESTHUACHEN/Image_based_HPV_Prediction_for_Oropharyngeal_Cancer/blob/main/DICOMtoVIDEO/feat_extract_pytorch.py).

## (3) tfrecord files making

Before training and testing established models, you need to build tfrecord files at first, you can finish this action by using this [file](https://github.com/FORRESTHUACHEN/Image_based_HPV_Prediction_for_Oropharyngeal_Cancer/blob/main/TFRecords/tfwrite_3batch.py).Before running our codes, you need to implement 'tensorflow' platform at first

## (4) model training and validation

You can use file 'Classifierv2_3batch.py' to train and valid our build model. 


