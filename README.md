# Image_based_HPV_Prediction_for_Oropharyngeal_Cancer

Here is the source codes, Radiomics and deep features, data for statistical analysis, and Supporting Information for manuscript 'Human Papillomavirus (HPV) Prediction for Oropharyngeal Cancer Based on CT by Using Training-needless Features: a Multi-dataset Study'  

Pipeline of study is shown as follow: while A. means Radiomics features and its related analysis workflow; B. means Deep features and its related analysis workflow.
![image](https://github.com/FORRESTHUACHEN/Image_based_HPV_Prediction_for_Oropharyngeal_Cancer/blob/main/Figures/Figure2.png)

Several steps are needed to execute our method (1)radiomics feature extraction; (2)deep feature extraction; (3) tfrecord files making; (4) model training and validation.

## (1)Radiomics Feature extraction

3D mask of regoin of interest is needed in radiomics feature extraction, mask files in RADCURE dataset are stored in RTSTRUC file, you need to use ['Pre_Processing For Radiomics
/Create_3D_structure_for_Radiomics_RADCURE.py'](https://github.com/FORRESTHUACHEN/Image_based_HPV_Prediction_for_Oropharyngeal_Cancer/blob/main/Pre_Processing%20For%20Radiomics/Create_3D_structure_for_Radiomics_RADCURE.py) to reconstrucated 3D image data and corresonding mask. 3D mask of ROI available in H&N 1 dataset.

Pyradiomics is used to extracted radiomics features, instructions of using Pyradiomics to extracted radiomics features can be found [here](https://pyradiomics.readthedocs.io/en/latest/)

Extracted and normalized features for RADCUE dataset and H&N 1 dataset can be found [here](https://github.com/FORRESTHUACHEN/Image_based_HPV_Prediction_for_Oropharyngeal_Cancer/tree/main/Extracted%20Features)

## (2)Deep Feature Extraction

For extracted deep feature from action recognition networks, we need to transfer DICOM data to videos, You can use ['DICOMtoVIDEO
/GenerateOPCVideo.py'](https://github.com/FORRESTHUACHEN/Image_based_HPV_Prediction_for_Oropharyngeal_Cancer/blob/main/DICOMtoVIDEO/GenerateOPCVideo.py) to finish this transform. An example of transferred video of patient 'RADCURE-0005' can be found in here.
