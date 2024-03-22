import numpy as np
import decord
import torch
import os

from gluoncv.torch.utils.model_utils import download
from gluoncv.torch.data.transforms.videotransforms import video_transforms, volume_transforms
from gluoncv.torch.engine.config import get_cfg_defaults
from gluoncv.torch.model_zoo import get_model


video_root_fname='F:/DataforExtract/H&N1/VideoforFeatures/'
features_root_path='F:/DataforExtract/H&N1/ircsn_deep_features/'

video_name_list=os.listdir(video_root_fname)

config_file = 'F:/DataforExtract/r2plus1d_v2_resnet152_kinetics400.yaml'
for videoID in range(80):
    video_fname = video_root_fname + video_name_list[videoID]
    print(video_name_list[videoID])
    features_path=features_root_path+video_name_list[videoID][:-3]+'txt'
    vr = decord.VideoReader(video_fname)

    frame_id_list = range(0, len(vr), 25)

    video_data = vr.get_batch(frame_id_list).asnumpy()


    crop_size = 224
    short_side_size = 256
    transform_fn = video_transforms.Compose([video_transforms.Resize(short_side_size, interpolation='bilinear'),
                                             video_transforms.CenterCrop(size=(crop_size, crop_size)),
                                             volume_transforms.ClipToTensor(),
                                             video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    clip_input = transform_fn(video_data)
#print('Video data is downloaded and preprocessed.')

    
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_file)
    model = get_model(cfg)
    model.eval()
#print('%s model is successfully loaded.' % cfg.CONFIG.MODEL.NAME)

    with torch.no_grad():
        pred = model(torch.unsqueeze(clip_input, dim=0)).numpy()
#print('The input video clip is classified to be class %d' % (np.argmax(pred)))
    
    f=open(features_path,'w')

    for i in range(400):
        print (pred[0,i],file=f)
    f.close()