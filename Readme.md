# THUman-dynamic Dataset

### Structured Local Radiance Fields for Human Avatar Modeling
Zerong Zheng, Han Huang, Tao Yu, Hongwen Zhang, Yandong Guo, Yebin Liu.  CVPR 2022

[[Project Page]](http://www.liuyebin.com/slrf/slrf.html)

![teaser](./teaser.png)

This dataset contains three multi-view image sequences used in our paper "Structured Local Radiance Fields for Human Avatar Modeling". They are captured with 24 well-calibrated RGB cameras, with lengths ranging from 2500 to 5000 frames. We use the data to evaluate our method for building animatable human body avatars. 

We also provide the SMPL fitting in the dataset. 


## Agreement
1. The THUman-dynamic dataset (the "Dataset") is available for **non-commercial** research purposes only. Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, as training data for a commercial product, for commercial ergonomic analysis (e.g. product design, architectural design, etc.), or production of other artifacts for commercial purposes including, for example, web services, movies, television programs, mobile applications, or video games. The dataset may not be used for pornographic purposes or to generate pornographic material whether commercial or not. The Dataset may not be reproduced, modified and/or made available in any form to any third party without Tsinghua University’s prior written permission.

2. You agree **not to** reproduce, modified, duplicate, copy, sell, trade, resell or exploit any portion of the images and any portion of derived data in any form to any third party without Tsinghua University’s prior written permission.

3. You agree **not to** further copy, publish or distribute any portion of the Dataset. Except, for internal use at a single site within the same organization it is allowed to make copies of the dataset.

4. Tsinghua University reserves the right to terminate your access to the Dataset at any time.


## Download Instructions 
The dataset can be directly downloaded from the following links.

* Subject00: [this link](), 2500 frames in total, ~30 GB
* Subject01: [this link](), 5060 frames in total, ~30 GB
* Subject02: [this link](), 3110 frames in total, ~30 GB


Note again that by downloading the dataset you acknowledge that you have read the agreement, understand it, and agree to be bound by them. If you do not agree with these terms and conditions, you must not download and/or use the Dataset.


## Data Explanation
For each subject, we provide the multi-view images (```./subject0*/images/cam**/```) as well as the foreground segmentation (```./subject0*/masks/cam**/```), which are obtained using [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2). The calibration data is provided in ```calibration.json```, and the SMPL fitting in ```smpl_params.npz```. Some frames are losed during the capture process, and we provide their filename in ```missing_img_files.txt```. 

Here we provide a code snip to show how to parse and visualize the data:
```python
import os
import json
import numpy as np
import cv2 as cv
import torch
import smplx  # (please setup the official SMPL-X model according to: https://pypi.org/project/smplx/)

# subject, frame_num = './subject00', 2500
# subject, frame_num = './subject01', 5060
subject, frame_num = './subject02', 3110

# initialize smpl model
smpl = smplx.SMPLX(model_path='./smplx', gender='neutral', use_pca=False, num_pca_comps=45, flat_hand_mean=True, batch_size=1)

# load camera data
with open(os.path.join(subject, 'calibration.json'), 'r') as fp:
    cam_data = json.load(fp)
    
# load smpl data
smpl_data = np.load(os.path.join(subject, 'smpl_params.npz'), allow_pickle=True)
smpl_data = dict(smpl_data)
smpl_data = {k:torch.from_numpy(v.astype(np.float32)) for k,v in smpl_data.items()}

for frame_id in range(0, frame_num, 30):
    smpl_out = smpl.forward(
        global_orient=smpl_data['global_orient'][frame_id].unsqueeze(0), 
        transl=smpl_data['transl'][frame_id].unsqueeze(0), 
        body_pose=smpl_data['body_pose'][frame_id].unsqueeze(0), 
        jaw_pose=smpl_data['jaw_pose'][frame_id].unsqueeze(0), 
        betas=smpl_data['betas'][0].unsqueeze(0), 
        expression=smpl_data['expression'][frame_id].unsqueeze(0), 
        left_hand_pose=smpl_data['left_hand_pose'][frame_id].unsqueeze(0), 
        right_hand_pose=smpl_data['right_hand_pose'][frame_id].unsqueeze(0), 
    )
    smpl_verts = smpl_out.vertices  # smpl vertices in live poses
    smpl_verts = smpl_verts.detach().cpu().numpy().squeeze(0)

    smpl_proj_vis = []
    for cam_id in range(0, len(cam_data), 4):
        cam_sn = 'cam%02d' % cam_id
        
        img_fpath = os.path.join(subject, 'images/%s/%08d.jpg' % (cam_sn, frame_id))
        msk_fpath = os.path.join(subject, 'masks/%s/%08d.jpg' % (cam_sn, frame_id))
        
        if (not os.path.isfile(img_fpath)) or (not os.path.isfile(msk_fpath)):
            break

        img = cv.imread(img_fpath, cv.IMREAD_UNCHANGED)
        msk = cv.imread(msk_fpath, cv.IMREAD_GRAYSCALE)
        img = img*np.uint8(msk>128)[:, :, np.newaxis]   # remove background
        img_ = cv.resize(img, (img.shape[1]//2, img.shape[0]//2))

        # transform smpl from world to camera
        cam_R = np.array(cam_data[cam_sn]['R']).astype(np.float32).reshape((3, 3))
        cam_t = np.array(cam_data[cam_sn]['T']).astype(np.float32).reshape((3,))
        smpl_verts_cam = np.matmul(smpl_verts, cam_R.transpose()) + cam_t.reshape(1, 3)
        
        # project smpl vertices to the image        
        cam_K = np.array(cam_data[cam_sn]['K']).astype(np.float32).reshape((3, 3))
        cam_K *= np.array([img_.shape[1]/img.shape[1], img_.shape[0]/img.shape[0], 1.0], dtype=np.float32).reshape(3, 1)
        smpl_verts_proj = np.matmul(smpl_verts_cam/smpl_verts_cam[:, 2:], cam_K.transpose())

        # visualize the projection        
        smpl_verts_proj = np.round(smpl_verts_proj).astype(np.int32)
        smpl_verts_proj[:, 0] = np.clip(smpl_verts_proj[:, 0], 0, img_.shape[1] - 1)
        smpl_verts_proj[:, 1] = np.clip(smpl_verts_proj[:, 1], 0, img_.shape[0] - 1)
    
        for v in smpl_verts_proj:
            img_[v[1], v[0], :] = np.array([255, 255, 255], dtype=np.uint8)
        smpl_proj_vis.append(img_)

    if len(smpl_proj_vis) != 6: 
        continue

    vis = np.concatenate([
        np.concatenate(smpl_proj_vis[:3], axis=1), 
        np.concatenate(smpl_proj_vis[3:], axis=1), 
    ], axis=0)
    cv.imshow('vis', vis)
    cv.waitKey(1)
```
If everything is setup properly, you can see an animation like this:
<p align="center"> 
    <img src="0000.gif">
</p>


## Related Datasets from THU3DV Lab [[Link]](http://liuyebin.com/)
[[MultiHuman Dataset]](https://github.com/y-zheng18/MultiHuman-Dataset/) Containing 453 high-quality scans, each contains 1-3 persons. The dataset can be used to train and evaluate multi-person reconstruction algorithms.

[[THuman 2.0 Dataset]](https://github.com/ytrock/THuman2.0-Dataset) Containing 500 high-quality human scans captured by a dense DLSR rig, with SMPL annotations. 



## Citation
If you use this dataset for your research, please consider citing:
```
@InProceedings{zheng2022structured,
title={Structured Local Radiance Fields for Human Avatar Modeling},
author={Zheng, Zerong and Huang, Han and Yu, Tao and Zhang, Hongwen and Guo, Yandong and Liu, Yebin},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2022},
pages = {}
}
```

## Contact
- Zerong Zheng [(ytrock@126.com)](mailto:zrzheng1995@foxmail.com)
- Yebin Liu [(liuyebin@mail.tsinghua.edu.cn)](mailto:liuyebin@mail.tsinghua.edu.cn)