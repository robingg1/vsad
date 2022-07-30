# Deep Video Anomaly Detection in Surveillance Videos : Pytorch 

This repository is set up for the implementation of contrastive loss on the UCF-Crimes dataset.



## Datasets

Download following data [link](https://drive.google.com/file/d/18nlV4YjPM93o-SdnPQrvauMN_v-oizmZ/view?usp=sharing) and unzip in the proper directory. The path to the dataset should look as follows.

(../UCF_and_Shanghai/UCF-Crime/all_rgbs)

The repository is based on the feature extraction process as described in this paper. [link](https://github.com/junha-kim/Learning-to-Adapt-to-Unseen-Abnormal-Activities)


The dataset contains the following files :
* ShanghaiTech - Contains features extracted for the ShanghaiTech Dataset. (We do not consider this in the current implementation)
* UCF-Crime - Contains features extracted for the UCF-Crime Dataset. (Our current implementation is based on this)
    * all_flows - Contains the optical flow feature vectors.
    * all_rgbs - Contains the extracted features vectors from I3D model.
    * splits - Splits to consider for cross validation testing. (Not used currently)
    * GT_anomaly.pkl: Temporal annotations for all videos.
    * exclusion.pkl: We find some of duplicate videos (e.g. same videos but different video name)
    * frames.pkl: Number of frames for all videos


* Directory tree
 ```
    UCF_and_Shanghai/
        UCF-Crime/ 
            ../all_rgbs
                ../~.npy
            ../all_flows
                ../~.npy
        
```

## Results of various methods on the UCF-Crime dataset

| Method        | Feature       | AUC      |  Train Backbone |
| ------------- | ------------- | -------- | ----------------|
| [Sultani et al](https://arxiv.org/pdf/1801.04264.pdf)| C3D RGB        | 75.41  | No |
| [Sultani et al](https://arxiv.org/pdf/1801.04264.pdf) | I3D RGB        | 77.92  | No |
| [Zhang et al](https://ieeexplore.ieee.org/document/8803657)| C3D RGB        | 78.66  |
| [Motion-Aware](https://arxiv.org/pdf/1907.10211.pdf) | PWC Flow        | 79.00  |
| [GCN-Anomaly](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhong_Graph_Convolutional_Label_Noise_Cleaner_Train_a_Plug-And-Play_Action_Classifier_CVPR_2019_paper.pdf)| C3D RGB        | 81.08  |
| [Weakly Supervised GCN-Anomaly](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhong_Graph_Convolutional_Label_Noise_Cleaner_Train_a_Plug-And-Play_Action_Classifier_CVPR_2019_paper.pdf) | TSN Flow        | 78.08  |
| [GCN-Anomaly](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhong_Graph_Convolutional_Label_Noise_Cleaner_Train_a_Plug-And-Play_Action_Classifier_CVPR_2019_paper.pdf)| TSN RGB        | 82.12  |
| [MIST](https://openaccess.thecvf.com/content/CVPR2021/papers/Feng_MIST_Multiple_Instance_Self-Training_Framework_for_Video_Anomaly_Detection_CVPR_2021_paper.pdf) | C3D RGB        | 81.40| Yes |
| [MIST](https://openaccess.thecvf.com/content/CVPR2021/papers/Feng_MIST_Multiple_Instance_Self-Training_Framework_for_Video_Anomaly_Detection_CVPR_2021_paper.pdf) | I3D RGB        | 82.30| Yes |
| [Wu et al](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750324.pdf) | I3D RGB        | 82.44  |
| [RTFM](https://arxiv.org/pdf/2101.10030.pdf) | C3D RGB        | 83.28  | No |
| [RTFM](https://arxiv.org/pdf/2101.10030.pdf) | I3D RGB        | 84.30  | No |


## train-test script
```
python main.py                 #This will save a pseudolabel generator head based on the MIL Loss
python contrastivemain.py      #This will train the constrastive model using the constrastive loss function
python contrastivelineval.py   #This will evaluate the contrastive model based on the contrastive loss function
```

## For evaluating MIL method 
```
python mil.py  #This will train and evaluate the MIL model
```

## Result of the current methodology
| Method   | feature   |  AUC    |
| -------- | --------- | ------- |
| MIL model (Exact implementation of Sultani et al.) |  I3D RGB | 0.8014 |
| MIL + Constrastive loss function | I3D RGB | 0.8097 |
| MIL + Constrastive loss + attention based temporal encoder | I3D RGB | 0.8132 |
