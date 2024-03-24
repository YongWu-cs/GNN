# GNN - Colorizing black and white portrait photographs
The process involves employing generative networks to colorize black and white portrait images, specifically utilizing two methods: VAE (Variational Autoencoders) and cINN (conditional Invertible Neural Networks).
## about dataset
### 1. celebA[^1]
The CelebA dataset, short for Celebrities Attributes dataset, is a large-scale face attributes dataset designed for research in the fields of machine learning and computer vision, particularly for the development and evaluation of algorithms involving facial recognition, facial attribute recognition, facial editing, and many others.

#### dataset preprocess
a.vae:Divide the data set into training set and test set according to 7:3  
b.cINN:Divide the data set into training set, validation set and test set according to 9:0.5:0.5

### 2.our own dataset
Here we have expanded the task. The data set with only face portraits is not very difficult, so we regenerated a data set including all human bodies and complex backgrounds.
#### dataset source
a.OCHuman(Occluded Human) Dataset[^2]  
b.MPII Human Pose [^3]  
c.COCO2017 keypoints[^4]  
#### dataset preprocess
a.We first delete all images with only brightness channel  
b.Then select a picture with a resolution greater than 256*256   
c.Use MMDection[^5] pre-trained solov2[^6] to perform instance segmentation, and select human body instances whose area exceeds 1/5 of the image area to join our data set.
d.After screening, we obtained about 50k data sets and divided them into training set, validation set and test set according to 9:0.5:0.5.

### 3. Model
#### I.cINN
All code is based on FrEIA[^7] and pytorch.And opencv-python is used to implement joint filtering upsampling.

##### a. celebA architeture
In terms of the overall architecture, we chose the network architecture of GUIDED IMAGE GENERATION WITH CONDITIONAL INVERTIBLE NEURAL NETWORKS[^8].
![cinn_architeture](https://github.com/YongWu-cs/GNN/blob/main/pic_source/architecture.png)
In addition, for feature extraction of l channel, we also tried resnet_18 as a feature extractor.
##### b. our own dataset architeture
As we moved toward more complex data sets, our initial model became too crude, so we updated the model's architecture.   
![cinn_complex_architeture](https://github.com/YongWu-cs/GNN/blob/main/pic_source/complex_architecture.png)
And to address the problem of lack of semantic information, a new training pipeline was constructed with reference to the Instance-aware Image Colorization method[^9].  
![pipeline](https://github.com/YongWu-cs/GNN/blob/main/pic_source/pipeline.png)

#### II.VAE

## References
[^1]: @inproceedings{liu2015faceattributes,
      title = {Deep Learning Face Attributes in the Wild},
      author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
      booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
      month = {December},
      year = {2015} 
      }
[^2]: @misc{zhang2019pose2seg,
      title={Pose2Seg: Detection Free Human Instance Segmentation}, 
      author={Song-Hai Zhang and Ruilong Li and Xin Dong and Paul L. Rosin and Zixi Cai and Han Xi and Dingcheng Yang and Hao-Zhi Huang and Shi-Min Hu},
      year={2019},
      eprint={1803.10683},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
      }
[^3]: @inproceedings{andriluka14cvpr,
       author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt}
       title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
       booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
       year = {2014},
       month = {June}
        }
[^4]: T. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick, "Microsoft COCO: Common Objects in Context," in European Conference on Computer Vision (ECCV), 2014.
[^5]: @article{mmdetection,
      title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
      author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
                 Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
                 Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
                 Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
                 Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
                 and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
      journal= {arXiv preprint arXiv:1906.07155},
      year={2019}
    }
[^6]: @misc{wang2020solov2,
      title={SOLOv2: Dynamic and Fast Instance Segmentation}, 
      author={Xinlong Wang and Rufeng Zhang and Tao Kong and Lei Li and Chunhua Shen},
      year={2020},
      eprint={2003.10152},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
      }
[^7]:@software{freia,
        author = {Ardizzone, Lynton and Bungert, Till and Draxler, Felix and Köthe, Ullrich and Kruse, Jakob and Schmier, Robert and Sorrenson, Peter},
        title = {{Framework for Easily Invertible Architectures (FrEIA)}},
        year = {2018-2022},
        url = {https://github.com/vislearn/FrEIA}
      }
[^8]: @misc{ardizzone2019guided,
      title={Guided Image Generation with Conditional Invertible Neural Networks}, 
      author={Lynton Ardizzone and Carsten Lüth and Jakob Kruse and Carsten Rother and Ullrich Köthe},
      year={2019},
      eprint={1907.02392},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
      }
[^9]: @article{DBLP:journals/corr/abs-2005-10825,
        author       = {Jheng{-}Wei Su and
                        Hung{-}Kuo Chu and
                        Jia{-}Bin Huang},
        title        = {Instance-aware Image Colorization},
        journal      = {CoRR},
        volume       = {abs/2005.10825},
        year         = {2020},
        url          = {https://arxiv.org/abs/2005.10825},
        eprinttype    = {arXiv},
        eprint       = {2005.10825},
        timestamp    = {Thu, 06 Jul 2023 10:01:56 +0200},
        biburl       = {https://dblp.org/rec/journals/corr/abs-2005-10825.bib},
        bibsource    = {dblp computer science bibliography, https://dblp.org}
      }
