# GNN - Colorizing black and white portrait photographs
The process involves employing generative networks to colorize black and white portrait images, specifically utilizing two methods: VAE (Variational Autoencoders) and cINN (conditional Invertible Neural Networks).
## about dataset
### 1. celebA[^1]
The CelebA dataset, short for Celebrities Attributes dataset, is a large-scale face attributes dataset designed for research in the fields of machine learning and computer vision, particularly for the development and evaluation of algorithms involving facial recognition, facial attribute recognition, facial editing, and many others.

#### dataset preprocess
a. cVAE:Divide the data set into training set and test set according to 7:3  
b. cINN:Divide the data set into training set, validation set and test set according to 9:0.5:0.5

### 2. our own dataset
Here we have expanded the task. The data set with only face portraits is not very difficult, so we regenerated a data set including all human bodies and complex backgrounds.
#### dataset source
a. OCHuman(Occluded Human) Dataset[^2]  
b. MPII Human Pose [^3]  
c. COCO2017 keypoints[^4]  
#### dataset preprocess
a. We first delete all images with only brightness channel  
b. Then select a picture with a resolution greater than 256*256   
c. Use MMDection[^5] pre-trained solov2[^6] to perform instance segmentation, and select human body instances whose area exceeds 1/5 of the image area to join our data set.
d. After screening, we obtained about 50k data sets and divided them into training set, validation set and test set according to 9:0.5:0.5.

### 3. Model
#### I. cINN
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

#### II. cVAE
##### a. cVAE-Naive
We initially employed the naive cVAE structure. We utilized VGG19[^10] as a feature extractor to extract features from grayscale images, and concatenated it with the sampled results from the encoder as input to the decoder.

pic

During training, our loss function consists of the following two parts:
- Reconstruction Loss: This component measures how well the VAE can reconstruct the input data. It encourages the decoder to generate outputs that are close to the original inputs.
  $$L_{MSE}^{NaiveVAE} = \frac{1}{N} \sum_{i=1}^{N} (x^c_i - \hat{x}^c_i)^2 $$
  Where $N$ is the number of samples, $x^c_i$ represents the pixel values of the original input color image, and $\hat{x}^c_i$ represents the pixel values of the color image generated by the decoder.
- Regularization Loss: This term ensures that the latent space is smooth and continuous, making it easier to sample meaningful points. To achieve this goal, we need to minimize the KL divergence between the encoder's sampling distribution $p(z|x)$ and the standard normal distribution.
  $$L_{KL}^{NaiveVAE} = -\frac{1}{2} \sum_{j=1}^{J} \left(1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$
  Where $J$ is the dimensionality of the latent space, $\mu_j$ and $\sigma_j$ respectively represent the mean and standard deviation of the learned latent distribution.

So our total loss function is:
$$L^{NaiveVAE}=L_{MSE}^{NaiveVAE}+w*L_{KL}^{NaiveVAE}$$
Where $w$ is a hyperparameter. Through testing, if its value is too large, it fails to generate images; if its value is too small, the diversity of generated images is very weak. In this experiment, its value is set to 0.00003.

After training, we obtained the following results. 

![NaiveVAEResult](https://github.com/YongWu-cs/GNN/blob/main/pic_source/NaiveVAE/NaiveVAEResult.png)

It can be observed that the generated images exhibit blurriness. This appears to be a common issue with naive cVAE. Furthermore, the variability of the generated images is weak.
##### a. cVAE-UNet
In response to the blurriness issue observed in the Naive cVAE, we proposed a new cVAE architecture based on the UNet structure. UNet utilizes skip connections, which better preserve the details of images compared to traditional convolutional layers, enabling the decoder to better recover fine details from the input images. Additionally, we redesigned the feature extractor based on VGG19.

We first trained the feature extractor.

pic

Initially, our loss function only focused on the reconstruction of grayscale images. However, upon analyzing the results, we found that although it performed well on grayscale images, the generated color images tended to cluster around a single hue, which was not conducive to the subsequent training of our cVAE. 

pic

Therefore, we improved the loss function to not only focus on the reconstruction of grayscale images but also on the reconstruction of color images.
$$L_{MSE}^{UNet} = \frac{1}{N} \sum_{i=1}^{N} (x^c_i - \hat{x}^c_i)^2 + \frac{1}{N} \sum_{i=1}^{N} (x^g_i - \hat{x}^g_i)^2 $$

After training, we obtained the following results.

pic

From this, it can be seen that we not only achieved the reconstruction of grayscale images but also ensured that the generated color images have a consistent color distribution with the original color images.

Due to training the feature extractor based on UNet, which essentially accomplishes the transformation from grayscale images to color images, in order to prevent subsequent networks from failing to learn from the results sampled from the cVAE encoder, we changed the fusion method of features from concatenation to addition. Furthermore, to avoid introducing unnecessary linear layers (whose parameters might become zero, which could also prevent the network from learning from the results sampled from the cVAE encoder and still achieve good performance), we set the dimensionality of the sampled results from the encoder to be the same as the dimensionality of the output tensor of the feature extractor.

pic

Like the loss function of the naive cVAE, the loss function of the UNet-based cVAE also focuses on both the reconstruction loss and the regularization loss.
- Reconstruction Loss:
  $$L_{MSE}^{UNetVAE} = \frac{1}{N} \sum_{i=1}^{N} (x^c_i - \hat{x}^c_i)^2 $$
  Where $N$ is the number of samples, $x^c_i$ represents the pixel values of the original input color image, and $\hat{x}^c_i$ represents the pixel values of the color image generated by the decoder.
- Regularization Loss: 
  $$L_{KL}^{UNetVAE} = -\frac{1}{2} \sum_{j=1}^{J} \left(1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$
  Where $J$ is the dimensionality of the latent space, $\mu_j$ and $\sigma_j$ respectively represent the mean and standard deviation of the learned latent distribution.
- total loss function:
  $$L^{UNetVAE}=L_{MSE}^{UNetVAE}+w*L_{KL}^{UNetVAE}$$
  In this experiment, the value of $w$ is set to 0.00003.
  
After training, we obtained the following results.

pic
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
[^10]: @article{simonyan2014very,
        title={Very deep convolutional networks for large-scale image recognition},
        author={Simonyan, Karen and Zisserman, Andrew},
        journal={arXiv preprint arXiv:1409.1556},
        year={2014}
      }
