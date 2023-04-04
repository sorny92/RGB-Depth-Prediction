# RGB depth estimation

### Datasets

For the task of depth estimation there's multiple sources of data.  
As this case describes, a viable dataset is one with RGB images and depth maps as outputs.
This type or datasets are not easy to produce in a reliable way as there's no high resolution depth sensor currently.
Nowadays, this is done through several systems: through disparity maps in stereo images, with a lidar, or a pattern
matching system.
Each one of them has different advantages and disadvantages, which I'm not going to study here, but they should be taken
into account.

For this task we can look for some datasets in one of the most popular resources pages:

* [Papers with code](https://paperswithcode.com/datasets?task=depth-estimation&mod=stereo):
  Has multiple datasets which we can split in the next categories:
    * **Stereo**:  
      This data could be useful for pretaining as a disparity map could be generated and pretrain the model trying to
      predict the disparity map. Then the model would be fine-tuned in depth data.
        * [HRWSI](https://kexianhust.github.io/Structure-Guided-Ranking-Loss/): Can be downloaded
        * [ETH3D](https://www.eth3d.net/overview): Can be downloaded
        * [UASOL](https://osf.io/64532/): Can be downloaded
        * [Holopix50k](https://github.com/leiainc/holopix50k): Collection in the wild and it can be downloaded.
        * [WSVD](https://sites.google.com/view/wsvd/home): More than 500 videos from youtube recorded in stereoscopic.
          Same as the others
    * **RGB-D**:
        * Real data:
            * [DIODE](https://diode-dataset.org/): Promising dataset as it contains data from indoor and
              outdoor. 8k images indoor, 17k images outdoor. It can also be downloaded.
            * [NYUv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html): 1.4k images dense, 400k sparse.
            * [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download): Data extracted from a Microsoft
              Kinect. Can be downloaded
            * [SUN3D](https://sun3d.cs.princeton.edu/): Data extracted from a Microsoft Kinect. Can be downloaded
            * [ScanNet](https://github.com/ScanNet/ScanNet): It would be useful as it also has segmentation mask and
              bounding boxes. But it can't be downloaded directly without asking for permission.
        * Synthetic data:
            * [Hypersim](https://github.com/apple/ml-hypersim): Useful dataset for pretraining as it contains synthetic
              data. It can also be downloaded.
            * [EDEN](https://lhoangan.github.io/eden/): Synthetic dataset but it's specific for gardens it might not be
              great as it could overfit. It's nice that it has segmentation mask too.
            * [Virtual KITTI](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-1/):
              Dataset with
            * [SUNCG](https://sscnet.cs.princeton.edu/): Interesting dataset for pretraining as it contains synthetic
              data
              that is dense. Can't be downloaded due to
              a [legal problem](https://futurism.com/tech-suing-facebook-princeton-data).
            * [SuperCaustics](https://github.com/MMehdiMousavi/SuperCaustics): synthetic data from UE. It could be
              useful to generate data on the fly, but it doesn't have a dataset with depth.

    * **Lidar/Pointcloud**:
        * [ETH3D](https://www.eth3d.net/overview):

* [Huggingface](https://huggingface.co/datasets?task_categories=task_categories:depth-estimation&sort=downloads):
  There's only one dataset available for this task in Huggingface which is nyu_depth_v2.

## Models

This is a task which is considered as a image to image problem. The input is an image and the output is another image.
For this problem the output is a depth map, so the values are continuous therefor it's a regression per pixel.

In the file [model.py](model.py) it can be seen the model used for this task.
In this case a model based on DeepLabv3 as it's mostly used for image segmentation, the last layer can be changed to
just one channel without activation, so it can solve regression problems.

### Data preparation

On the file [loader.py](data%2Floader.py) it can be seen the pipeline for data loading.
A generator is created that will feed paths to images so tensorflow can automatically prefetch the images and process
them to have them ready for the GPU when it needs them.

The scaling used is the one used for imagenet as the backbone of the model is pretrained in it.

## Method

For this project I will use [DIODE](https://diode-dataset.org/) as it's much bigger than NYUv2 with more variety of
scenes.

For the training it's used the whole train set split in 90% of the scenes for training and the other 10% for validation.
A set of data augmentation operations are used to train that can be seen in the
file [data_augmentation.py](data%2Fdata_augmentation.py).

The training pipeline can be found in the file [training.py](training.py).
For this task as the literature similar to this project does, Adam with default learning rate is used.
The learning rate decay is also used during training to reduce when the learning achieves a plateau.
The loss used will be detailed in the next point.

### Loss function

The loss function used is the one used in [here](http://cs231n.stanford.edu/reports/2022/pdfs/58.pdf) and it's
implemented in [training.py](training.py).

It consists in three parts:

* L1 loss over the whole image
* L1 loss over the edges of the image
* structural similarity of the image.
  The loss is known in this task to help converge faster.
  ![MAE.png](assets%2FMAE.png)
  In the previous image you can see three training with the same data but with one component of the loss, with two and
  the three of them.

### Metrics

Metrics used are the standard for the task:

* Mean absolute error
* Root mean squared error
* And thresold

## Results

Over validation set (including indoor and outdoor) of DIODE the model [model-04-182311](models%2Fmodel-04-182311)
achieves the next results:

* mean_absolute_error: 0.0215
* root_mean_squared_error: 0.0403
* delta1: 0.2337
* delta2: 0.4377
* delta3: 0.5808

The current model achieves realtime processing speed at 512x512 in a RTX3060 without quantization or layer fusion.
In a modern 12 threads system CPU can run at 2FPS.

## Future work

* The model need to be trained for longer as it never got to achieve a plateau.
* The depth maps in the DIODE dataset have a lot of null values where there's no depth data, specially in the outdoor
  set. This needs to be addressed as the model has a bias to set everything mostly black and it's good enough in many
  cases.
* In works as NYUv2 they apply image [inpainting](https://www.cs.huji.ac.il/~yweiss/Colorization/) to fill the gaps in
  the depth map, but there's definitely more options
  as [this](https://www.researchgate.net/figure/Inpainting-Classification_fig3_306310171)
* The model used is based on a backbone that is not the most efficient, Resnet50. A model based in something more
  modern, for example EfficientNet would definitely bring better results in less time.
* Apply quantization to the model to decrease latency.
* Better pretraining making use of the extensive access to synthetic data in this domain.