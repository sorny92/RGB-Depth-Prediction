# RGB depth estimation

## Related work

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
              useful to generate data on the fly but it doesn't have a dataset with depth.

    * **Lidar/Pointcloud**:
        * [ETH3D](https://www.eth3d.net/overview):

* [Huggingface](https://huggingface.co/datasets?task_categories=task_categories:depth-estimation&sort=downloads):
  There's only one dataset available for this task in Huggingface which is nyu_depth_v2.

### Models

## Method
For this project I will use [DIODE](https://diode-dataset.org/) as it's much bigger than NYUv2 with more variety of
scenes so hopefully a pretrained model in NYUv2 can be used.

## Results

## Future work
