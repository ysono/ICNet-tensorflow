# ICNet_tensorflow
## Introduction
  This is an implementation of ICNet in TensorFlow for semantic segmentation on the [cityscapes](https://www.cityscapes-dataset.com/) dataset. We first convert weight from [Original Code](https://github.com/hszhao/ICNet) by using [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow) framework.  
  
 ![](https://github.com/hellochick/ICNet-tensorflow/blob/master/utils/icnet.png)
 
## Update
#### 2018/2/16:
1. Fix bug in 'ICNet_BN' model, which related to issue [#32](https://github.com/hellochick/ICNet-tensorflow/issues/32) and [#41](https://github.com/hellochick/ICNet-tensorflow/issues/41).
2. Update new pre-trained weight of ADE20k dataset, which reached **32.0% mIoU**.

#### 2018/1/30:
1. `Support evaluation and inference code for ADE20k dataset`, and the model reached **30.2% mIoU** after 200k steps of training.  
> Note: I **trained on the non-pruned model**, and I **haven't done model pruning and merge bn parameters.** However, it still maintain `Real-time` property.
2. Support two version of models: **Non-pruned and pruned version**. By adding flag `--filter-scale=1 or 2` to select different configurations. I recommend set `--filter-scale=2` during training phase, and this doubles the number of filters (if you have any doubt, see the implementation part which described in the paper first).

#### 2018/1/27:
1. Improve evaluation results by changing `interp` operation and add `zero padding` in front of max pooling layer. Such modification improve the mIoU to **67.35%** ( much closer to original work ).  [Pull request #35](https://github.com/hellochick/ICNet-tensorflow/pull/35)

#### 2017/11/15:
1. Support `training phase`, you can train on your own dataset. Please read the guide below.

#### 2017/11/13:
1. Add `bnnomerge model` which reparing for training phase. Choose different model using flag `--model=train, train_bn, trainval, trainval_bn` (Upload model in google drive).
2. Change `tf.nn.batch_normalization` to `tf.layers.batch_normalization`.

#### 2017/11/07:
`Support every image size larger than 128x256` by changing the avg pooling ksize and strides in the pyramid module. If input image size cannot divided by 32, it will be padded in to mutiple of 32.


## Install
Get restore checkpoint from [Google Drive](https://drive.google.com/drive/folders/1pBN07IW_zxEVlL2q9ColGs6QkUNkplsi?usp=sharing) and put into `model` directory.

## Inference
To get result on your own images, use the following command:

### Cityscapes example
```
python inference.py --img-path=./input/outdoor_1.png --dataset=cityscapes --filter-scale=1 
```
### ADE20k example
```
python inference.py --img-path=./input/indoor_1.png --dataset=ade20k --filter-scale=2
```

List of Args:
```
--model=train       - To select train_30k model (Default)
--model=trainval    - To select trainval_90k model
--model=train_bn    - To select train_30k_bn model
--model=trainval_bn - To select trainval_90k_bn model
--model=others      - To select your own checkpoint

--dataset=cityscapes - To select inference on cityscapes dataset
--dataset=ade20k     - To select inference on ade20k dataset

--filter-scale      - 1 for pruned model, while 2 for non-pruned model. (if you load pre-trained model, always set to 1. 
                      However, if you want to try pre-trained model on ade20k, set this parameter to 2)
```
### Inference time
* **Including time of loading images**: ~0.04s
* **Excluding time of loading images (Same as described in paper)**: ~0.03s

## Evaluation
### Cityscapes
Perform in single-scaled model on the cityscapes validation dataset. (We have sucessfully re-produced the performance same to caffe framework!)

| Model | Accuracy |  Missing accuracy |
|:-----------:|:----------:|:---------:|
| train_30k   | **67.67/67.7** | **0.03%** |
| trainval_90k| **81.06%**    | None |

To get evaluation result, you need to download Cityscape dataset from [Official website](https://www.cityscapes-dataset.com/) first. Then change `cityscapes_param` to your dataset path in `evaluate.py`:
```
# line 29
'data_dir': '/PATH/TO/YOUR/CITYSCAPES_DATASET'
```

Then run the following command: 
```
python evaluate.py --dataset=cityscapes --filter-scale=1 --model=trainval
```
List of Args:
```
--model=train    - To select train_30k model (Default)
--model=trainval - To select trainval_90k model
--measure-time   - Calculate inference time (e.q subtract preprocessing time)
```

### ADE20k
Reach **30.2% mIoU** on ADE20k validation set.
```
python evaluate.py --dataset=cityscapes --filter-scale=2 --model=others
```
> Note: to use model provided by us, set `filter-scale` to 2

## Image Result
### Cityscapes
Input image                |  Output image
:-------------------------:|:-------------------------:
![](https://github.com/hellochick/ICNet_tensorflow/blob/master/input/outdoor_1.png)  |  ![](https://github.com/hellochick/ICNet_tensorflow/blob/master/output/outdoor_1.png)

### ADE20k
Input image                |  Output image
:-------------------------:|:-------------------------:
![](https://github.com/hellochick/ICNet_tensorflow/blob/master/input/indoor_2.jpg)  |  ![](https://github.com/hellochick/ICNet_tensorflow/blob/master/output/indoor_2.jpg)
![](https://github.com/hellochick/ICNet_tensorflow/blob/master/input/outdoor_2.png)  |  ![](https://github.com/hellochick/ICNet_tensorflow/blob/master/output/outdoor_2.png)

## Training on your own dataset
> Note: This implementation is different from the details descibed in ICNet paper, since I did not re-produce model compression part. Instead, we train on the half kernel directly.

### Step by Step
**1. Change the `DATA_LIST_PATH`** in line 22, make sure the list contains the absolute path of your data files, in `list.txt`:
```
/ABSOLUTE/PATH/TO/image /ABSOLUTE/PATH/TO/label
```
**2. Set Hyperparameters** (line 21-35) in `train.py`
```
BATCH_SIZE = 48
IGNORE_LABEL = 0
INPUT_SIZE = '480,480'
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
NUM_CLASSES = 27
NUM_STEPS = 60001
POWER = 0.9
RANDOM_SEED = 1234
WEIGHT_DECAY = 0.0001
```
Also **set the loss function weight** (line 38-40) descibed in the paper:
```
# Loss Function = LAMBDA1 * sub4_loss + LAMBDA2 * sub24_loss + LAMBDA3 * sub124_loss
LAMBDA1 = 0.4
LAMBDA2 = 0.6
LAMBDA3 = 1.0

```
**3.** Run following command and **decide whether to update mean/var or train beta/gamma variable**.
```
python train.py --update-mean-var --train-beta-gamma
```
After training the dataset, you can run following command to get the result:  
```
python inference.py --img-path=YOUR_OWN_IMAGE --model=others
```
### Result ( inference with my own data )

Input                      |  Output
:-------------------------:|:-------------------------:
![](https://github.com/hellochick/ICNet_tensorflow/blob/master/input/indoor_1.jpg)  |  ![](https://github.com/hellochick/ICNet-tensorflow/blob/master/output/indoor_1.jpg)

## Citation
    @article{zhao2017icnet,
      author = {Hengshuang Zhao and
                Xiaojuan Qi and
                Xiaoyong Shen and
                Jianping Shi and
                Jiaya Jia},
      title = {ICNet for Real-Time Semantic Segmentation on High-Resolution Images},
      journal={arXiv preprint arXiv:1704.08545},
      year = {2017}
    }
Scene Parsing through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso and A. Torralba. Computer Vision and Pattern Recognition (CVPR), 2017. (http://people.csail.mit.edu/bzhou/publication/scene-parse-camera-ready.pdf)

    @inproceedings{zhou2017scene,
        title={Scene Parsing through ADE20K Dataset},
        author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
        year={2017}
    }
    
Semantic Understanding of Scenes through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso and A. Torralba. arXiv:1608.05442. (https://arxiv.org/pdf/1608.05442.pdf)

    @article{zhou2016semantic,
      title={Semantic understanding of scenes through the ade20k dataset},
      author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
      journal={arXiv preprint arXiv:1608.05442},
      year={2016}
    }
