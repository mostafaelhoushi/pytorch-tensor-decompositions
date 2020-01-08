# Tensor Decompositions

This GitHub repo is an extension to https://github.com/jacobgil/pytorch-tensor-decompositions.
It provided an implementation to this paper:
```
@misc{elhoushi2019accelerating,
    title={Accelerating Training using Tensor Decomposition},
    author={Mostafa Elhoushi and Ye Henry Tian and Zihao Chen and Farhan Shafiq and Joey Yiwei Li},
    year={2019},
    eprint={1909.05675},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

as well as:
```
@misc{kim2015compression,
    title={Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications},
    author={Yong-Deok Kim and Eunhyeok Park and Sungjoo Yoo and Taelim Choi and Lu Yang and Dongjun Shin},
    year={2015},
    eprint={1511.06530},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

It provides an implementation of Tucker and CP decomposition of convolutional layers.

It depends on [TensorLy](https://github.com/tensorly/tensorly) for performing tensor decompositions.

# Usage

The repo supports training/inference on both CIFAR10 and Imagenet datasets.

The following commands are for Imagenet dataset, if you want to perform the same functionality on CIFAR10, just replace `imagenet.py` with `cifar10.py`.

- Training from scratch
```
python imagenet.py --arch resnet50
```

this will train ResNet50 from scratch, saving the model. train log, as well as a checkpoint every 10 epochs in `./models/imagenet/resnet50/no_decompose`

- Decompose a pre-trained model using the approach of [Kim et al.](https://arxiv.org/abs/1511.06530):
```
python imagenet.py --arch resnet50 --pretrained True --decompose --epochs 15 --lr 0.001 --lr-step-size 5
```

- Decompose a model that was trained till the 5th epoch:
```
python imagenet.py --arch resnet50 --weights ./models/imagenet/resnet50/no_decompose/checkpoint_10.pth.tar --decompose --start-epoch 11
```
Here, we use the checkpoint of trainnig from scratch at the 10th epoch

# References

- CP Decomposition for convolutional layers is described here: https://arxiv.org/abs/1412.6553
- Tucker Decomposition for convolutional layers is described here: https://arxiv.org/abs/1511.06530
- VBMF for rank selection is described here: http://www.jmlr.org/papers/volume14/nakajima13a/nakajima13a.pdf
- VBMF code was taken from here: https://github.com/CasvandenBogaard/VBMF
- Tensorly: https://github.com/tensorly/tensorly
