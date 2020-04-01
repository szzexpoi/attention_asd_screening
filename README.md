# Attention-based Autism Spectrum Disorder Screening
This code implements the image-viewing based ASD screening model proposed in the paper "Attention-based Autism Spectrum Disorder Screening with Privileged Modality". It is used to reproduce the results on the [Saliency4ASD](https://saliency4asd.ls2n.fr/datasets/) dataset. The high-level architecture of the proposed model is visualized below:

![teaser](asset/model.png?raw=true)

### Reference
If you use our code or data, please cite our paper:
```
@InProceedings{Chen_2019_ICCV,
author = {Chen, Shi and Zhao, Qi},
title = {Attention-Based Autism Spectrum Disorder Screening With Privileged Modality},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}
```

### Requirements
1. Requirements for Pytorch. We use Pytorch 1.2.0 in our experiments.
2. Requirements for Tensorflow. We only use the tensorboard for visualization.
3. Python 3.6+

### Data Processing
The code performs leave-one-subject-out evaluation on the training splits of [Saliency4ASD](https://saliency4asd.ls2n.fr/datasets/) dataset. Please download the dataset accordingly and unzip it to folder `saliency4asd`.

### Experiments
Running the experiments with our code is straightforward, as the default parameters have already been set following the paper, simply call:
```
python main.py --checkpoint_path $CHECKPOINT_DIR
```
The tensorboard visualization (stored in `$CHECKPOINT_DIR`) provides the prediction accuracy (predicted confidence on the correct labels) on different hold-out subjects during the leave-one-subject-out evaluation.
