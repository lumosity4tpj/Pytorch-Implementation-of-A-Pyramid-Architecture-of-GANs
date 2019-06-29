# Pytorch Implementation of A Pyramid Architecture of GANs

------

This is a pytorch implementation for reproducing PAGAN results in the paper [Learning Face Age Progression: A Pyramid Architecture of GANs](https://arxiv.org/pdf/1711.10352v1.pdf)<br>

Please note that this is not the official code and The code may still have errors for the results did not reach the original results.<br>

<br>

### Requirements

------

- Pytorch 1.0
- Python 3.6
- Visdom 0.1.8
- Pillow 6.0

<br>

### Dataset

------

- CACD
- FGnet

<br>

### Pretrained Models

------

You can download pretrained vgg-face models from (http://www.robots.ox.ac.uk/~albanie/pytorch-models.html) and refer to this paper (https://arxiv.org/ftp/arxiv/papers/1709/1709.01664.pdf) to train age estimation networks, then move the two models to `./model_vgg`.

It will require about 1.1 GB of disk space.

<br>

### Running Models

------

you can use the script `train.py`.

(Please note that modifying the path in the `train.py` and splitting data refer to `make_label.py` to the path `data_train/young`, `data_train/elder1` ...... )

<br>

### Results

------

