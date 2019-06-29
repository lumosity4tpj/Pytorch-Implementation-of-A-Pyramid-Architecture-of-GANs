# Pytorch Implementation of A Pyramid Architecture of GANs

------

This is a pytorch implementation for reproducing PAGAN results in the paper [Learning Face Age Progression: A Pyramid Architecture of GANs](https://arxiv.org/pdf/1711.10352v1.pdf).

**Please note that this is not the official code and The code may still have errors for the results did not reach the original results.**:weary:

### Requirements

------

- Pytorch 1.0
- Python 3.6
- Visdom 0.1.8
- Pillow 6.0

### Dataset

------

- CACD
- FGnet

**Please pay attention to** splitting CACD_dataset to train_dataset & val_dataset. and after `make_label.py` , move dataset to the path like`data_train/young(or elder1,elder2,elder3,val,test)`.

### Pretrained Models

------

You can download pretrained vgg-face models from (http://www.robots.ox.ac.uk/~albanie/pytorch-models.html) and refer to this paper (https://arxiv.org/ftp/arxiv/papers/1709/1709.01664.pdf) to train age estimation networks, then move the two models to `./model_vgg`.

It will require about 1.1 GB of disk space.

### Running Models

------

you can run the shell script `train.sh` and `test.sh`.

**Please note that modifying the path in the `CONFIG`  when different age cluster.** 

### Results

------

Here are some visualization results. And age estimation & face verification results by using [face++ API](https://www.faceplusplus.com.cn/).

- train:

  - age_cluster_1:
    - original: ![](https://github.com/lumosity4tpj/PAGAN/blob/master/result_imgs/train_1_input.png)
    - generate: ![](https://github.com/lumosity4tpj/PAGAN/blob/master/result_imgs/train_1_fake.png)
  - age_cluster_2:
    - original: ![](https://github.com/lumosity4tpj/PAGAN/blob/master/result_imgs/train_2_input.png)
    - generate: ![](https://github.com/lumosity4tpj/PAGAN/blob/master/result_imgs/train_2_fake.png)
  - age_cluster_3:
    - original: ![](https://github.com/lumosity4tpj/PAGAN/blob/master/result_imgs/train_3_input.png)
    - generate: ![](https://github.com/lumosity4tpj/PAGAN/blob/master/result_imgs/train_3_fake.png)

- val(CACD):

  - val_age: 14

    ![](https://github.com/lumosity4tpj/PAGAN/blob/master/result_imgs/val_14_female.jpg)

    ![](https://github.com/lumosity4tpj/PAGAN/blob/master/result_imgs/val_14_male.jpg)

  - val_age: 22

    ![](https://github.com/lumosity4tpj/PAGAN/blob/master/result_imgs/val_22_female.jpg)

    ![](https://github.com/lumosity4tpj/PAGAN/blob/master/result_imgs/val_22_male.png)

  - val_age: 30

    ![](https://github.com/lumosity4tpj/PAGAN/blob/master/result_imgs/val_30_female.jpg)

    ![](https://github.com/lumosity4tpj/PAGAN/blob/master/result_imgs/val_30_male.jpg)

  - age estimation & face verification results:

    |                                                   | age cluster1 | age cluster2 | age cluster3 |
    | ------------------------------------------------- | ------------ | ------------ | ------------ |
    | average estimate age                              | 42.1         | 50.7         | 61.7         |
    | age accuracy(if estimate age in the age cluster ) | 33.1%        | 33.0%        | 90.2%        |
    | average veriﬁcation conﬁdence(with age cluster 0) |              | 86.6         | 79.9         |
    | veriﬁcation rate(FAR = 1e-5)                      |              | 97.8%        | 84.0%        |

- test(FGnet):

  - test_age: 14

    ![](https://github.com/lumosity4tpj/PAGAN/blob/master/result_imgs/test_14_female.jpg)

    ![](https://github.com/lumosity4tpj/PAGAN/blob/master/result_imgs/test_14_male.jpg)

  - test_age: 22

    ![](https://github.com/lumosity4tpj/PAGAN/blob/master/result_imgs/test_22_female.jpg)

    ![](https://github.com/lumosity4tpj/PAGAN/blob/master/result_imgs/test_22_male.jpg)

  - test_age: 30

    ![](https://github.com/lumosity4tpj/PAGAN/blob/master/result_imgs/test_30_female.jpg)

    ![](https://github.com/lumosity4tpj/PAGAN/blob/master/result_imgs/test_30_male.jpg)

  - age estimation & face verification results:

    |                                                   | age cluster1 | age cluster2 | age cluster3 |
    | :-----------------------------------------------: | :----------: | :----------: | :----------: |
    |               average estimate age                |     37.6     |     48.4     |     51.1     |
    | age accuracy(if estimate age in the age cluster ) |    44.3%     |    42.7%     |    56.9%     |
    | average veriﬁcation conﬁdence(with age cluster 0) |     92.3     |     87.7     |     87.7     |
    |           veriﬁcation rate(FAR = 1e-5)            |    99.7%     |    98.1%     |    97.2%     |

    

