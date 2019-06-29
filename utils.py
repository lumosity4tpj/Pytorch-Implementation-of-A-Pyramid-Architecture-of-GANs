import torch.nn as nn
import torch
import numpy as np 
import random

def pixel_loss(fake_img,input_img):
    criterion = nn.MSELoss()
    loss = criterion(fake_img,input_img)
    return loss

def identity_loss(id_fake_img,id_input_img):
    criterion = nn.MSELoss()
    loss = criterion(id_fake_img,id_input_img)
    return loss

def GAN_G_loss(age_fake_img_logits,real_label):
    criterion = nn.MSELoss()
    loss = criterion(age_fake_img_logits,real_label)
    return loss 

def young_GAN_D_loss(age_fake_img_logits,age_input_img_logits,real_label,fake_label):
    criterion = nn.MSELoss()
    # # real
    # loss_real = criterion(age_input_img_logits,real_label)
    # fake
    loss_fake_g = criterion(age_fake_img_logits,fake_label)
    loss_fake_input = criterion(age_input_img_logits,fake_label)
    loss_fake = (loss_fake_g + loss_fake_input) * 0.5
    # loss = (loss_fake + loss_real) * 0.5
    return loss_fake,loss_fake_g * 0.5,loss_fake_input * 0.5

def elder_GAN_D_loss(age_input_img_logits,real_label):
    criterion = nn.MSELoss()
    # real
    loss_real = criterion(age_input_img_logits,real_label)
    loss_real = loss_real * 0.5
    return loss_real

def setup_seed(seed):
    torch.manual_seed(seed)#cpu
    torch.cuda.manual_seed(seed)#gpu
    np.random.seed(seed)#numpy
    random.seed(seed)#random and transforms
    torch.backends.cudnn.deterministic = True#cudnn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data,0.0,0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0)
    # elif classname.find('InstanceNorm') != -1:
    #     nn.init.normal_(m.weight.data,1.0,0.02)
    #     nn.init.constant_(m.bias.data,0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0,0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)   
