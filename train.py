import torch
import os
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models,transforms,datasets
import torchfile
import torch.optim as optim
import random
import argparse
import time
import PIL.Image as Image
import visdom

# from make_label import makeDir,moveFiles,encodeAge,moveFiles_test
from utils import pixel_loss,young_GAN_D_loss,elder_GAN_D_loss,GAN_G_loss,identity_loss,weights_init,setup_seed

#################################################### VGG module ##############################################
class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()
        # 3 x 224 x 224 -> 64 x 224 x 224 
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_2 = nn.ReLU(inplace=True)
        # 64 x 224 x 224 -> 64 x 112 x 112 
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        # 64 x 112 x 112 -> 128 x 112 x 112 
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_2 = nn.ReLU(inplace=True)
        # 128 x 112 x 112 -> 128 x 56 x 56 
        self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        # 128 x 56 x 56 -> 256 x 56 x 56 
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_3 = nn.ReLU(inplace=True)
        # 256 x 56 x 56 -> 256 x 28 x 28
        self.pool3 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        # 256 x 28 x 28 -> 512 x 28 x 28 
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_3 = nn.ReLU(inplace=True)
        # 512 x 28 x 28 -> 512 x 14 x 14
        self.pool4 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        # 512 x 14 x 14 -> 512 x 14 x 14 
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_3 = nn.ReLU(inplace=True)
        # 512 x 14 x 14 -> 512 x 7 x 7
        self.pool5 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.fc6 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout(p=0.5)
        # self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)

    def forward(self, x0):
        x1 = self.conv1_1(x0)
        x2 = self.relu1_1(x1)
        x3 = self.conv1_2(x2) #
        x4 = self.relu1_2(x3)
        x5 = self.pool1(x4)
        x6 = self.conv2_1(x5)
        x7 = self.relu2_1(x6)
        x8 = self.conv2_2(x7) #
        x9 = self.relu2_2(x8)
        x10 = self.pool2(x9)
        x11 = self.conv3_1(x10)
        x12 = self.relu3_1(x11)
        x13 = self.conv3_2(x12)
        x14 = self.relu3_2(x13)
        x15 = self.conv3_3(x14) #
        x16 = self.relu3_3(x15)
        x17 = self.pool3(x16)
        x18 = self.conv4_1(x17)
        x19 = self.relu4_1(x18)
        x20 = self.conv4_2(x19)
        x21 = self.relu4_2(x20)
        x22 = self.conv4_3(x21) #
        x23 = self.relu4_3(x22)
        x24 = self.pool4(x23)
        x25 = self.conv5_1(x24)
        x26 = self.relu5_1(x25)
        x27 = self.conv5_2(x26)
        x28 = self.relu5_2(x27)
        x29 = self.conv5_3(x28)
        x30 = self.relu5_3(x29)
        x31_preflatten = self.pool5(x30)
        x31 = x31_preflatten.view(x31_preflatten.size(0), -1)
        x32 = self.fc6(x31)
        x33 = self.relu6(x32)
        x34 = self.dropout6(x33)
        x35 = self.fc7(x34)
        x36 = self.relu7(x35)
        x37 = self.dropout7(x36)
        # x38 = self.fc8(x37)
        return x3,x8,x15,x22,x37

def vgg_identity(weights_path=None):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = VGG16()
    if weights_path:
        model_state_dict = model.state_dict()
        weights_state_dict = torch.load(weights_path,map_location=lambda storage,loc:storage)
        state_dict = {k:v for k,v in weights_state_dict.items() if k in model_state_dict.keys()}
        model.load_state_dict(state_dict)
        print('Load from: ',weights_path)

    for param in model.parameters():
        param.requires_grad = False
    
    if device.type == 'cuda' and opt.ngpu > 1:
        model = nn.DataParallel(model,device_ids=list(range(opt.ngpu)))

    if torch.cuda.is_available() and opt.ngpu > 0:
        model = model.to(device)

    return model

def vgg_age(weights_path=None):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = VGG16()
    if weights_path:
        model_state_dict = model.state_dict()
        weights_state_dict = torch.load(weights_path,map_location=lambda storage,loc:storage)
        state_dict = {k:v for k,v in weights_state_dict.items() if k in model_state_dict.keys()}
        model.load_state_dict(state_dict)
        print('Load from: ',weights_path)
    
    if device.type == 'cuda' and opt.ngpu > 1:
        model = nn.DataParallel(model,device_ids=list(range(opt.ngpu)))

    if torch.cuda.is_available() and opt.ngpu > 0:
        model = model.to(device)

    return model

###################################### Generator module #############################################
class ResidualBlock(nn.Module):
    def __init__(self,channels):
        super(ResidualBlock,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1,bias=True),
            nn.InstanceNorm2d(channels,affine=True),
            nn.ReLU(True),
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1,bias=True),
            nn.InstanceNorm2d(channels,affine=True),
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.block(x)
        out += x
        out = self.relu(out)
        return out

def block_encoder(in_channels,out_channels):
    block = nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=2,padding=1,bias=True),
        nn.InstanceNorm2d(out_channels,affine=True),
        nn.ReLU(True)
    )
    return block

def block_decoder(in_channels,out_channels):
    block = nn.Sequential(
        nn.ConvTranspose2d(in_channels,out_channels,kernel_size=3,stride=2,padding=1,output_padding=1,bias=True),
        nn.InstanceNorm2d(out_channels,affine=True),
        nn.ReLU(True)
    )
    return block

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.ngf = opt.ngf # 32
        self.in_channels = opt.num_img # 3
        self.num_age = opt.num_age

        self.encoder = nn.Sequential(
            # 3 x 224 x 224 -> 32 x 224 x 224
            nn.Conv2d(self.in_channels,self.ngf,kernel_size=9,stride=1,padding=4,bias=True),
            nn.InstanceNorm2d(self.ngf,affine=True),
            nn.ReLU(True),
            # 32 x 224 x 224 -> 64 x 112 x 112
            block_encoder(self.ngf,self.ngf*2),
            # 64 x 112 x 112 -> 128 x 56 x 56
            block_encoder(self.ngf*2,self.ngf*4)
        )

        # 128 x 56 x 56 -> 128 x 56 x 56
        self.residual = nn.Sequential(
            ResidualBlock(self.ngf*4),
            ResidualBlock(self.ngf*4),
            ResidualBlock(self.ngf*4),
            ResidualBlock(self.ngf*4)
        )

        self.decoder = nn.Sequential(
            # 128 x 56 x 56 -> 64 x 112 x 112
            block_decoder(self.ngf*4,self.ngf*2),
            # 64 x 112 x 112 -> 32 x 224 x 224
            block_decoder(self.ngf*2,self.ngf),
            # 32 x 224 x 224 -> 3 x 224 x 224
            nn.ConvTranspose2d(self.ngf,self.in_channels,kernel_size=9,stride=1,padding=4,bias=True),
            nn.Tanh()
        )

    def forward(self,x):
        x = self.encoder(x)
        x = self.residual(x)
        x = self.decoder(x)
        return x

########################################### Discriminator module ####################################
def block_discriminator(in_channels,out_channels):
    block = nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=4,stride=2,padding=1,bias=True),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2,inplace=True)
    )
    return block

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.ndf = opt.ndf # 64

        # 64 x 224 x 224 -> 128 x 112 x 112
        # self.stage1 = block_discriminator(self.ndf,self.ndf*2)
        # 128 x 112 x 112 -> 256 x 56 x 56 
        # self.stage2 = block_discriminator(self.ndf*2,self.ndf*4)
        # 256 x 56 x 56 -> 512 x 28 x 28
        # self.stage3 = block_discriminator(self.ndf*4,self.ndf*8)
        # 512 x 28 x 28 -> 512 x 14 x 14
        # self.stage4 = block_discriminator(self.ndf*8,self.ndf*8)
        # 512 x 14 x 14 -> 512 x 7 x 7
        # self.stage5 = block_discriminator(self.ndf*8,self.ndf*8)
        # 512 x 7 x 7 -> 1 x 3 x 3
        # self.stage6 = nn.Conv2d(self.ngf*8,1,kernel_size=4,stride=2,padding=1,bias=False)
    
        self.path4 = nn.Sequential(
            block_discriminator(self.ndf*8,self.ndf*8),
            block_discriminator(self.ndf*8,self.ndf*8),
            nn.Conv2d(self.ndf*8,1,kernel_size=4,stride=2,padding=1,bias=True)
        )

        self.path3 = nn.Sequential(
            block_discriminator(self.ndf*4,self.ndf*8),
            block_discriminator(self.ndf*8,self.ndf*8),
            block_discriminator(self.ndf*8,self.ndf*8),
            nn.Conv2d(self.ndf*8,1,kernel_size=4,stride=2,padding=1,bias=True)
        )

        self.path2 = nn.Sequential(
            block_discriminator(self.ndf*2,self.ndf*4),
            block_discriminator(self.ndf*4,self.ndf*8),
            block_discriminator(self.ndf*8,self.ndf*8),
            block_discriminator(self.ndf*8,self.ndf*8),
            nn.Conv2d(self.ndf*8,1,kernel_size=4,stride=2,padding=1,bias=True)
        )

        self.path1 = nn.Sequential(
            block_discriminator(self.ndf,self.ndf*2),
            block_discriminator(self.ndf*2,self.ndf*4),
            block_discriminator(self.ndf*4,self.ndf*8),
            block_discriminator(self.ndf*8,self.ndf*8),
            block_discriminator(self.ndf*8,self.ndf*8),
            nn.Conv2d(self.ndf*8,1,kernel_size=4,stride=2,padding=1,bias=True)
        )

    def forward(self,h1,h2,h3,h4):
        out1 = self.path1(h1).squeeze(1) # [batch_size,1,3,3] -> [batch_size,3,3]
        out2 = self.path2(h2).squeeze(1)
        out3 = self.path3(h3).squeeze(1)
        out4 = self.path4(h4).squeeze(1)
        out = torch.cat((out1,out2,out3,out4),1)
        return out

################################## Dataloader ######################################################
def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    # print(classes)
    # print(class_to_idx)
    return classes, class_to_idx

def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    # print(images)
    return images

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

####################### test dataloader ###################################
class test_dataset(torch.utils.data.Dataset):
    def __init__(self, root, extensions, loader, transform=None):
        classes, class_to_idx = find_classes(root) # classes is list. class_to_idx is dict.
        samples = make_dataset(root, class_to_idx, extensions) # samples is list.
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))

        self.root = root
        self.loader = loader
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        name = os.path.basename(path).split('.')[0]
        return sample,name

    def __len__(self):
        return len(self.samples)

def test_dataloader():
    transform = transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    dataset = test_dataset(root=opt.test_dataroot,extensions=IMG_EXTENSIONS,loader=default_loader,transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=opt.batch_size,shuffle=True,num_workers=8)
    return dataloader

####################### train dataloader ###################################
def young_dataloader():
    transform = transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    dataset = datasets.ImageFolder(opt.young_dataroot,transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=opt.batch_size,shuffle=True,num_workers=8)
    return dataloader

def elder_dataloader():
    transform = transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        # transforms.Normalize([0.43184996,0.35932106,0.31960008],[0.33192977,0.2960793,0.28740713])
    ])
    dataset = datasets.ImageFolder(opt.elder_dataroot,transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=opt.batch_size,shuffle=True,num_workers=8)
    return dataloader

####################### val dataloader ###################################
class val_dataset(torch.utils.data.Dataset):
    def __init__(self, root, extensions, loader, transform=None):
        classes, class_to_idx = find_classes(root) # classes is list. class_to_idx is dict.
        samples = make_dataset(root, class_to_idx, extensions) # samples is list.
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))

        self.root = root
        self.loader = loader
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        name = os.path.basename(path).split('.')[0]
        return sample,name

    def __len__(self):
        return len(self.samples)

def val_dataloader():
    transform = transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    dataset = val_dataset(root=opt.val_dataroot,extensions=IMG_EXTENSIONS,loader=default_loader,transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=opt.batch_size,shuffle=True,num_workers=8)
    return dataloader

################################## PAGAN module ############################################################
class PAGAN():
    def __init__(self):
        self.G_net,self.D_net = self.load_model()
        self.vgg_age = vgg_age(opt.age_weights_pth)
        self.vgg_identity = vgg_identity(opt.identity_weights_pth)
        self.lr_D = opt.lr
        self.lr_G = opt.lr
        self.optim_D = optim.Adam(self.D_net.parameters(),lr=self.lr_D,betas=(0.5,0.999))
        self.optim_G = optim.Adam(self.G_net.parameters(),lr=self.lr_G,betas=(0.5,0.999))
        self.dataloader_young = young_dataloader()
        self.dataloader_elder = elder_dataloader()
        self.data_iter_young = iter(self.dataloader_young)
        self.data_iter_elder = iter(self.dataloader_elder)

    def train(self):
        vis = visdom.Visdom(env=opt.env)
        pixel_loss_win = vis.line(np.arange(10))
        z_loss_win = vis.line(np.arange(10))
        gan_loss_win = vis.line(X=np.column_stack((np.array(0),np.array(0))),Y=np.column_stack((np.array(0),np.array(0))))
        iter_count_1 = 0

        start_t = time.time()
        epoch_start_t = time.time()
        self.D_net.train()
        for i in range(opt.iters):
            if i % 2000 == 0:
                self.optim_D = optim.Adam(self.D_net.parameters(),lr=self.lr_D,betas=(0.5,0.999),weight_decay=0.5)
                self.optim_G = optim.Adam(self.G_net.parameters(),lr=self.lr_G,betas=(0.5,0.999),weight_decay=0.5)
            else:
                self.optim_D = optim.Adam(self.D_net.parameters(),lr=self.lr_D,betas=(0.5,0.999))
                self.optim_G = optim.Adam(self.G_net.parameters(),lr=self.lr_G,betas=(0.5,0.999))
            self.G_net.train()
            # Fetch a batch young samples
            try:
                young_img, _ = next(self.data_iter_young)
            except:
                self.data_iter_young = iter(self.dataloader_young)
                young_img, _ = next(self.data_iter_young)
            # Fetch a batch elderly samples
            try:
                elderly_img, _ = next(self.data_iter_elder)
            except:
                self.data_iter_elder = iter(self.dataloader_elder)
                elderly_img, _ = next(self.data_iter_elder)

            elderly_img = elderly_img.to(device)
            input_img = young_img.to(device)

            # elderly & young is different sometimes
            elderly_batch_size = elderly_img.size(0)
            young_batch_size = young_img.size(0)

            elderly_real_label = torch.ones((elderly_batch_size, 12, 3)).to(device)
            real_label = torch.ones((young_batch_size, 12, 3)).to(device)
            fake_label = torch.zeros((young_batch_size, 12, 3)).to(device)

            ############# D ######################
            self.optim_D.zero_grad()

            # elder samples:only update D     
            elder_h1,elder_h2,elderly_h3,elderly_h4,_ = self.vgg_age(elderly_img.detach())
            age_elderly_img_logits = self.D_net(elder_h1,elder_h2,elderly_h3,elderly_h4)
            # real
            d_loss_real = elder_GAN_D_loss(age_elderly_img_logits,elderly_real_label)
            D_loss_real = opt.par_ad_d * d_loss_real

            # young & generate samples
            fake_img = self.G_net(input_img) 
            fake_h1,fake_h2,fake_h3,fake_h4,_ = self.vgg_age(fake_img.detach())
            age_fake_img_logits = self.D_net(fake_h1,fake_h2,fake_h3,fake_h4)
            input_h1,input_h2,input_h3,input_h4,_ = self.vgg_age(input_img)
            age_input_img_logits = self.D_net(input_h1,input_h2,input_h3,input_h4)                
            # fake
            d_loss_fake, _, _ = young_GAN_D_loss(age_fake_img_logits,age_input_img_logits,real_label,fake_label)
            D_loss_fake = opt.par_ad_d * d_loss_fake
            
            D_loss = D_loss_real + D_loss_fake
            D_loss.backward()
            self.optim_D.step()

            ############# G #################################
            self.optim_G.zero_grad()

            # young & generate samples    
            fake_img = self.G_net(input_img)
            _, _, _, _, id_fake_img = self.vgg_identity(fake_img)
            _, _, _, _, id_input_img = self.vgg_identity(input_img)

            fake_h1,fake_h2,fake_h3,fake_h4,_ = self.vgg_age(fake_img)
            age_fake_img_logits = self.D_net(fake_h1,fake_h2,fake_h3,fake_h4)
            
            loss_identity = identity_loss(id_fake_img,id_input_img)
            g_loss = GAN_G_loss(age_fake_img_logits,real_label)                
            # pixel_loss every five iteration
            if i % 5 == 0:
                loss_pixel = pixel_loss(fake_img,input_img)
                G_loss = opt.par_ad_g * g_loss + opt.par_pix * loss_pixel + opt.par_identity * loss_identity
            else:
                G_loss = opt.par_ad_g * g_loss + opt.par_identity * loss_identity
            G_loss.backward()
            self.optim_G.step()

            if i % 25 == 0:
                print('[iter/total_iter:%d/%d]\t[D_loss:%.7f\tD_loss_fake:%.7f\tD_loss_real:%.7f]\t[G_loss:%.7f\tG_loss_fake:%.7f\tG_loss_pixel:%.7f\tG_loss_identity:%.7f]'
                    %(i,opt.iters,D_loss.item(),d_loss_fake.item(),d_loss_real.item(),G_loss.item(),
                    g_loss.item(),loss_pixel.item(),loss_identity.item()))

            if i % 100 == 0:
                self.save_model(self.G_net,self.D_net,i,opt.model_dir)
                self.visualize_results(input_img,fake_img,i)
                epoch_t = time.time() - epoch_start_t
                print('[iter:{:d}\ttime:{:.0f}h {:.0f}m {:.0f}s]'.format(i,epoch_t//3600,(epoch_t%3600)//60,epoch_t%60))
                epoch_start_t = time.time()

            iter_count_1 += 1
            vis.line(Y=np.array([loss_pixel.item()]),
                    X=np.array([iter_count_1]),
                    update='append',
                    win=pixel_loss_win,
                    opts=dict(legend=['pixel_loss']))
            vis.line(Y=np.array([loss_identity.item()]), 
                    X=np.array([iter_count_1]), 
                    update='append', 
                    win=z_loss_win,
                    opts=dict(legend=['identity_loss']))
            vis.line(Y=np.column_stack((np.array([D_loss.item()]), np.array([G_loss.item()]))),
                    X=np.column_stack((np.array([iter_count_1]), np.array([iter_count_1]))),
                    win=gan_loss_win, update='append',
                    opts=dict(legned=['D_loss', 'G_loss']))
        
        total_t = time.time() - start_t
        print('Training finsh in %.0f h %.0f m %.0f s'%(total_t//3600,(total_t%3600)//60,total_t%60))

    def visualize_results(self,input_img,fake_img,epoch):
        with torch.no_grad():
            self.G_net.eval()
            input_img_vis = input_img.cpu().detach()
            fake_img_vis = fake_img.cpu().detach()
            torchvision.utils.save_image(input_img_vis,'%s/epoch_%d_age_%d_input.png'%(opt.train_img_dir,epoch,opt.train_age),normalize=True,range=(-1,1))
            torchvision.utils.save_image(fake_img_vis,'%s/epoch_%d_age_%d_fake.png'%(opt.train_img_dir,epoch,opt.train_age),normalize=True,range=(-1,1))

    def val(self):
        dataloader_val = val_dataloader()
        self.G_net.eval()
        self.D_net.eval()
        for i,(img,names) in enumerate(dataloader_val):
            val_img = img.to(device)
            fake_img = self.G_net(val_img).cpu().detach()

            for j in range(len(names)):
                name = names[j]
                origin_img = img.cpu().detach()[j]
                save_img = fake_img[j]
                save_path = os.path.join(opt.val_img_dir,name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                input_filename = os.path.join(save_path,'input.png')
                if not os.path.exists(input_filename):
                    torchvision.utils.save_image(origin_img,input_filename,normalize=True,range=(-1,1))
                output_filename = os.path.join(save_path,'%d.png'%(opt.train_age))
                torchvision.utils.save_image(save_img,output_filename,normalize=True,range=(-1,1))

    def test(self):
        dataloader_test = test_dataloader()
        self.G_net.eval()
        self.D_net.eval()
        for i,(img,names) in enumerate(dataloader_test):
            test_img = img.to(device)
            fake_img = self.G_net(test_img).cpu().detach()

            for j in range(len(names)):
                name = names[j]
                origin_img = img.cpu().detach()[j]
                save_img = fake_img[j]
                save_path = os.path.join(opt.test_img_dir,name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                input_filename = os.path.join(save_path,'input.png')
                if not os.path.exists(input_filename):
                    torchvision.utils.save_image(origin_img,input_filename,normalize=True,range=(-1,1))
                output_filename = os.path.join(save_path,'%d.png'%(opt.train_age))
                torchvision.utils.save_image(save_img,output_filename,normalize=True,range=(-1,1))

    def load_model(self):
        print('Training in age:',opt.train_age)    
        D_net = Discriminator()
        G_net = Generator()
 
        if opt.netG_pth != ' ':
            G_net.load_state_dict(torch.load(opt.netG_pth,map_location=lambda storage,loc:storage))##gpu -> cpu
            print('Load from: ',opt.netG_pth)
        if opt.netD_pth != ' ':
            D_net.load_state_dict(torch.load(opt.netD_pth,map_location=lambda storage,loc:storage))##gpu -> cpu
            print('Load from: ',opt.netD_pth)
        if opt.netG_pth == ' ' and opt.netD_pth == ' ':
            D_net.apply(weights_init)
            G_net.apply(weights_init)
            print('no model & init weight')
        if (device.type == 'cuda') and (opt.ngpu > 1):
            G_net = nn.DataParallel(G_net,list(range(opt.ngpu)))
            D_net = nn.DataParallel(D_net,list(range(opt.ngpu)))
        if opt.ngpu > 0:
            G_net.to(device)
            D_net.to(device)
        return G_net,D_net

    def save_model(self,netG,netD,epoch,model_dir):
        if (device.type == 'cuda') and (opt.ngpu > 1): 
            torch.save(netG.module.state_dict(),'%s/netG_epoch_%d.pth'%(model_dir,epoch))
            torch.save(netD.module.state_dict(),'%s/netD_epoch_%d.pth'%(model_dir,epoch))
        else:
            torch.save(netG.state_dict(),'%s/netG_epoch_%d.pth'%(model_dir,epoch))
            torch.save(netD.state_dict(),'%s/netD_epoch_%d.pth'%(model_dir,epoch))
        print('the model has saved(epoch:%d)'%(epoch))

######################################## param #################################################
def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_img', type = int, \
                        default = 3,
                        help = 'the number of image_channel(default:3)')
    parser.add_argument('--img_size', type = int, \
                        default = 224,
                        help = 'the size of input image(default:224)')
    parser.add_argument('--num_age', type = int, \
                        default = 3,
                        help = 'the number of age label')
    # the params of model
    parser.add_argument('--ngf', type = int, \
                        default = 32,
                        help='the base size of generator(default:32)')
    parser.add_argument('--ndf', type = int, \
                        default = 64,
                        help = 'the base size of discriminator(default:64)')
    parser.add_argument('--nvf', type = int, \
                        default = 64,
                        help = 'the base size of vgg16(default:64)')
    # the params when training
    parser.add_argument('--iters', type = int, \
                        default = 50000,
                        help = 'number of epochs(default:50000)')
    parser.add_argument('--batch_size', type = int, \
                        default = 8, 
                        help = 'the batch size(default:8)')
    parser.add_argument('--par_ad_g', type = float, \
                        default = 750, 
                        help = 'parameter of adversarial loss(default:750)')
    parser.add_argument('--par_ad_d', type = float, \
                        default = 1, 
                        help = 'parameter of adversarial loss(default:1)')
    parser.add_argument('--par_pix', type = float, \
                        default = 0.2, 
                        help = 'parameter of pixel_loss(default:0.2)')
    parser.add_argument('--par_identity', type = float, \
                        default = 0.005, 
                        help = 'parameter of identity_loss(default:0.005)')
    parser.add_argument('--lr', type = float, \
                        default = 0.0001,
                        help = 'the learning rate(default:0.0001)')
    # the params of path
    parser.add_argument('--young_dataroot', type = str, \
                        default = './data_train/young/',
                        help = 'the path of young dataset')
    parser.add_argument('--test_dataroot', type = str, \
                        default = './data_train/test/',
                        help = 'the path of test dataset')
    parser.add_argument('--val_dataroot', type = str, \
                        default = './data_train/val/',
                        help = 'the path of val dataset')
    parser.add_argument('--test_img_dir', type = str, \
                        default = './img/test',
                        help = 'the path of saving the test img')
    parser.add_argument('--val_img_dir', type = str, \
                        default = './img/val',
                        help = 'the path of saving the val img')
    parser.add_argument('--age_weights_pth', type = str, \
                        default = './model_vgg/vgg_age.pth',
                        help = 'the path of loading the vgg_age')
    parser.add_argument('--identity_weights_pth', type = str, \
                        default = './model_vgg/vgg_identity.pth',
                        help = 'the path of loading the vgg_identity')
    # the other params
    parser.add_argument('--ngpu', type = str, \
                        default = 1,
                        help = 'the number of gpus available')
    parser.add_argument('--manual_Seed', type = int, \
                        default = 2018,
                        help = 'manual_seed')
    # param may be modify !!!
    parser.add_argument('--train_age', type = int, \
                        default = 3,
                        help = 'the number of training the age on G(eg. 1, 2, 3.)(modify!!!)')                        
    parser.add_argument('--elder_dataroot', type = str, \
                        default = './data_train/elder3/',
                        help = 'the path of elder dataset(modify!!!)')
    parser.add_argument('--model_dir', type = str, \
                        default = './model_crop_3',
                        help = 'the path of saving the model(modify!!!)')
    parser.add_argument('--train_img_dir', type = str, \
                        default = './img/train_crop_3',
                        help = 'the path of saving the train img(modify!!!)')
    parser.add_argument('--netD_pth', type = str, \
                        default = ' ',
                        help = 'the path of loading the discriminator')
    parser.add_argument('--netG_pth', type = str, \
                        default = ' ',
                        help = 'the path of loading the generator')
    parser.add_argument('--env', type = str, \
                        default = 'PAGAN_crop_3',
                        help = 'the name of env of visdom')
    parser.add_argument('--is_training', type = bool, \
                        default = False,
                        help = 'train or test')
    return parser.parse_args()

if __name__ == '__main__':
    # makeDir()
    # moveFiles()
    # moveFiles_test()
    opt = args()
    setup_seed(opt.manual_Seed)
    torch.backends.cudnn.benchmark = True
    if opt.ngpu > 1:
        device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    if not os.path.exists(opt.model_dir):
        os.makedirs(opt.model_dir)
    if not os.path.exists(opt.train_img_dir):
        os.makedirs(opt.train_img_dir)
    if not os.path.exists(opt.test_img_dir):
        os.makedirs(opt.test_img_dir)
    if not os.path.exists(opt.val_img_dir):
        os.makedirs(opt.val_img_dir)
    if opt.is_training == True:
        gan = PAGAN()
        gan.train()
    else:
        opt.ngpu = 1
        gan = PAGAN()
        gan.val()
        gan.test()
