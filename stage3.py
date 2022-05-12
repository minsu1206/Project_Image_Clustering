from copy import deepcopy
import json
from pydoc import apropos
import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import Dataset, DataLoader
import glob
import os
import cv2
import pandas as pd
from PIL import Image
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from random import sample
import matplotlib.pyplot as plt

# --------------------------------------------------------------------- #
# Global Level Setting
# --------------------------------------------------------------------- #

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# GTX 1060 6GB

import logging


def get_logger(exp_name='BigDataMidTerm_stage3'):
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    import time
    log_time = time.strftime("%Y-%m-%d-%H-%M", time.gmtime())
    file_handler = logging.FileHandler(f'log/{exp_name}_{log_time}.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

logger = get_logger()



# --------------------------------------------------------------------- #
# My own class & method 
# --------------------------------------------------------------------- #

class BD_Dataset(Dataset):
    def __init__(self, data_path, img_tag_table, img_coords_table, transforms=None):
        super().__init__()
        # self.images = list(glob.glob(os.path.join(data_path, '*.jpg')))
        self.data_path = data_path
        self.img_tag_table = img_tag_table
        self.img_coords_table = img_coords_table
        self.transforms = transforms

    def __len__(self):
        return len(self.img_tag_table)

    def __getitem__(self, idx):
        tag_table_row = self.img_tag_table.loc[idx]
        img_id = tag_table_row['photo_id']
        coords_table_row_idx = self.img_coords_table[self.img_coords_table['photo_id'] == img_id].index[0]
        coords_table_row = self.img_coords_table.loc[coords_table_row_idx]
        coords = (round(coords_table_row['longitude'],5), round(coords_table_row['latitude'],5))
        img_path = os.path.join(self.data_path, str(int(img_id)) + '.jpg')
        

        try:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            
            tag = tag_table_row['tag']
            if img is None:
                self.__getitem__((idx + 1) // self.__len__() )

            img = Image.fromarray(np.uint8(img)).convert('RGB')
        except:
            return self.__getitem__((idx + 1) // self.__len__())

        if self.transforms is not None:
            img = self.transforms(img)

        return img, tag, coords


class BD_TestDataset(Dataset):
    def __init__(self, data_path, tag_img_dict, transforms=None):
        super().__init__()
        self.data_path = data_path
        self.tag_img_dict = tag_img_dict        
        self.transforms = transforms

    def __len__(self):
        return len(list(self.tag_img_dict.keys()))

    def __getitem__(self, idx):
        
        key = list(self.tag_img_dict.keys())[idx]

        img_names = self.tag_img_dict[key]

        imgs = []
        imgs_names = []
        for img_name in img_names:
            img = cv2.imread(self.data_path, img_name + '.jpg')
            if img is None:
                continue
            
            img = Image.fromarray(np.uint8(img)).convert('RGB')

            if self.transforms is not None:
                img = self.transforms(img)
            
            imgs_names.append(img_name + '.jpg')
            imgs.append(img)
        
        if self.transforms is not None:
            return torch.tensor(imgs), imgs_names, key
        else:
            return imgs, imgs_names, key


class ResNetWrapper(nn.Module):
    def __init__(self, backbone, n_cls):
        super().__init__()
        self.conv1 = backbone._modules['conv1']
        self.bn1 = backbone._modules['bn1']
        self.relu = backbone._modules['relu']
        self.maxpool = backbone._modules['maxpool']

        self.layer1 = backbone._modules['layer1']
        self.layer2 = backbone._modules['layer2']
        self.layer3 = backbone._modules['layer3']
        self.layer4 = backbone._modules['layer4']
        self.squeeze = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.avgpool = backbone._modules['avgpool']
        self.fc = nn.Linear(in_features=512, out_features=n_cls)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.squeeze(x)

        feat_vec = self.avgpool(x)
        cls_vec = self.fc(feat_vec.reshape(x.size(0), -1))
        return feat_vec, cls_vec


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.5):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label1, label2):
        label = label1 == label2
        label = label.float().to(DEVICE) 
        euclidean_dist = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (1-label)*torch.pow(euclidean_dist, 2) + \
                (label) * torch.pow(torch.clamp(self.margin - euclidean_dist, min=0.0), 2))
        return loss_contrastive


class MultiContrastiveLoss(nn.Module):
    def __init__(self, margin=1.5, dim=1000):
        super().__init__()
        self.margin = margin
        self.loss = ContrastiveLoss(self.margin)
        self.dim = dim

    def forward(self, output, gt_label):
        loss = 0
        anchor = output
        compare = output
        compare_label = gt_label

        for _ in range(output.shape[0]):
            compare = torch.roll(output, 1, 0)
            compare_label = torch.roll(compare_label, 1, 0)
            loss += self.loss(anchor, compare, gt_label, compare_label)
        
        # divide feature vectors's dimension size for scalable loss value
        return loss / self.dim


class ClsOneHot:
    def __init__(self, tag_list):
        self.tag_list = tag_list 
    
    def convert(self, x):
        # x = (B, 1)
        result = []
        for x_ in x:
            result.append(self.tag_list.index(x_))
        return torch.tensor(result, dtype=torch.int64).view(-1)


class CenterLoss(nn.Module):
    def __init__(self, n_centers=856, table=None, dim=512):
        super().__init__()
        self.cluster_centers = torch.rand(n_centers, dim) * n_centers / dim # uniform distribution
        # Gaussian -> tendency to gather at center
        self.center_table = table
        self.mse = nn.MSELoss()

    def forward(self, batch_tensors, batch_coords):
        loss = 0
        for pred, coord in zip(batch_tensors, batch_coords):
            loss += self.mse(pred, self.cluster_centers[self.center_table.index(coord)])
        return 
        
def compute_distance(batch_tensors):
    batch = batch_tensors.shape[0]
    distance_list = []
    pair_list = []
    for i in range(batch):
        anchor = batch_tensors[i]
        for j in range(i+1, batch):
            compare = batch_tensors[j]
            distance = F.pairwise_distance(anchor, compare)
            distance_list.append(distance)
            pair_list.append((i, j))

    topk_distance = torch.topk(torch.tensor(distance_list), k=4)
    topk_pair = torch.tensor(pair_list)[topk_distance.indices]

    return topk_distance, topk_pair

def visualize_imgs(imgs, data_path, key):
    w = 10
    h = 10
    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = 2
    for i in range(len(imgs)):
        img = cv2.imread(os.path.join(data_path, imgs[i]))
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    
    # plt.show()
    plt.savefig(f'viz_{key}.jpg')


# --------------------------------------------------------------------- #
# My own code for model training
# --------------------------------------------------------------------- #

def train(resume=None):

    tag_filtered_path = 'tag_filtered.csv'
    img_data_path = 'Photo_new'
    filtered_photos = 'filtered_photos.csv'

    checkpoint = None
    if resume is not None:
        checkpoint = torch.load(resume)

    # Configure image transformation for resnet model
    # stage3 version
    transform = tv.transforms.Compose([
        tv.transforms.RandomHorizontalFlip(0.5),
        tv.transforms.RandomResizedCrop((224, 224), scale=(0.6, 1.0), ratio=(0.8, 1.2)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Bring tag & photo_id pair table
    tag_table = pd.read_csv(tag_filtered_path, usecols=[1, 3])      # only use photo_id & tag
    unique_tags = list(set(tag_table.values[:, 1]))
    # print(tag_table.head())   # Just test!

    photo_table = pd.read_csv(filtered_photos, usecols=[1, 2, 3])   # photo_id, longitude, latitude
    coords = photo_table.values[:, [1, 2]]
    coords_list = [[round(val[0], 5), round(val[1], 5)] for val in coords]
    coords_unqiue = []
    for coords in coords_list:
        if coords not in coords_unqiue:
            coords_unqiue.append(coords)
    # print(len(coords_unqiue))   # 856
    # print(photo_table.head())     # Just test!

    
    # Build Dataset for training
    dataset = BD_Dataset(data_path=img_data_path, img_tag_table=tag_table, img_coords_table=photo_table, transforms=transform)


    # ex_img, ex_label = BD_dataset[0]    # Just test!
    # print(ex_img.shape, ex_label)       # Just test!

    # Load pretrained ResNet50 model
    model = tv.models.resnet50(pretrained=True, progress=True)
    model = ResNetWrapper(backbone=model, n_cls=len(unique_tags))

    if resume is not None:
        model.load_state_dict(checkpoint['model_state_dict'])

    # Configure loss function / optimizer / scheduler / training scheme
    cls_loss = nn.CrossEntropyLoss()
    contrastive_loss = MultiContrastiveLoss()
    center_loss = CenterLoss(n_centers=len(coords_unqiue), table=coords_list, dim=512)

    epoch = 120
    lr = 0.0001
    batch = 24
    save_period = 10
    log_step = 20
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.5)

    if resume is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Training
    
    onehot = ClsOneHot(tag_list=list(set(tag_table.values[:, 1])))

    model.to(DEVICE)

    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)

    for epoch_ in range(epoch):
        epoch_loss = [0, 0, 0]
        for i, (imgs, tags, coords) in tqdm(enumerate(dataloader)):
            imgs = imgs.to(DEVICE)
            gt_cls = onehot.convert(tags)
            gt_cls = gt_cls.to(DEVICE)
            pred_feat, pred_cls = model(imgs)
            pred_cls = pred_cls.float()
            gt_cls = gt_cls.long()

            loss1 = cls_loss(pred_cls, gt_cls)    
            # nn.CrossEntropyLoss : prediction = (B, C), float // GT = (B), long
            loss2 = contrastive_loss(pred_feat, pred_cls).cuda()

            loss3 = center_loss(pred_feat, coords)

            loss = loss1 + loss2 + loss3
            epoch_loss[0] += loss1.item()
            epoch_loss[1] += loss2.item()
            epoch_loss[2] += loss3.item()

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % log_step == 0:
                logger.info(f'Epoch : {epoch_} Step : {i} Cls Loss : {loss1.item()} Dist Loss : {loss2.item()} Center Loss : {loss3.item()}')

        epoch_loss[0] /= len(dataloader)
        epoch_loss[1] /= len(dataloader)
        epoch_loss[2] /= len(dataloader)

        scheduler.step()

        if epoch_ and epoch_ % save_period == 0:
            torch.save({'model_state_dict':model.state_dict(),
                        'epoch': epoch_,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()}, 
                        f'model_{epoch_}.pth')

    return model

# --------------------------------------------------------------------- #
# My own code for test
# --------------------------------------------------------------------- #

def test(trained_model=None, load=False):
    

    # Build test dataset 
    with open('tag_n_photoid.json', 'r') as f:
        tag_n_photoid = json.load(f)

    img_data_path = 'Photo_new'

    logger = get_logger('BigDataMidTerm_EVAL')

    transform = tv.transforms.Compose([
        tv.transforms.Resize((224, 224)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_dataset = BD_TestDataset(img_data_path, tag_n_photoid, transforms=transform)


    # Load trained model
    if trained_model is not None:
        model = trained_model
    else:
        unique_tags = 173 # already checked at train()
        model = tv.models.resnet50(pretrained=True, progress=True)
        model = ResNetWrapper(backbone=model, n_cls=unique_tags)
        if load:
            model.load_state_dict(load)
    model.to(DEVICE)

    for i in range(len(test_dataset)):
        test_imgs, test_names, test_key = test_dataset[i]
        test_imgs.to(DEVICE)
        eval_feats, _ = model(test_imgs)
        topk_dist, topk_pair = compute_distance(eval_feats, k=min(len(test_imgs), 4))

        if len(topk_dist) == 4:
            visualize_imgs(test_names, topk_pair)
                
        logger.info(f'EVAL {test_key} : {topk_dist}')




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--load', type=str, default='')

    opt = parser.parse_args()

    if opt.train:
        resume = None
        if opt.load:
            # TODO : resume training 
            resume = opt.load
        model_ = train(resume)

    if opt.eval:
        state_dict=False
        if opt.load:
            state_dict = torch.load(opt.load, map_location=DEVICE)
        test(trained_model=None, load=state_dict)
    