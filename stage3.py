from copy import deepcopy
import json
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
import gc
gc.collect()


# --------------------------------------------------------------------- #
# Global Level Setting
# --------------------------------------------------------------------- #

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()

# 1. GTX 1060 6GB / RAM 16GB
# 2. GTX 3070 8GB / RAM 32GB

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
# - Dataset
# --------------------------------------------------------------------- #

class BD_Dataset(Dataset):
    def __init__(self, data_path, img_tag_table, img_coords_table, transforms=None):
        super().__init__()
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
    # TODO : 
    # [x] diff among same class 
    # [x] diff btw two classes 
    def __init__(self, data_path, tag_img_dict, transforms=None, batch=16, mode='SAME'):
        super().__init__()
        self.data_path = data_path
        self.tag_img_dict = tag_img_dict        
        self.transforms = transforms
        self.buffer = None
        self.batch = batch 
        assert mode in ['SAME', 'DIFF']
        self.mode = mode

    def __len__(self):
        return len(list(self.tag_img_dict.keys()))

    def __getitem__(self, idx):
        if self.mode == 'SAME':
            # Solve Memory problem --> Set buffer and slice! 
            if self.buffer is None:
                key = list(self.tag_img_dict.keys())[idx]

                img_names = self.tag_img_dict[key]
                if len(img_names) > self.batch:     #set buffer
                    self.buffer = {"img":img_names[self.batch:], "key": key}
                    img_names = img_names[:self.batch]
                else:
                    self.buffer = None
            else:
                img_names = self.buffer['img'][:self.batch]
                key = self.buffer["key"]
                if len(self.buffer["img"]) <= self.batch:
                    self.buffer = None
                else:
                    self.buffer["img"] = self.buffer["img"][self.batch:]

            # prepare data
            imgs_names = []
            imgs = []
            for img_name in img_names:
                img = cv2.imread(os.path.join(self.data_path, str(img_name) + '.jpg'), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                
                img = Image.fromarray(np.uint8(img)).convert('RGB')

                if self.transforms is not None:
                    img = self.transforms(img)
                
                imgs_names.append(str(img_name) + '.jpg')
                imgs.append(img)
            
            return imgs, imgs_names, key

        elif self.mode == 'DIFF':
            
            # 1st class (anchor) images selection
            anchor_key = list(self.tag_img_dict.keys())[idx]
            anchor_img_names = self.tag_img_dict[anchor_key]

            if len(anchor_img_names) > int(self.batch // 2):
                anchor_img_names = sample(anchor_img_names, self.batch // 2)
            else:
                pass

            # 2nd class (negative) images selection
            sampling = list(self.tag_img_dict.keys()).remove(anchor_key)
            negative_key = sample(sampling, 1)
            negative_img_names = self.tag_img_dict[negative_key]
            
            if len(negative_img_names) > int(self.batch // 2):
                negative_img_names = sample(negative_img_names, self.batch // 2)
            else:
                pass

            # prepare anchor data
            anchor_imgs_names = []
            anchor_imgs = []
            for img_name in anchor_img_names:
                img = cv2.imread(os.path.join(self.data_path, str(img_name) + '.jpg'), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                    
                img = Image.fromarray(np.uint8(img)).convert('RGB')

                if self.transforms is not None:
                    img = self.transforms(img)
                
                anchor_imgs_names.append(str(img_name) + '.jpg')
                anchor_imgs.append(img)

            # prepare negative data
            negative_imgs_names = []
            negative_imgs = []
            for img_name in negative_img_names:
                img = cv2.imread(os.path.join(self.data_path, str(img_name) + '.jpg'), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                    
                img = Image.fromarray(np.uint8(img)).convert('RGB')

                if self.transforms is not None:
                    img = self.transforms(img)

                negative_imgs_names.append(str(img_name) + '.jpg')
                negative_imgs.append(img)


            return anchor_imgs, negative_imgs, anchor_imgs_names, negative_imgs_names, \
                    anchor_key, negative_key

# --------------------------------------------------------------------- #
# My own class & method 
# - Model
# --------------------------------------------------------------------- #

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


# --------------------------------------------------------------------- #
# My own class & method 
# - Loss
# --------------------------------------------------------------------- #

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


class CenterLoss(nn.Module):
    def __init__(self, n_centers=856, table=None, dim=512):
        super().__init__()
        self.cluster_centers = torch.rand(n_centers, dim) * n_centers / dim 
        self.cluster_centers.to(DEVICE)
        """
        Gaussian -> tendency to gather at center
        Uniform distribution is better at this situation
        """
        self.center_table = table
        self.mse = nn.MSELoss()

    def forward(self, batch_tensors, batch_coords):
        loss = 0
        longitude = batch_coords[0]
        latitude = batch_coords[1]
        batch_coords = [(i.item(), j.item()) for i, j in zip(longitude, latitude)]
        for pred, coord in zip(batch_tensors, batch_coords):
            loss += self.mse(pred, self.cluster_centers[self.center_table.index(coord)].to(DEVICE))
        return loss / len(batch_tensors)    # divide len of batch_tensors for scalable loss value


# --------------------------------------------------------------------- #
# My own class & method 
# - Utils
# --------------------------------------------------------------------- #

class ClsOneHot:    # one-hot vector converter
    def __init__(self, tag_list):
        self.tag_list = tag_list 
    
    def convert(self, x):
        # x = (B, 1)
        result = []
        for x_ in x:
            result.append(self.tag_list.index(x_))
        return torch.tensor(result, dtype=torch.int64).view(-1)

        
def compute_distance_same(batch_tensors, k=4):
    batch = batch_tensors.shape[0]
    distance_list = []
    pair_list = []
    for i in range(batch):
        anchor = batch_tensors[i]
        for j in range(i+1, batch):
            compare = batch_tensors[j]
            distance = F.pairwise_distance(anchor, compare)
            distance = torch.sqrt(distance.sum())
            distance_list.append(distance.item())
            pair_list.append((i, j))
    
    distance_list = torch.tensor(distance_list)
    # closest images
    topk_distance_close = torch.topk(distance_list, k=k, largest=False)
    topk_pair_close = torch.tensor(pair_list)[topk_distance_close.indices]

    return topk_distance_close, topk_pair_close


def compute_distance_diff(anchor_tensors, negative_tensors, k=4):
    batch1 = anchor_tensors.shape[0]
    batch2 = negative_tensors.shape[1]

    distance_list = []
    pair_list = []
    for i in range(batch1):
        anchor = anchor_tensors[i]
        for j in range(batch2):
            negative = anchor_tensors[j]
            distance = F.pairwise_distance(anchor, negative)
            distance = torch.sqrt(distance.sum())
            distance_list.append(distance.item())
            pair_list.append((i, j))
        
    distance_list = torch.tensor(distance_list)
    distance_mean = torch.mean(distance_list)
    distance_std = torch.mean(distance_list)
    topk_distance_close = torch.topk(distance_list, k=k, largest=False)
    topk_pair_close = torch.tensor(pair_list)[topk_distance_close.indices]

    return topk_distance_close, topk_pair_close, distance_mean, distance_std


VIZ_PATH = 'visualize'


def visualize_imgs(imgs_names, img_pairs, data_path, key):

    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = len(img_pairs)
    for i in range(len(img_pairs)):
        img = cv2.imread(os.path.join(data_path, imgs_names[img_pairs[i][0]]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig.add_subplot(rows, columns, 2*i+1)
        plt.imshow(img)
        img = cv2.imread(os.path.join(data_path, imgs_names[img_pairs[i][1]]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig.add_subplot(rows, columns, 2*i+2)
        plt.imshow(img)

    # plt.show()
    if os.path.exists(f'{VIZ_PATH}/viz_{key}.jpg'):
        count = 2
        while os.path.exists(f'{VIZ_PATH}/viz_{key}_{count}.jpg'):
            count += 1
        plt.savefig(f'{VIZ_PATH}/viz_{key}_{count}.jpg')
    else:
        plt.savefig(f'{VIZ_PATH}/viz_{key}.jpg')

def visualize_imgs2(anchor_names, negative_names, img_pairs, data_path, key1, key2):

    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = len(img_pairs)
    for i in range(len(img_pairs)):
        img = cv2.imread(os.path.join(data_path, anchor_names[img_pairs[i][0]]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig.add_subplot(rows, columns, 2*i+1)
        plt.imshow(img)
        img = cv2.imread(os.path.join(data_path, negative_names[img_pairs[i][1]]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig.add_subplot(rows, columns, 2*i+2)
        plt.imshow(img)

    # plt.show()
    if os.path.exists(f'{VIZ_PATH}/viz_{key1}_{key2}.jpg'):
        count = 2
        while os.path.exists(f'{VIZ_PATH}/viz_{key1}_{key2}_{count}.jpg'):
            count += 1
        plt.savefig(f'{VIZ_PATH}/viz_{key1}_{key2}_{count}.jpg')
    else:
        plt.savefig(f'{VIZ_PATH}/viz_{key1}_{key2}.jpg')


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
    coords_list = [(round(val[0], 5), round(val[1], 5)) for val in coords]
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
    center_loss = CenterLoss(n_centers=len(coords_unqiue), table=coords_unqiue, dim=512)

    epoch = 120
    lr = 0.0001
    batch = BATCH
    save_period = 10
    log_step = 20
    balance_weights = [1, 1, 0.5]

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

            loss = balance_weights[0] * loss1 + balance_weights[1] * loss2 + balance_weights[2] * loss3
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
                        f'model_stage3_{epoch_}.pth')

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

    test_dataset = BD_TestDataset(img_data_path, tag_n_photoid, transforms=transform, batch=BATCH, mode='SAME')

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
        topk_dist, topk_pair = compute_distance_same(eval_feats, \
                                        k= 4 if len(test_imgs) >= 4 else len(test_imgs) -1)


        visualize_imgs(test_names, topk_pair, data_path=img_data_path, key=test_key)
                
        logger.info(f'EVAL {test_key} : {topk_dist.values}  PAIR : {topk_pair}')

    test_dataset.mode = 'DIFF'

    for i in range(len(test_dataset)):
        anchor_imgs, negative_imgs, anchor_names, negative_names, anchor_key, negative_key = test_dataset[i]
        anchor_imgs = anchor_imgs.to(DEVICE)
        negative_imgs = negative_imgs.to(DEVICE)
        anchor_feats, _ = model(anchor_imgs)
        negative_feats, _ = model(negative_imgs)
        topk_dist, topk_pair, diff_mean, diff_std = compute_distance_diff(anchor_feats, negative_feats, \
            k=4 if min(len(anchor_imgs), len(negative_imgs)) >= 4 else min(len(anchor_imgs), len(negative_imgs)))
        
        visualize_imgs2(anchor_names, negative_names, topk_pair, data_path=img_data_path, key1=anchor_key, key2=negative_key)

        logger.info(f'EVAL {anchor_key} vs {negative_key} :: MEAN = {diff_mean} STD = {diff_std}')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--batch', type=int, default=8)

    opt = parser.parse_args()

    BATCH = opt.batch

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
    
