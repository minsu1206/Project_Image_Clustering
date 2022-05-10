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
import torch.nn.Functional as F
from tqdm import tqdm

# with open('tag_n_photoid.json', 'r') as f:
#     tag_id_label = json.load(f)

# print(len(tag_id_label.keys()))    # 173 


class BD_Dataset(Dataset):
    def __init__(self, data_path, img_tag_table, transforms=None):
        super().__init__()
        self.images = glob.glob(os.path.join(data_path, '*.jpg'))
        self.img_tag_table = img_tag_table
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = Image.fromarray(np.uint8(img)).convert('RGB')
        img_name = os.path.basename(img_path).split('.')[0]
        tag = self.img_tag_table[int(img_name)]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, tag

class ResNetWrapper(nn.Module):
    def __init__(self, backbone, cls):
        backbone.trai
        self.conv1 = backbone._modules['conv1']
        self.bn1 = backbone._modules['bn1']
        self.relu = backbone._modules['relu']
        self.maxpool = backbone._modules['maxpool']

        self.layer1 = backbone._modules['layer1']
        self.layer2 = backbone._modules['layer2']
        self.layer3 = backbone._modules['layer3']
        self.layer4 = backbone._modules['layer4']

        self.avgpool = backbone._modules['avgpool']

        self.fc = nn.Linear(2048, cls)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feat_vec = self.avgpool(x)
        cls_vec = self.fc(feat_vec)
        return feat_vec, cls_vec


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.5):
        super().__init__()
        self.margin = margin

    # FIXME 1:1 계산 말고 matrix 로 한번에 쫙 되도록...
    def forward(self, output1, output2, label1, label2):
        label = label1 == label2
        euclidean_dist = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (1-label)*torch.pow(euclidean_dist, 2) + \
                (label) * torch.pow(torch.clamp(self.margin - euclidean_dist, min=0.0)))

        return loss_contrastive

if __name__ == "__main__":
    tag_filtered_path = 'tag_filtered.csv'
    img_data_path = 'Photos'

    # Load pretrained ResNet50 model
    model = tv.models.resnet50(pretrained=True, progress=True)

    # Configure image transformation for resnet model
    transform = tv.transforms.Compose([
        tv.transforms.Resize((224, 224)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Bring tag & photo_id pair table
    tag_table = pd.read_csv(tag_filtered_path, usecols=[1, 3])      # only use photo_id & tag
   
    # # Just test!
    # # print(tag_table.head())

    imgname_tag_dict = {}

    for photo_id, tag in zip(tag_table.values[:, 0], tag_table.values[:, 1]):
        imgname_tag_dict[photo_id] = tag    


    # Build Dataset for training
    BD_dataset = BD_Dataset(data_path=img_data_path, img_tag_table=imgname_tag_dict, transforms=transform)

    # # Just test!
    # # ex_img, ex_label = BD_dataset[0]
    # # print(ex_img.shape, ex_label)

    
    # Configure loss function / optimizer / scheduler / training scheme
    cls_loss = nn.CrossEntropyLoss()
    contrastive_loss = ContrastiveLoss()

    epoch = 300
    lr = 0.0001
    batch = 8
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.5)

    # Training
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    model.to(device)

    dataloader = DataLoader(BD_dataset, batch_size=8, shuffle=True)

    for epoch_ in range(len(epoch)):
        for i, (imgs, tags) in enumerate(dataloader):
            imgs = imgs.to(device)
            tags = tags.to(device)

            loss = 

        




    




