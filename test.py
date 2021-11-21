# importing the libraries
import numpy as np
import torch
import csv
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from customclasses import MNISTDataset, Net

mean = 0.13101533792088266
std = 0.30854016060963374

# # transformations to be applied on images
# transform = transforms.Compose([transforms.ToTensor(),
#                               transforms.Normalize((0.5,), (0.5,)),
#                               ])
#
# # defining the training and testing set
# trainset = datasets.MNIST('./data', download=True, train=True, transform=transform)
# testset = datasets.MNIST('./', download=True, train=False, transform=transform)
#
#
# # defining trainloader and testloader
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
#
# # shape of training data
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
#
# print(images.dtype, labels.dtype)
# print(images.shape)
# print(labels.shape)
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[mean], std=[std]),   # mean, std for each channel
                                ])

testset = MNISTDataset("./digit-recognizer/test.csv", False, transform)

model = Net()
if torch.cuda.is_available():
    print('Cuda is available')
    model = model.cuda()
model.load_state_dict(torch.load('./last.pt'))
model.eval()

#############################################
# test
#############################################

testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

def write_csv(conts):
    with open('submission.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(conts)

write_csv(['ImageId','Label'])

idx = 1
for images in testloader:
    for i in range(len(images)):
        if torch.cuda.is_available():
            images = images.cuda()
        img = images[i].view(1, 1, 28, 28)
        with torch.no_grad():
            logps = model(img)

        ps = torch.exp(logps)
        probab = list(ps.cpu()[0])
        pred_label = probab.index(max(probab))

        write_csv([str(idx), str(pred_label)])
        idx += 1
