import numpy as np
import pandas as pd
import torch
import torchvision
import csv
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

from customclasses import MNISTDataset, Net

mean = 0.13101533792088266
std = 0.30854016060963374
# print(torch.cuda.is_available())
# exit(0)
# transformations to be applied on images, preprocessing like normalize
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[mean], std=[std]),   # mean, std for each channel
                                ])

# trainset = pd.read_csv("./digit-recognizer/train.csv")
# testset = pd.read_csv("./digit-recognizer/test.csv")

trainset = MNISTDataset("./digit-recognizer/train.csv", True, transform)
testset = MNISTDataset("./digit-recognizer/test.csv", False, transform)

# y = trainset['label'].copy()
# X = trainset.drop(['label'], axis=1)
#
# print(y.shape, X.shape)
#
#
# def viz_num(features, labels, num):
#     image = features.values[num].reshape([28,28])
#     # image = features.values[num]
#     plt.title('Sample:%d, Label:%d' % (num, labels[num]))
#     plt.imshow(image, cmap=plt.get_cmap('gray'))
#     plt.show()
#
#
# viz_num(X, y, 1111)
#
# X_train = X.values.reshape(-1, 1, 28, 28)
# y_train = y
#
# X_test = testset.values.reshape(-1, 1, 28, 28)
#
# print(X_train.shape, X_test.shape)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()

print('Training set')
print(images.shape)
print(labels.shape)

plt.imshow(images[0].numpy().squeeze(), cmap='gray')
plt.show()

#############################################
model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    print('Cuda available')
    model = model.cuda()
    criterion = criterion.cuda()

print(model)

#############################################

for i in range(10):
    running_loss = 0
    for images, labels in trainloader:

        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        # training pass
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        # get model learn
        loss.backward()

        # optimize weights
        optimizer.step()

        running_loss += loss.item()
    else:
        print('Epoch {} - Training loss: {}'.format(i+1, running_loss/len(trainloader)))

#############################################
torch.save(model.state_dict(), "./last.pt")


