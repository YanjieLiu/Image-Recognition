#coding:utf-8
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from torchvision import transforms
from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import Dataset,DataLoader
import time
import matplotlib.pyplot as plt 
path = './kaggledogvscat/'
def imshow(inp, name,cmap=None):
    """Imshow for Tensor."""
#     print(inp)
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp,cmap)
    plt.savefig(name+".png")
    plt.close('all')

# 检查是否用GPU
is_cuda = False
if torch.cuda.is_available():
    is_cuda = True

# 预处理图片数
simple_transform = transforms.Compose([transforms.Resize((224,224))
                                       ,transforms.ToTensor()
                                       ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ])
train = ImageFolder(path+'train/',simple_transform)
valid = ImageFolder(path+'valid/',simple_transform)


# 加载训练和测试数
train_data_loader = torch.utils.data.DataLoader(train,batch_size=32,num_workers=0,shuffle=True)
valid_data_loader = torch.utils.data.DataLoader(valid,batch_size=32,num_workers=0,shuffle=True)
imshow(valid[2][0],'cat')
imshow(valid[1333][0],'dog') #1333

# for img, label in valid_data_loader:
#     print(img.size(), img.float().mean(), label)
#卷积网络模型

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(56180, 500)
        self.fc2 = nn.Linear(500,50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x,training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x,dim=1)
model = Net()
if is_cuda:
    model.cuda()
# 设置优化
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.5)
#训练或测试数
def fit(epoch,model,data_loader,phase='训练',volatile=False):
    if phase == '训练':
        model.train()
    if phase == '验证':
        model.eval()
        volatile=True
    running_loss = 0.0
    running_correct = 0
    for batch_idx , (data,target) in enumerate(data_loader):
        if is_cuda:
            data,target = data.cuda(),target.cuda()
        data , target = Variable(data,volatile),Variable(target)
        if phase == '训练':
            optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output,target)
        
        running_loss += F.nll_loss(output,target,reduction='sum').item()
        preds = output.data.max(dim=1,keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == '训练':
            loss.backward()
            optimizer.step()
    
    loss = running_loss/len(data_loader.dataset)
    accuracy = float(100. * running_correct/len(data_loader.dataset))
    
    print(f'迭代{epoch}次：{phase}损失  {loss:{5}.{2}} {phase}正确率是 {accuracy:{7}.{4}}% ({running_correct}/{len(data_loader.dataset)})')
    return loss,accuracy
train_losses , train_accuracy = [],[]
val_losses , val_accuracy = [],[]
# print(train.class_to_idx)
print(f'数据集包含类别：{train.classes}') 
for epoch in range(1,21): # 20
    epoch_loss, epoch_accuracy = fit(epoch,model,train_data_loader,phase='训练')
    val_epoch_loss , val_epoch_accuracy = fit(epoch,model,valid_data_loader,phase='验证')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False
plt.plot(range(1,len(train_losses)+1),train_losses,'bo',label = '训练损失')
plt.plot(range(1,len(val_losses)+1),val_losses,'r',label = '验证损失')
plt.xlabel("迭代次数")
plt.legend()
# plt.show()
plt.savefig("loss.png")
plt.close('all')

plt.plot(range(1,len(train_accuracy)+1),train_accuracy,'bo',label = '训练正确')
plt.plot(range(1,len(val_accuracy)+1),val_accuracy,'r',label = '验证正确')
plt.xlabel("迭代次数")
plt.legend()
# plt.show()
plt.savefig("accuracy.png")
