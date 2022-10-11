import numpy as np
import pandas as pd
import matplotlib as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# CNN Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32,3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# DNN Model
class Net(nn.Midule):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 10)
    def forward(self, x):
        x = x.float()
        h1 = F.relu(self.fc1(x.view(-1, 784)))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(h3))
        h5 = F.relu(self.fc5(h4))
        h6 = F.relu(self.fc6(h5))
        return F.log_softmax(h6, dim=1)

# Set Hyper parameters and other variables to train the model
# reference: https://xangmin.tistory.com/129
batch_size = 64
test_batch_size = 1000
epochs = 10
lr = 0.01
momentum = 0.5
no_cuda = True
seed = 1
log_interval = 200

use_cuda = not no_cuda and torch.cuda.is_available()
#cpu 연산 무작위 고정
torch.manual_seed(seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
print("set vars and device done")


# Prepare Data Loader for Training and Validation.
# 훈련데이터 60,000개 + 테스트데이터 10,000
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=transform), 
    batch_size = batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transform), 
    batch_size=test_batch_size, shuffle=True, **kwargs)

# Calling up DNN / Declaration Optimizer.
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

# Define Train function and Test function to validate.
def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(log_interval, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() 
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format
        (test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# Train and Test the model and save it.
for epoch in range(1, 11):
    train(log_interval, model, device, train_loader, optimizer, epoch)
    test(log_interval, model, device, test_loader)

torch.save(model, './model.pt')
torch.save(model.state_dict(), './model.pt')

# Final Result: Test set: Average loss: 0.0337, Accuracy: 9889/10000 (99%), epoch=10

'''
Train Epoch: 1 [0/60000 (0%)]   Loss: 2.301216
Train Epoch: 1 [12800/60000 (21%)]      Loss: 0.380214
Train Epoch: 1 [25600/60000 (43%)]      Loss: 0.357060
Train Epoch: 1 [38400/60000 (64%)]      Loss: 0.478809
Train Epoch: 1 [51200/60000 (85%)]      Loss: 0.170317

Test set: Average loss: 0.1658, Accuracy: 9460/10000 (95%)

Train Epoch: 2 [0/60000 (0%)]   Loss: 0.268325
Train Epoch: 2 [12800/60000 (21%)]      Loss: 0.154823
Train Epoch: 2 [25600/60000 (43%)]      Loss: 0.249963
Train Epoch: 2 [38400/60000 (64%)]      Loss: 0.120705
Train Epoch: 2 [51200/60000 (85%)]      Loss: 0.173003

Test set: Average loss: 0.0894, Accuracy: 9720/10000 (97%)

Train Epoch: 3 [0/60000 (0%)]   Loss: 0.154783
Train Epoch: 3 [12800/60000 (21%)]      Loss: 0.088348
Train Epoch: 3 [25600/60000 (43%)]      Loss: 0.109618
Train Epoch: 3 [38400/60000 (64%)]      Loss: 0.118304
Train Epoch: 3 [51200/60000 (85%)]      Loss: 0.102374

Test set: Average loss: 0.0656, Accuracy: 9799/10000 (98%)

Train Epoch: 4 [0/60000 (0%)]   Loss: 0.077867
Train Epoch: 4 [12800/60000 (21%)]      Loss: 0.034911
Train Epoch: 4 [25600/60000 (43%)]      Loss: 0.081153
Train Epoch: 4 [38400/60000 (64%)]      Loss: 0.080394
Train Epoch: 4 [51200/60000 (85%)]      Loss: 0.133417

Test set: Average loss: 0.0539, Accuracy: 9819/10000 (98%)

Train Epoch: 5 [0/60000 (0%)]   Loss: 0.117520
Train Epoch: 5 [12800/60000 (21%)]      Loss: 0.135022
Train Epoch: 5 [25600/60000 (43%)]      Loss: 0.044997
Train Epoch: 5 [38400/60000 (64%)]      Loss: 0.255876
Train Epoch: 5 [51200/60000 (85%)]      Loss: 0.052751

Test set: Average loss: 0.0457, Accuracy: 9842/10000 (98%)

Train Epoch: 6 [0/60000 (0%)]   Loss: 0.016124
Train Epoch: 6 [12800/60000 (21%)]      Loss: 0.111076
Train Epoch: 6 [25600/60000 (43%)]      Loss: 0.058309
Train Epoch: 6 [38400/60000 (64%)]      Loss: 0.056140
Train Epoch: 6 [51200/60000 (85%)]      Loss: 0.320479

Test set: Average loss: 0.0414, Accuracy: 9859/10000 (99%)

Train Epoch: 7 [0/60000 (0%)]   Loss: 0.232670
Train Epoch: 7 [12800/60000 (21%)]      Loss: 0.068792
Train Epoch: 7 [25600/60000 (43%)]      Loss: 0.077938
Train Epoch: 7 [38400/60000 (64%)]      Loss: 0.075234
Train Epoch: 7 [51200/60000 (85%)]      Loss: 0.054288

Test set: Average loss: 0.0395, Accuracy: 9873/10000 (99%)

Train Epoch: 8 [0/60000 (0%)]   Loss: 0.076282
Train Epoch: 8 [12800/60000 (21%)]      Loss: 0.204002
Train Epoch: 8 [25600/60000 (43%)]      Loss: 0.150735
Train Epoch: 8 [38400/60000 (64%)]      Loss: 0.036967
Train Epoch: 8 [51200/60000 (85%)]      Loss: 0.051248

Test set: Average loss: 0.0383, Accuracy: 9879/10000 (99%)

Train Epoch: 9 [0/60000 (0%)]   Loss: 0.062804
Train Epoch: 9 [12800/60000 (21%)]      Loss: 0.012132
Train Epoch: 9 [25600/60000 (43%)]      Loss: 0.025932
Train Epoch: 9 [38400/60000 (64%)]      Loss: 0.048140
Train Epoch: 9 [51200/60000 (85%)]      Loss: 0.060565

Test set: Average loss: 0.0362, Accuracy: 9882/10000 (99%)

Train Epoch: 10 [0/60000 (0%)]  Loss: 0.038839
Train Epoch: 10 [12800/60000 (21%)]     Loss: 0.036048
Train Epoch: 10 [25600/60000 (43%)]     Loss: 0.021334
Train Epoch: 10 [38400/60000 (64%)]     Loss: 0.049147
Train Epoch: 10 [51200/60000 (85%)]     Loss: 0.062746

Test set: Average loss: 0.0337, Accuracy: 9889/10000 (99%)
'''