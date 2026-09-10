import torch 
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

transform = transforms.ToTensor()
train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)

data_loader_train = torch.utils.data.DataLoader(dataset=train_data,batch_size=64,shuffle=True)
data_loader_test = torch.utils.data.DataLoader(dataset=test_data,batch_size=64)

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(400, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
device = torch.device('cuda')
model = MyCNN().to(device)    
loss_init = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
count =0

 
train_losses = []
test_accuracies = []
for images, labels in data_loader_train:
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    output= model(images)
    loss =loss_init(output,labels)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    print(f'{count},:{loss}')
count = count+1
counts =0

for images, labels in data_loader_test:
    images, labels = images.to(device), labels.to(device)
    output = model(images)
    predictions=torch.argmax(output, dim=1)
    accuracy = (predictions == labels).float().mean()
    test_accuracies.append(accuracy.cpu())
    counts =counts+1
    print(f'{counts},{accuracy}')
    
torch.save(model.state_dict(),'2layernetwork.pth')
np.save('2layer_train_losses.npy', np.array(train_losses))
np.save('2layer_test_accuracies.npy', np.array(test_accuracies))


