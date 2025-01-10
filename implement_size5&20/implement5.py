#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import egg.core as core

from torchvision import datasets, transforms
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import wandb

wandb.init(project="egg", name="try5")



# For convenince and reproducibility, we set some EGG-level command line arguments here
opts = core.init(params=['--random_seed=7', # will initialize numpy, torch, and python RNGs
                         '--lr=1e-3',   # sets the learning rate for the selected optimizer 
                         '--batch_size=32',
                         '--optimizer=adam'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
print(f"device: {device}")





# prepare data
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json


datapath = 'data/guess_position_5.json'

map_ = (5, 5)
position1 = None
position2 = None
datasize = 10000
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
transform = transforms.ToTensor()

batch_size = opts.batch_size # set via the CL arguments above

def cover(x, y, map, value=0.5):
    if x < 0 or x >= map_[0] or y < 0 or y >= map_[1]:
        return 
    if map[x, y] == 1:
        return 
    map[x, y] = value

def check(x, y, map, visited, value=0.5):
    if x < 0 or x >= map_[0] or y < 0 or y >= map_[1]:
        return 1
    if visited[x, y] == 1:
        return 0
    visited[x, y] = 1
    if map[x, y] == 1:
        return check(x + 1, y, map, visited, value) + check(x - 1, y, map, visited, value) + check(x, y + 1, map, visited, value) + check(x, y - 1, map, visited, value) + check(x + 1, y + 1, map, visited, value) + check(x - 1, y - 1, map, visited, value) + check(x + 1, y - 1, map, visited, value) + check(x - 1, y + 1, map, visited, value)
    if map[x, y] == value:
        return 1
    return 0

def generate_data_epoch(value = 0.5):
    position1 = (np.random.randint(0, map_[0]), np.random.randint(0, map_[1]))
    position2 = (np.random.randint(0, map_[0]), np.random.randint(0, map_[1]))
    map = np.zeros(map_)
    map[position1] = 1
    map[position2] = 1
    ground_truth = np.zeros(map_)
    ground_truth[position1] = 1
    ground_truth[position2] = 1
    if np.random.rand() > 0.5:
        for i in range(-1, 2):
            for j in range(-1, 2):
                cover(position1[0] + i, position1[1] + j, map, value)
    if np.random.rand() > 0.5:
        for i in range(-1, 2):
            for j in range(-1, 2):
                cover(position2[0] + i, position2[1] + j, map, value)
    visited = np.zeros(map_)
    count = check(position1[0], position1[1], map, visited, value)
    if count >=8:
        ground_truth[position1] = 0
    visited = np.zeros(map_)
    count = check(position2[0], position2[1], map, visited, value)
    if count >=8:
        ground_truth[position2] = 0
        
    return map, ground_truth

def get_data(save_path, datasize = 10000, value = 0.5):
    save_path.replace('.json', f'_{datasize}.json')
    if os.path.exists(save_path):
        print(f"load data from {save_path}")
        with open(save_path, 'r') as f:
            data = json.load(f)
        train_, test_ = train_test_split(data, test_size=0.2)
        return train_, test_
    data = []
    
    for i in tqdm(range(datasize)):
        map, ground_truth = generate_data_epoch(value=value)
        data.append({
            'id': i,
            'map': map.tolist(),
            'ground_truth': ground_truth.tolist()
        })
    with open(save_path, 'w') as f:
        json.dump(data, f)
    print(f"save and load data from {save_path}")
    with open(save_path, 'r') as f:
        data = json.load(f)
    train_, test_ = train_test_split(data, test_size=0.2)
    return train_, test_




train_data, test_data = get_data(datapath, datasize)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        map = torch.tensor(self.data[idx]['map']).float()
        ground_truth = torch.tensor(self.data[idx]['ground_truth']).float()
        return map.view(1, 5, 5), ground_truth.view(1, 5, 5)

train_loader = torch.utils.data.DataLoader(Dataset(train_data), batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(Dataset(test_data), batch_size=batch_size, shuffle=False, **kwargs)





# visualize data
def visualize_data(inputlist: list, path=None):
    number = len(inputlist)
    input = np.array(inputlist[0])
    ground_truth_ = np.array(inputlist[1])
    if number > 2:
        predict_ = np.array(inputlist[2])
    if number > 3:
        message_ = np.array(inputlist[3])
    plt.subplot(1, number, 1)
    plt.imshow(input, cmap='gray')
    plt.title('input')
    plt.subplot(1, number, 2)
    plt.imshow(ground_truth_, cmap='gray')
    plt.title('ground_truth')
    if number > 2:
        plt.subplot(1, number, 3)
        plt.imshow(predict_, cmap='gray')
        plt.title('predict')
    if number > 3:
        plt.subplot(1, number, 4)
        plt.imshow(message_, cmap='gray')
        plt.title('message')
    
    if path:
        plt.savefig(path)

visualize_data_ = np.random.choice(train_data, 5)
for data in visualize_data_:
    visualize_data([data['map'], data['ground_truth']], path=f"visualize/visualize5/{data['id']}.png")
    

class Vision(nn.Module):
    def __init__(self):
        super(Vision, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 20, 3, 1, padding=1)  
        self.conv2 = nn.Conv2d(20, 50, 3, 1, padding=1) 
        self.fc1 = nn.Linear(5 * 5 * 50, 125)  

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 50 * 5 * 5)  
        x = F.relu(self.fc1(x))
        return x


class PretrainNet(nn.Module):
    def __init__(self, vision_module):
        super(PretrainNet, self).__init__()
        self.vision_module = vision_module
        self.fc = nn.Linear(125, 25)
        
    def forward(self, x):
        x = self.vision_module(x)
        x = self.fc(F.leaky_relu(x))
        x = F.softmax(x, dim=1)
        return x





vision = Vision()
class_prediction = PretrainNet(vision) #  note that we pass vision - which we want to pretrain
optimizer = core.build_optimizer(class_prediction.parameters()) #  uses command-line parameters we passed to core.init
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
class_prediction = class_prediction.to(device)



if os.path.exists('pretrain5.pth'):
    class_prediction.load_state_dict(torch.load('pretrain5.pth'))
if os.path.exists('vision5.pth'):
    vision.load_state_dict(torch.load('vision5.pth'))



from tqdm import tqdm

if not os.path.exists('pretrain5.pth'):
    pbar = tqdm(range(50), desc="Training")
    for epoch in pbar:
        mean_loss, n_batches = 0, 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = class_prediction(data)
            
            
            target = target.view(-1, 25)
            loss = F.binary_cross_entropy(output, target)
            
            loss.backward()
            optimizer.step()
            
            mean_loss += loss.mean().item()
            n_batches += 1
        
        mean_loss /= n_batches
        pbar.set_description(f"Training loss: {mean_loss:.10f}")
        wandb.log({"epoch": epoch, "loss": mean_loss})
        scheduler.step()
        
    # 保存模型
    torch.save(class_prediction.state_dict(), 'pretrain5.pth')
    # 保存vision
    torch.save(vision.state_dict(), 'vision5.pth')



# Evaluate the pretraining
class_prediction.eval()
mean_loss, n_batches = 0, 0
outputs = []
for batch_idx, (data, target) in enumerate(test_loader):
    data, target = data.to(device), target.to(device)
    output = class_prediction(data)
    outputs.append(output)
    target = target.view(-1, 25)
    loss = F.binary_cross_entropy(output, target)
    mean_loss += loss.mean().item()
    n_batches += 1
mean_loss /= n_batches
print(f"Test loss: {mean_loss:.10f}")

wandb.log({"test_loss": mean_loss})

outputs = torch.cat(outputs, dim=0)
pca = PCA(n_components=2)
outputs_pca = pca.fit_transform(outputs.cpu().detach().numpy())
plt.figure(figsize=(10, 10))
plt.scatter(outputs_pca[:, 0], outputs_pca[:, 1], alpha=0.5)
plt.title("PCA of the pretrained model 5x5")
plt.savefig("pca_pretrain_5.png")


# visualize the prediction
class_prediction.eval()
data, target = next(iter(test_loader))
data, target = data.to(device), target.to(device)
output = class_prediction(data)

output = output.view(-1, 5, 5)
target = target.view(-1, 5, 5)
for i in range(10):
    visualize_data([data[i].cpu().detach().numpy().squeeze(), target[i].cpu().detach().numpy().squeeze(), output[i].cpu().detach().numpy().squeeze()], path=f"pretrain/pretrain_{i}.png")



class Sender(nn.Module):
    def __init__(self, vision, output_size):
        super(Sender, self).__init__()
        self.fc = nn.Linear(125, output_size)
        self.vision = vision
        
    def forward(self, x, aux_input=None):
        x = self.vision(x)
        x = self.fc(x)
        return x
    
    
class Receiver(nn.Module):
    def __init__(self, input_size):
        super(Receiver, self).__init__()
        self.fc = nn.Linear(input_size, 25)

    def forward(self, channel_input, receiver_input=None, aux_input=None):
        x = self.fc(channel_input)
        x = F.leaky_relu(x)
        x = F.softmax(x, dim=1)
        return x




def loss(sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input=None):
       
    labels = _labels.view(-1, 25)
    
    loss = F.binary_cross_entropy(receiver_output,  labels, reduction='none').mean(dim=1)
    
    return loss, {}


vocab_size = 100


sender = Sender(vision, vocab_size)
sender = core.GumbelSoftmaxWrapper(sender, temperature=1.0) # wrapping into a GS interface, requires GS temperature
receiver = Receiver(input_size=100)
receiver = core.SymbolReceiverWrapper(receiver, vocab_size, agent_input_size=100)
game = core.SymbolGameGS(sender, receiver, loss)




if os.path.exists('game5.pth'):
    game.load_state_dict(torch.load('game5.pth'))
else:
    game = core.SymbolGameGS(sender, receiver, loss)
    optimizer = torch.optim.Adam(game.parameters())
    optimizer_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
    trainer = core.Trainer(
        game=game, optimizer=optimizer, train_data=train_loader, optimizer_scheduler=optimizer_scheduler, 
        validation_data=test_loader, callbacks=[core.TemperatureUpdater(agent=sender, decay=0.9, minimum=0.1)],device=device    
    )


    
    n_epochs = 50
    trainer.train(n_epochs)

    # 保存模型
    torch.save(game.state_dict(), 'game5.pth')


game = game.to(device)
game.eval()

cnt = 0
messages = []

for z in test_loader:
    data, target = z
    data, target = data.to(device), target.to(device)

    with torch.no_grad():
        message = game.sender(data)
        
        output = game.receiver(message)


    messages.append(message)

    output = output.view(-1, 5, 5)
    target = target.view(-1, 5, 5)
    message = message.view(-1, 10, 10)

    if cnt < 10:
        

        visualize_data([data[0].cpu().detach().numpy().squeeze(), target[0].cpu().detach().numpy().squeeze(), output[3].cpu().detach().numpy().squeeze(), message[0].cpu().detach().numpy().squeeze()], path=f"symbol_5/symbol_{cnt}.png")

        cnt += 1

wandb.finish()

pca = PCA(n_components=2)
messages_pca = pca.fit_transform(messages.cpu().numpy())

plt.figure(figsize=(10, 10))
plt.scatter(messages_pca[:, 0], messages_pca[:, 1], alpha=0.5)
plt.title("PCA of the messages")
plt.savefig("pca_5.png")

