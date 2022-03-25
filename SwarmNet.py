import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv

import fish_models
import robofish.io


def create_swarm_dataset(poses, actions, views, batchsize=64, filter_distance=None, view_width=150):
    data_list = []
    for graph_pose, graph_action, graph_view in zip(poses, actions, views):

        graph_pose_nonna = graph_pose[(~np.isnan(graph_pose)).any(axis=-1).any(axis=-1)]
        graph_action_nonna = graph_action[(~np.isnan(graph_pose)).any(axis=-1).any(axis=-1)]
        graph_view_nonna = graph_view[(~np.isnan(graph_pose)).any(axis=-1).any(axis=-1)]

        graph_pose_nonna = np.swapaxes(graph_pose_nonna,0,1)
        graph_action_nonna = np.swapaxes(graph_action_nonna,0,1)
        graph_view_nonna = np.swapaxes(graph_view_nonna,0,1)

        num_nodes = graph_pose_nonna.shape[1]
        list2 = []
        list1 = np.repeat(np.arange(num_nodes), num_nodes-1)
        for i in range(num_nodes):
            list2.append(np.concatenate((np.arange(num_nodes)[:i], np.arange(num_nodes)[i+1:])))
        list2 = np.concatenate(list2)
        edge_index = torch.tensor(np.array([list1, list2]), dtype=torch.long)

        for pose, label, view in zip(graph_pose_nonna, graph_action_nonna, graph_view_nonna):
            if filter_distance is not None:
                filt = torch.arange(edge_index.shape[1]) == torch.arange(edge_index.shape[1])
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if i == j:
                            continue
                        if np.linalg.norm(pose[i,:2]-pose[j,:2]) > filter_distance:
                            if i > j:
                                filt = filt & (torch.arange(edge_index.shape[1])!=i*(num_nodes-1) + (j)) & (torch.arange(edge_index.shape[1])!=j*(num_nodes-1) + (i-1))
                            else:
                                filt = filt & (torch.arange(edge_index.shape[1])!=i*(num_nodes-1) + (j-1)) & (torch.arange(edge_index.shape[1])!=j*(num_nodes-1) + (i))
                filt_edge_index = edge_index[:,filt]
            x1 = torch.tensor(pose, dtype=torch.float)
            x2 = torch.tensor(np.stack((view[:,:view_width],
                              view[:,view_width:2*view_width],
                              view[:,2*view_width:3*view_width]), axis=1), dtype=torch.float)
            y = torch.tensor(label[:,1], dtype=torch.float)
            if filter_distance is not None:
                data = Data(pose=x1, view=x2 , edge_index=filt_edge_index, y=y, num_nodes=num_nodes)
            else:
                data = Data(pose=x1, view=x2 , edge_index=edge_index, y=y, num_nodes=num_nodes)
            data_list.append(data)
    return DataLoader(data_list, batch_size=batchsize)


def train_SwarmNet(model, optimizer, train_loader, val_loader, criterion, epochs=10, device="cpu"):
    train_losses = []
    val_losses = []
    pbar = tqdm(range(epochs), desc='Epoch:')
    for epoch in pbar:
        model.train()
        running_loss = 0
        for data in train_loader:
            data = data.to(device)
            true_lbl = data.y.reshape((-1,1))
            optimizer.zero_grad()
            pred_lbl = model(data.pose, data.view, data.edge_index)
            loss = criterion(pred_lbl, true_lbl)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))
        model.eval()
        running_loss2 = 0
        for data in val_loader:
            data = data.to(device)
            true_lbl = data.y.reshape((-1,1))
            pred_lbl = model(data.pose, data.view, data.edge_index)
            loss = criterion(pred_lbl, true_lbl)
            running_loss2 += loss.item()
        val_losses.append(running_loss2 / len(val_loader))
        pbar.set_description(f"[Train: {train_losses[-1]:.3f}][Val: {val_losses[-1]:.3f}]")
        #print(f'Epoch: {epoch:03d}, Train: {losses[-1]:.6f}, Val: {val_losses[-1]:.6f}    ', end="\r")
    return train_losses, val_losses, epochs


def save_model(model, optimizer, epoch, train_losses, val_losses, path):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            }, path)
    
def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    prev_train_losses = checkpoint['train_losses']
    prev_val_losses = checkpoint['val_losses']
    return epoch, prev_train_losses, prev_val_losses


class Pose_and_View_SwarmNet(nn.Module):
    def __init__(self, pos_channels, hidden_channels, out_channels, conv_channels, conv_width, n_linear_layers=3):
        super().__init__()
        torch.manual_seed(42)
        self.encoder1 = nn.Conv1d(conv_channels, conv_channels**2, kernel_size=3, stride=2)
        w1 = round((conv_width-3)/2+1)
        self.encoder2 = nn.Conv1d(conv_channels**2, 1, kernel_size=3, stride=2)
        w2 = round((w1-3)/2+1)
        self.gconv1 = GraphConv(pos_channels + w2, hidden_channels)
        self.linears = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels) for i in range(n_linear_layers)])
        self.predictor = nn.Linear(hidden_channels, out_channels)
        self.lrelu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, pose, views, edge_index):
        views = self.lrelu(self.encoder1(views))
        views = self.lrelu(self.encoder2(views))
        x = self.lrelu(self.gconv1(torch.cat((pose, views.squeeze(1)), dim=1), edge_index))
        for linear_i in self.linears:
            x = self.dropout(self.lrelu(linear_i(x)))
        x = self.predictor(x)
        return x


class Edgeweight_SwarmNet(nn.Module):
    def __init__(self, pos_channels, hidden_channels, out_channels, conv_channels, conv_width, n_linear_layers=3):
        super().__init__()
        torch.manual_seed(42)
        self.encoder1 = nn.Conv1d(conv_channels, conv_channels**2, kernel_size=3, stride=2)
        w1 = round((conv_width-3)/2+1)
        self.encoder2 = nn.Conv1d(conv_channels**2, 1, kernel_size=3, stride=2)
        w2 = round((w1-3)/2+1)
        self.edge_weighter1 = nn.Linear(pos_channels + w2, hidden_channels)
        self.edge_weighter2 = nn.Linear(hidden_channels, 1)
        self.gconv1 = GraphConv(pos_channels + w2, hidden_channels)
        self.linears = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels) for i in range(n_linear_layers)])
        self.predictor = nn.Linear(hidden_channels, out_channels)
        self.lrelu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.1)
    
    def get_edge_weights(self, pose, views, edge_index):
        views = self.lrelu(self.encoder1(views))
        views = self.lrelu(self.encoder2(views))
        x_new = torch.cat((pose, views.squeeze(1)), dim=1)
        edge_weights = self.edge_weighter2(self.lrelu(self.edge_weighter1(x_new))).squeeze()
        return edge_weights

    def forward(self, pose, views, edge_index):
        views = self.lrelu(self.encoder1(views))
        views = self.lrelu(self.encoder2(views))
        x_new = torch.cat((pose, views.squeeze(1)), dim=1)
        edge_weights = self.edge_weighter2(self.lrelu(self.edge_weighter1(x_new))).squeeze()
        x = self.lrelu(self.gconv1(x_new, edge_index, torch.flatten(edge_weights)))
        for linear_i in self.linears:
            x = self.dropout(self.lrelu(linear_i(x)))
        x = self.predictor(x)
        return x

    
class CouzinModel(fish_models.AbstractModel):
    def __init__(self, model, raycast, view_width, max_n_fish=8):
        self.raycast = raycast
        self.model = model
        self.view_width = view_width
        self.chosen_actions = [[] for i in range(max_n_fish)]
    
    def reset_chosen_actions():
        self.chosen_actions = [[] for i in range(max_n_fish)]

    def choose_action(self, poses_3d, self_id):
        speed = 8
        
        num_nodes = poses_3d.shape[0]
        list2 = []
        list1 = np.repeat(np.arange(num_nodes), num_nodes-1)
        for k in range(num_nodes):
            list2.append(np.concatenate((np.arange(num_nodes)[:k], np.arange(num_nodes)[k+1:])))
        list2 = np.concatenate(list2)
        edge_index = torch.tensor(np.array([list1, list2]), dtype=torch.long)
        x_pose = torch.tensor(poses_3d, dtype=torch.float)
        new_view = np.array([self.raycast.cast(poses_3d, new_id) for new_id in range(poses_3d.shape[0])])
        x_view = torch.tensor(np.stack((new_view.reshape(-1, new_view.shape[-1])[:,:self.view_width],
                              new_view.reshape(-1, new_view.shape[-1])[:,self.view_width:self.view_width*2],
                              new_view.reshape(-1, new_view.shape[-1])[:,self.view_width*2:self.view_width*3]), axis=1), 
                              dtype=torch.float)
        
        turn = self.model(x_pose, x_view, edge_index).detach().numpy()[self_id].item()
        self.chosen_actions[self_id].append(turn)
        return speed, turn