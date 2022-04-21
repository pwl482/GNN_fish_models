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
    """
    A method to create a DataLoader from fish_models poses, actions and views data.

    Parameters
    ----------
    poses : numpy.ndarray
        array of positional agent information of shape (n_files, n_agents, n_timesteps, (x, y, orientation))
    actions : numpy.ndarray
        array of agent action information of shape (n_files, n_agents, n_timesteps, (speed, turn))
    views : numpy.ndarray
        array of agent view information in three channels of shape (n_files, n_agents, n_timesteps, view_width*3)
    batchsize : int, optional
        number of graphs to put into one batch
    filter_distance : int or float, optional
        the euclidean distance, where node edges with a L2 distance > filter_distance are kept out of the edge_index
    view_width : int, optional
        number of bins for each of the three visual channels ["fish", "fish_oris", "walls"] of 'views'

    Returns
    -------
    DataLoader : torch_geometric.loader.DataLoader
        torch dataloader, containing batched graph samples of the input dataset
    """
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
    """
    Train a pytorch model using the specified dataloaders for the specified number of epochs.

    Parameters
    ----------
    model : nn.Module
        pytorch model to be trained
    optimizer : torch.optim
        optimizer to train the mode with
    train_loader : torch_geometric.loader.DataLoader
        torch dataloader, containing batched graph samples of the training set
    val_loader : torch_geometric.loader.DataLoader
        torch dataloader, containing batched graph samples of the validation set
    criterion : function
        the loss function for training the model
    epochs : int, optional
        number of epochs to train for
    device : torch.device or string, optional
        use 'cpu' for cpu based models and 'cuda:0' for gpu based models

    Returns
    -------
    train_losses : list
        training loss values for each of the trained epochs
    val_losses : list
        validation loss values for each of the trained epochs
    epochs : int
        number of trained epochs
    """
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
    """
    Save a pytorch model and optimizer state dictionary to a checkpoint file.

    Parameters
    ----------
    model : nn.Module
        pytorch module to be saved
    optimizer : torch.optim
        optimizer with which the pytorch module was trained
    epoch : int
        number of previously trained epochs
    train_losses : list
        training loss values for each of the previously trained epochs
    val_losses : list
        validation loss values for each of the previously trained epochs
    path : string
        path to where to save the pytorch model checkpoint

    Returns
    -------
    None
    """
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            }, path)
    
def load_model(model, optimizer, path, map_location=torch.device('cuda:0')):
    """
    Load a pytorch model and optimizer state dictionary from a checkpoint file.
    The weights are loaded in place and the additional properties in the checkpoint are returned.

    Parameters
    ----------
    model : nn.Module
        pytorch module with layers fitting to the weights that should be loaded
    optimizer : torch.optim
        optimizer to train the pytorch module
    path : string
        path of the pytorch model checkpoint to load
    map_location : torch.device or string, optional
        use 'cpu' for cpu based models and 'cuda:0' for gpu based models

    Returns
    -------
    epoch : int
        number of previously trained epochs
    prev_train_losses : list
        training loss values for each of the previously trained epochs
    prev_val_losses : list
        validation loss values for each of the previously trained epochs
    """
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    prev_train_losses = checkpoint['train_losses']
    prev_val_losses = checkpoint['val_losses']
    return epoch, prev_train_losses, prev_val_losses


class Pose_and_View_SwarmNet(nn.Module):
    """
    A pytorch module variation of SwarmNet, a graph-convolutional neural network model for agent action prediction.

    Attributes
    ----------
    pos_channels : int
        number of arguments for the positional information of the agents, 3 for fish_models: (x, y, orientation)
    hidden_channels : int
        number of dimensions for the hidden linear layers after the graph-convolution layer
    out_channels : int
        number of floating point target values to predict, typically 1 for fish_models: (turn)
    conv_channels : int
        number of channels of the visual agent information, 3 for fish_models: ["fish", "fish_oris", "walls"]
    conv_width : int
        width of each of the channels of visual information, is the same for each of ["fish", "fish_oris", "walls"]
    n_linear_layers : int
        number of hidden linear layers after the graph convolutional layer

    Methods
    -------
    forward(pose, views, edge_index):
        Computes the model output for a given input of positions, views and graph edges.
    """
    def __init__(self, pos_channels, hidden_channels, out_channels, conv_channels, conv_width, n_linear_layers=3):
        """
        Constructs a Pose_and_View_SwarmNet model with randomly initialized weights.

        Parameters
        ----------
        pos_channels : int
            number of arguments for the positional information of the agents, 3 for fish_models: (x, y, orientation)
        hidden_channels : int
            number of dimensions for the hidden linear layers after the graph-convolution layer
        out_channels : int
            number of floating point target values to predict, typically 1 for fish_models: (turn)
        conv_channels : int
            number of channels of the visual agent information, 3 for fish_models: ["fish", "fish_oris", "walls"]
        conv_width : int
            width of each of the channels of visual information, is the same for each of ["fish", "fish_oris", "walls"]
        n_linear_layers : int
            number of hidden linear layers after the graph convolutional layer

        Returns
        -------
        None
        """
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
        """
        Computes the model output for a given input of positions, views and graph edges.
        Multiple graphs can be processed in one batch, the edge index separates the individual graphs.
        N is the number of nodes in each graph, but can be varied between graphs.
        
        Parameters
        ----------
        pose : torch.tensor
            Agent positions of shape (N * batch_size, pos_channels)
        views : torch.tensor
            Agent views of shape (N * batch_size, conv_channels, conv_width)
        edge_index : torch.tensor
            edges from node edge_index[0, i] to edge_index[1, i] of shape (2, N * batch_size * (N-1))

        Returns
        -------
        x : torch.tensor
            predicted targets for each node of shape (N * batch_size, out_channels)
        """
        views = self.lrelu(self.encoder1(views))
        views = self.lrelu(self.encoder2(views))
        x = self.lrelu(self.gconv1(torch.cat((pose, views.squeeze(1)), dim=1), edge_index))
        for linear_i in self.linears:
            x = self.dropout(self.lrelu(linear_i(x)))
        x = self.predictor(x)
        return x


class Edgeweight_SwarmNet(nn.Module):
    """
    A pytorch module variation of SwarmNet, a graph-convolutional neural network model for agent action prediction.
    This model has the addition of edge-weight prediction layers, compared to the 'Pose_and_View_SwarmNet' modlule.

    Attributes
    ----------
    pos_channels : int
        number of arguments for the positional information of the agents, 3 for fish_models: (x, y, orientation)
    hidden_channels : int
        number of dimensions for the hidden linear layers after the graph-convolution layer
    out_channels : int
        number of floating point target values to predict, typically 1 for fish_models: (turn)
    conv_channels : int
        number of channels of the visual agent information, 3 for fish_models: ["fish", "fish_oris", "walls"]
    conv_width : int
        width of each of the channels of visual information, is the same for each of ["fish", "fish_oris", "walls"]
    n_linear_layers : int
        number of hidden linear layers after the graph convolutional layer

    Methods
    -------
    get_edge_weights(self, pose, views, edge_index):
        Computes the edge weights the same way as in the models forward method and returns them for evaluation.
    forward(pose, views, edge_index):
        Computes the model output for a given input of positions, views and graph edges.
    """
    def __init__(self, pos_channels, hidden_channels, out_channels, conv_channels, conv_width, n_linear_layers=3):
        """
        Constructs a Edgeweight_SwarmNet model with randomly initialized weights.

        Parameters
        ----------
        pos_channels : int
            number of arguments for the positional information of the agents, 3 for fish_models: (x, y, orientation)
        hidden_channels : int
            number of dimensions for the hidden linear layers after the graph-convolution layer
        out_channels : int
            number of floating point target values to predict, typically 1 for fish_models: (turn)
        conv_channels : int
            number of channels of the visual agent information, 3 for fish_models: ["fish", "fish_oris", "walls"]
        conv_width : int
            width of each of the channels of visual information, is the same for each of ["fish", "fish_oris", "walls"]
        n_linear_layers : int
            number of hidden linear layers after the graph convolutional layer

        Returns
        -------
        None
        """
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
        """
        Computes the edge weights the same way as in the models forward method and returns them for evaluation.
        
        Parameters
        ----------
        pose : torch.tensor
            Agent positions of shape (N * batch_size, pos_channels)
        views : torch.tensor
            Agent views of shape (N * batch_size, conv_channels, conv_width)
        edge_index : torch.tensor
            edges from node edge_index[0, i] to edge_index[1, i] of shape (2, N * batch_size * (N-1))

        Returns
        -------
        edge_weights : torch.tensor
            predicted edge weights for each edge in edge_index of shape (N * batch_size * (N-1))
        """
        views = self.lrelu(self.encoder1(views))
        views = self.lrelu(self.encoder2(views))
        x_new = torch.cat((pose, views.squeeze(1)), dim=1)
        edge_weights = torch.mean(self.edge_weighter2(self.lrelu(self.edge_weighter1(x_new[edge_index]))), dim=0).squeeze()
        return edge_weights

    def forward(self, pose, views, edge_index):
        """
        Computes the model output for a given input of positions, views and graph edges.
        Multiple graphs can be processed in one batch, the edge index separates the individual graphs.
        N is the number of nodes in each graph, but can be varied between graphs.
        
        Parameters
        ----------
        pose : torch.tensor
            Agent positions of shape (N * batch_size, pos_channels)
        views : torch.tensor
            Agent views of shape (N * batch_size, conv_channels, conv_width)
        edge_index : torch.tensor
            edges from node edge_index[0, i] to edge_index[1, i] of shape (2, N * batch_size * (N-1))

        Returns
        -------
        x : torch.tensor
            predicted targets for each node of shape (N * batch_size, out_channels)
        """
        views = self.lrelu(self.encoder1(views))
        views = self.lrelu(self.encoder2(views))
        x_new = torch.cat((pose, views.squeeze(1)), dim=1)
        edge_weights = torch.mean(self.edge_weighter2(self.lrelu(self.edge_weighter1(x_new[edge_index]))), dim=0).squeeze()
        x = self.lrelu(self.gconv1(x_new, edge_index, torch.flatten(edge_weights)))
        for linear_i in self.linears:
            x = self.dropout(self.lrelu(linear_i(x)))
        x = self.predictor(x)
        return x

    
class CouzinModel(fish_models.AbstractModel):
    """
    A fish_models AbstractModel to produce predictions for the fish_models TrackGenerator.

    Attributes
    ----------
    model : nn.Module
        a pytorch SwarmNet model
    raycast : fish_models.Raycast
        a raycast object that produces the fish agent views
    view_width : int
        number of bins for each channel of the agent views
    max_n_fish : int
        maximal number of fish in the dataset to store actions/weights during model prediction
    save_edge_weights : bool, optional
        flag if the model can produce edge_weights and if they should be saved
    chosen_actions : list
        a list that contains for each agent a list of actions over all previously predicted timesteps 
    edge_weights : list
        a list that contains for each agent a list of edge_weights over all previously predicted timesteps 

    Methods
    -------
    get_edge_weights(self, pose, views, edge_index):
        Computes the edge weights the same way as in the models forward method and returns them for evaluation.
    forward(pose, views, edge_index):
        Computes the model output for a given input of positions, views and graph edges.
    """
    def __init__(self, model, raycast, view_width, max_n_fish=8, save_edge_weights=False):
        """
        Constructs a fish_models model shell for a pytorch prediction model.

        Parameters
        ----------
        model : nn.Module
            a pytorch SwarmNet model
        raycast : fish_models.Raycast
            a raycast object that produces the fish agent views
        view_width : int
            number of bins for each channel of the agent views
        max_n_fish : int, optional
            maximal number of fish in the dataset to store actions/weights during model prediction
        save_edge_weights : bool, optional
            flag if the model can produce edge_weights and if they should be saved

        Returns
        -------
        None
        """
        self.raycast = raycast
        self.model = model
        self.view_width = view_width
        self.save_edge_weights = save_edge_weights
        self.chosen_actions = [[] for i in range(max_n_fish)]
        self.edge_weights = [[] for i in range(max_n_fish)]
    
    def reset_chosen_actions():
        """
        Empties the list of previously predicted actions.
        """
        self.chosen_actions = [[] for i in range(max_n_fish)]
    
    def reset_edge_weights():
        """
        Empties the list of previously predicted edge_weights.
        """
        self.edge_weights = [[] for i in range(max_n_fish)]

    def choose_action(self, poses_3d, self_id):
        """
        Overwrites the fish_models.AbstractModel choose_action method to produce a (speed, turn) prediction
        from poses_3d and self_id.
        The speed is set to 8 for the original data and the model, while the turn is predicted by a pytorch model.

        Parameters
        ----------
        poses_3d : np.ndarray
            a array of shape (n_fish, 3)
        self_id : int
            a positional index for the n_fish dimension of 'poses_3d'

        Returns
        -------
        speed : float
            the output speed of the agent in cm/s, here constanly 8
        turn : float
            the output turn of the agent in radians, here predicted by the torch model
        """
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
        if self.save_edge_weights:
            self.edge_weights[self_id].append(self.model.get_edge_weights(x_pose, x_view, edge_index).detach().numpy())
        return speed, turn