# Fish Swarm Movement Prediction using GNNs
This repository was made for the research internship at the Berlin Biorobotics Lab at FU Berlin (October 2021-April 2022).
It contains multiple jupyter notebooks and a main python file explained below.

# Quickstart Guide
## Installation
Start by setting up a conda virtual environment in the console (tested with python 3.6, but other versions should also be viable):
```console
conda create -n torch_fish_env python=3.6 numpy pandas matplotlib seaborn tqdm pytorch torchvision torchaudio cudatoolkit=11.3 pyg -c anaconda -c conda-forge -c pytorch -c pyg
```
Activate the environment:
```console
conda activate torch_fish_env
```
To make the environment available to jupyter-notebooks/jupyter-lab:
```console
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=torch_fish_env
```
Install fish_models and robofish-io:
```console
pip config set global.extra-index-url https://git.imp.fu-berlin.de/api/v4/projects/6392/packages/pypi/simple
pip install fish_models
pip install robofish-io
```
If the installation of fish_models does not work this way, you could also try cloning the corresponding git repository:
```console
pip config set global.extra-index-url https://git.imp.fu-berlin.de/api/v4/projects/6392/packages/pypi/simple
git clone https://git.imp.fu-berlin.de/bioroboticslab/robofish/fish_models.git
cd fish_models
pip install -e .
```
## Load a fish_models dataset
To download a fish_models dataset with 2 fish agents you run in python (more data is available [here](https://userpage.fu-berlin.de/andigerken/model_server) under raw_data):
```python
import fish_models
data_path = fish_models.raw_data('pascal_vanilla_couzin')
```
Then you import the training and validation set for example with the following commands (choose max_files according to RAM limitations):
```python
raycast = fish_models.Raycast(
            n_fish_bins=150,
            n_wall_raycasts=150,
            fov_angle_fish_bins=2*np.pi,
            fov_angle_wall_raycasts=2*np.pi,
            world_bounds=([-50, -50], [50, 50]),
            view_of = ["fish", "fish_oris", "walls"]
        )
data_folder = data_path + r"\train"
dset_train = fish_models.IoDataset(data_folder, raycast, output_strings=["poses", "actions", "views"], max_files=500)

data_folder = data_path + r"\validation"
dset_val = fish_models.IoDataset(data_folder, raycast, output_strings=["poses", "actions", "views"], max_files=100)
```
## Train a SwarmNet model
To build the pytorch training and validation set use:
```python
from SwarmNet import create_swarm_dataset, train_SwarmNet, save_model, load_model, Pose_and_View_SwarmNet, CouzinModel, Edgeweight_SwarmNet
BATCH_SIZE = 64
train_loader = create_swarm_dataset(dset_train["poses"], dset_train["actions"], dset_train["views"], batchsize=BATCH_SIZE, filter_distance=None)
val_loader = create_swarm_dataset(dset_val["poses"], dset_val["actions"], dset_val["views"], batchsize=BATCH_SIZE, filter_distance=None)
```
Clear up some RAM:
```python
del dset_train
del dset_val
```
Train and save a SwarmNet model:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss()
PATH = f"your_model_name.pth"
model = Pose_and_View_SwarmNet(pos_channels=3, hidden_channels=128, out_channels=1, 
                               conv_channels=3, conv_width=150, n_linear_layers=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
train_losses, val_losses, epochs = train_SwarmNet(model, optimizer, train_loader, val_loader,
                                                  criterion, epochs=200, device=device)
save_model(model=model, optimizer=optimizer, epoch=epochs, train_losses=train_losses, val_losses=val_losses, path=PATH)
```
## Predict a Track using the SwarmNet model
Load a model again:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATH = f"your_model_name.pth"
model = Pose_and_View_SwarmNet(pos_channels=3, hidden_channels=128, out_channels=1, 
                               conv_channels=3, conv_width=150, n_linear_layers=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
prev_epochs, prev_train_losses, prev_val_losses = load_model(model, optimizer, PATH)
```
Predict a track:
```python
model.eval()
model.to("cpu")
raymodel = CouzinModel(model=model, raycast=raycast, view_width=150)
generator = fish_models.TrackGenerator([raymodel], world_size=[100,100], frequency=10)
track = generator.create_track(n_guppies=2, trackset_len=199)
```
Visualize the track:
```python
with generator.as_io_file(track) as f:
  f.plot(lw_distances=True)
plt.title("generated")
plt.show()
```
Generate and evaluate many tracks vs the test data
```python
import robofish.evaluate.evaluate as robo_eval
model_path = r"path\to\save\model\tracks"
model_label = "Model Data"
test_path = data_path + r"\test"
test_label = "Test Data"
for i in range(50):
    track = generator.create_track(n_guppies=2, trackset_len=199)
    filename = model_path + r"\track_" + str(i+1) + ".hdf5"
    f = generator.as_io_file(track, filename)
    f.close() 
robo_eval.evaluate_distance_to_wall([test_path, model_path], [test_label, model_label])
plt.show()
robo_eval.evaluate_tank_position([test_path, model_path], [test_label, model_label])
plt.show()
robo_eval.evaluate_relative_orientation([test_path, model_path], [test_label, model_label])
plt.show()
robo_eval.evaluate_speed([test_path, model_path], [test_label, model_label])
plt.show()
robo_eval.evaluate_turn([test_path, model_path], [test_label, model_label])
plt.show()
robo_eval.evaluate_orientation([test_path, model_path], [test_label, model_label])
plt.show()
robo_eval.evaluate_follow_iid([test_path, model_path], [test_label, model_label])
plt.show()
```

# GNN_fish_models Repository Exploration
## SwarmNet finalized implementations:
- [SwarmNet.py](https://github.com/pwl482/GNN_fish_models/blob/main/SwarmNet.py)
## Explorations for edge filtering and edge weight predictions:
### 2 fish:
- [SwarmNet_eval_less_connections.ipynb](https://github.com/pwl482/GNN_fish_models/blob/main/SwarmNet_eval_less_connections.ipynb)
### 6 fish:
- [SwarmNet_eval_all_6fish.ipynb](https://github.com/pwl482/GNN_fish_models/blob/main/SwarmNet_eval_all_6fish.ipynb)

# Old exploratory files:
## basic exploration of fish model features:
- [toying_with_fish_models.ipynb](https://github.com/pwl482/GNN_fish_models/blob/main/toying_with_fish_models.ipynb)
## first models for fish data:
- [torch_fish_models.ipynb](https://github.com/pwl482/GNN_fish_models/blob/main/torch_fish_models.ipynb)
- [fish_efficientnet.py](https://github.com/pwl482/GNN_fish_models/blob/main/fish_efficientnet.py)
## trying out torch geometric features and GNN explainer:
- [Torch_Geometric_Example.ipynb](https://github.com/pwl482/GNN_fish_models/blob/main/Torch_Geometric_Example.ipynb)
- [Graph_Classification.ipynb](https://github.com/pwl482/GNN_fish_models/blob/main/Graph_Classification.ipynb)
## SwarmNet implementation:
- [SwarmNet.ipynb](https://github.com/pwl482/GNN_fish_models/blob/main/SwarmNet.ipynb)
## SwarmNet implementations with Evaluations:
- 2 fish: [SwarmNet_more_experiments.ipynb](https://github.com/pwl482/GNN_fish_models/blob/main/SwarmNet_more_experiments.ipynb)
- 4 fish: [SwarmNet_4_fish.ipynb](https://github.com/pwl482/GNN_fish_models/blob/main/SwarmNet_4_fish.ipynb)
- [SwarmNet_Edge_Prediction.ipynb](https://github.com/pwl482/GNN_fish_models/blob/main/SwarmNet_Edge_Prediction.ipynb)
## Using Views together with Positions:
- with original couzin data: [SwarmNet_Positions_plus_Raycasts.ipynb](https://github.com/pwl482/GNN_fish_models/blob/main/SwarmNet_Positions_plus_Raycasts.ipynb)
- with Pascals smoother couzin data: [SwarmNet_Positions_plus_Raycasts_smooth_data.ipynb](https://github.com/pwl482/GNN_fish_models/blob/main/SwarmNet_Positions_plus_Raycasts_smooth_data.ipynb)
