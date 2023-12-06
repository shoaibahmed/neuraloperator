"""
Training a SFNO on the spherical Shallow Water equations
==========================================================

In this example, we demonstrate how to use the small Spherical Shallow Water Equations example we ship with the package
to train a Spherical Fourier-Neural Operator
"""

# %%
# 


import torch
import matplotlib.pyplot as plt
import os
import sys
from neuralop.models import SFNO
from neuralop import Trainer
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss

from modulus.datapipes.climate import ERA5HDF5Datapipe

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DataProcessor:
    """Needed for the trainer class"""
    def preprocess(self, sample):
        assert isinstance(sample, list)
        assert len(sample) == 1, len(sample)
        sample = sample[0]  # weird DALI index for GPU
        if len(sample["outvar"].shape) == 5:  # assuming that batch dim is already added
            assert sample["outvar"].shape[1] == 1, sample["outvar"].shape
            sample["outvar"] = sample["outvar"].squeeze(dim=1)
        return {"x": sample["invar"], "y": sample["outvar"]}

    def postprocess(self, x, y):
        return x, y


# %%
# Loading the ERA5 dataset
dataset_path = "/lustre/fsw/nvresearch/ssiddiqui/era5_small/"
num_channels = 75
train_loader = ERA5HDF5Datapipe(
    data_dir=os.path.join(dataset_path, "train"),
    stats_dir=os.path.join(dataset_path, "stats"),
    channels=list(range(num_channels)),
    num_samples_per_year=None,
    batch_size=2,
    patch_size=(8, 8),
    num_workers=0,
    device=device,
    process_rank=0,
    world_size=1,
)
print(f"Loaded train loader of size {len(train_loader)}")

test_loader = ERA5HDF5Datapipe(
    data_dir=os.path.join(dataset_path, "test"),
    stats_dir=os.path.join(dataset_path, "stats"),
    channels=list(range(num_channels)),
    num_steps=8,
    num_samples_per_year=4,
    batch_size=1,
    patch_size=(8, 8),
    device=device,
    num_workers=0,
    shuffle=False,
)
print(f"Loaded test loader of size {len(test_loader)}")
test_loaders = {(720, 1440): test_loader}

# %%
# We create a tensorized FNO model

model = SFNO(n_modes=(256, 256), in_channels=num_channels, out_channels=num_channels,
             hidden_channels=64, projection_channels=64, factorization='dense')
model = model.to(device)

n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()


# %%
#Create the optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=8e-4,
                             weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)


# %%
# Creating the losses
l2loss = LpLoss(d=2, p=2, reduce_dims=(0,1))
# h1loss = H1Loss(d=2, reduce_dims=(0,1))

train_loss = l2loss
eval_losses={'l2': l2loss} #'h1': h1loss, 


# %%


print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()


# %% 
# Create the trainer
trainer = Trainer(model=model, n_epochs=20,
                  device=device,
                  data_processor=DataProcessor(),
                  wandb_log=False,
                  log_test_interval=3,
                  use_distributed=False,
                  verbose=True)


# %%
# Actually train the model on our small Darcy-Flow dataset

trainer.train(train_loader=train_loader,
              test_loaders=test_loaders,
              optimizer=optimizer,
              scheduler=scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses)


# %%
# Plot the prediction, and compare with the ground-truth 
# Note that we trained on a very small resolution for
# a very small number of epochs
# In practice, we would train at larger resolution, on many more samples.
# 
# However, for practicity, we created a minimal example that
# i) fits in just a few Mb of memory
# ii) can be trained quickly on CPU
#
# In practice we would train a Neural Operator on one or multiple GPUs

fig = plt.figure(figsize=(7, 7))
for index, resolution in enumerate([(720, 1440)]):
    test_samples = test_loaders[resolution].dataset
    data = test_samples[0]
    # Input x
    x = data['x']
    # Ground-truth
    y = data['y'][0, ...].numpy()
    # Model prediction
    x_in = x.unsqueeze(0).to(device)
    out = model(x_in).squeeze()[0, ...].detach().cpu().numpy()
    x = x[0, ...].detach().numpy()

    ax = fig.add_subplot(2, 3, index*3 + 1)
    ax.imshow(x)
    ax.set_title(f'Input x {resolution}')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(2, 3, index*3 + 2)
    ax.imshow(y)
    ax.set_title('Ground-truth y')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(2, 3, index*3 + 3)
    ax.imshow(out)
    ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()
plt.savefig("sfno_swe.png", dpi=300)
