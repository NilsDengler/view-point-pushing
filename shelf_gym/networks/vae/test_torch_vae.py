import sys, os
default_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(default_path + '../../utils/')
sys.path.append(default_path +'models/')
from torch_model_vae import VariationalAutoencoder, CustomDataset
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
import matplotlib.pyplot as plt

def test_model(test_dataloader, vae, device):
    with torch.no_grad():
        image_batch = next(iter(test_dataloader))
        image_batch = image_batch.to(device)
        # vae reconstruction
        recon, _, loss, loglike_loss, kdl_loss = vae(image_batch)
    return image_batch, recon

# 32-d latent space, parameter count in same order of magnitude
# as in the original VAE paper (VAE paper has about 3x as many)
latent_dims = 32
num_epochs = 100
batch_size = 64
learning_rate = 1e-4
variational_beta = 1
use_gpu = True
beta = 0.5
torch.set_num_threads(2)
width = 300
height = 400

#load saved model
vae = VariationalAutoencoder(latent_dims, beta, width, height)
saved_model = os.path.join(os.path.dirname(__file__), '../saved_models/vae_vpp_real_and_sim_300_400_120_ep_4')
vae.load_state_dict(torch.load(saved_model))
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
vae = vae.to(device)
summary(vae, input_size=(1, 1, width, height))
vae.eval()

#load dataset
test_dataset = CustomDataset(os.path.join(os.path.dirname(__file__), '../train_data/combined_data_first_try.h5'), width, height)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

images, images_rec = test_model(test_dataloader, vae, device)
for i in range(0, batch_size-1):
    image = images_rec[i]
    orig = images[i]
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(image.cpu().detach().numpy().reshape(width, height))
    axarr[1].imshow(orig.cpu().detach().numpy().reshape(width, height))
    plt.show()
