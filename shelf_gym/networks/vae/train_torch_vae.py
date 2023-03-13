import sys, os
default_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(default_path + '../../utils/')
sys.path.append(default_path +'models/')
from torch_model_vae import VariationalAutoencoder, CustomDataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

# 32-d latent space, parameter count in same order of magnitude
# as in the original VAE paper (VAE paper has about 3x as many)
width = 300
height = 400
latent_dims = 32
num_epochs = 150
batch_size = 128
learning_rate = 1e-4
use_gpu = True
beta = 1
train_data_path = '../train_data/combined_data_300_400_30718_no_prep.h5'
test_data_path = '../train_data/combined_data_300_400_1530.h5'
model_name = "vae_vpp_real_and_sim_300_400_150_ep_1_no_prep_beta1"
Log_writer = SummaryWriter(log_dir="../tensorboard_logs/" + model_name)
torch.set_num_threads(6)


def train_model(train_dataloader, vae, optimizer, device):
    train_loss_avg = 0
    kld_loss_avg = 0
    loglike_loss_avg = 0
    num_batches = 0
    vae.train()

    for image_batch in train_dataloader:
        image_batch = image_batch.to(device)

        # vae reconstruction
        _, _, loss, loglike_loss, kdl_loss = vae(image_batch)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()

        # one step of the optmizer (using the gradients from backpropagation)
        optimizer.step()

        train_loss_avg += loss.item()
        kld_loss_avg += kdl_loss.item()
        loglike_loss_avg += loglike_loss.item()
        num_batches += 1

    train_loss_avg /= num_batches
    kld_loss_avg /= num_batches
    loglike_loss_avg /= num_batches
    Log_writer.add_scalar('train/loss', train_loss_avg, epoch)
    Log_writer.add_scalar('train/kld-loss', kld_loss_avg, epoch)
    Log_writer.add_scalar('train/loglike-loss', loglike_loss_avg, epoch)
    print('Training Epoch [%d / %d] average loss: %f, average kd_loss: %f with beta: %d, recon loss: %f' % (epoch + 1, num_epochs, train_loss_avg, kld_loss_avg, vae.beta, loglike_loss_avg))
    return


def test_model(test_dataloader, vae, device):
    test_loss_avg = 0
    kld_loss_avg = 0
    loglike_loss_avg = 0
    vae.eval()

    with torch.no_grad():
        image_batch = next(iter(test_dataloader))
        image_batch = image_batch.to(device)
        # vae reconstruction
        _, _, loss, loglike_loss, kdl_loss = vae(image_batch)

        test_loss_avg += loss.item()
        kld_loss_avg += kdl_loss.item()
        loglike_loss_avg += loglike_loss.item()

    Log_writer.add_scalar('train/loss', test_loss_avg, epoch)
    Log_writer.add_scalar('train/kld-loss', kld_loss_avg, epoch)
    Log_writer.add_scalar('train/loglike-loss', loglike_loss_avg, epoch)
    print('Testing Epoch [%d / %d] average reconstruction error: %f' % (epoch + 1, num_epochs, test_loss_avg))
    return


#initialize model
vae = VariationalAutoencoder(latent_dims, beta, width, height)
#set gpu device
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
vae = vae.to(device)
summary(vae, input_size=(1, 1, 300, 400))
#load dataset
train_dataset = CustomDataset(os.path.join(os.path.dirname(__file__), train_data_path), width, height)
test_dataset = CustomDataset(os.path.join(os.path.dirname(__file__), test_data_path), width, height)
print("loading dataset of size: ", train_dataset.__len__())
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#print number of trainable params
num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
print('Number of tunable parameters: %d' % num_params)
optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning_rate, weight_decay=1e-5)
# set to training mode
vae.train()

train_loss_avg = []

print('Training ...')
vae_save_path = os.path.join(os.path.dirname(__file__), '../saved_models/' + model_name)
os.makedirs(os.path.join(os.path.dirname(__file__), '../saved_models'), exist_ok=True)
for epoch in tqdm(range(num_epochs)):
    train_model(train_dataloader, vae, optimizer, device)
    #test_model(test_dataloader, vae, device)
    if epoch % 5 == 0:
        print("save intermediate model")
        torch.save(vae.state_dict(), vae_save_path)
# save model
torch.save(vae.state_dict(), vae_save_path)
