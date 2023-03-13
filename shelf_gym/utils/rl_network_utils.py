import torch
import numpy as np

def setup_encoder(latent_dims, beta, use_gpu, model_path):
    model = VariationalAutoencoder(latent_dims, beta)
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, device

def encode_latent_space(image, model, device):
    depth_image = torch.from_numpy(np.asarray(image).reshape((1, 1,252,252))).float()
    depth_image = depth_image.to(device)
    latent, _ = model.encoder(depth_image)
    latent = latent.cpu().detach().numpy().reshape(32)
    return latent

def preprocess_images_torch(image):
    unify = image.reshape(1,image.shape[0],image.shape[1])
    current_min, current_max = np.amin(unify), np.amax(unify)
    if current_min == current_max:
        return torch.from_numpy((unify*0).reshape(1, 1, image.shape[0], image.shape[1])).float()
    normed_min, normed_max = 0, 1
    x_normed = (unify - current_min) / (current_max - current_min)
    x_normed = x_normed * (normed_max - normed_min) + normed_min
    return torch.from_numpy(x_normed.reshape(1, 1, image.shape[0], image.shape[1])).float()

def preprocess_images_np(image):
    unify = image.reshape(1, image.shape[0], image.shape[1])
    current_min, current_max = np.amin(unify), np.amax(unify)
    if current_min == current_max:
        return (unify*0).astype(np.float16)
    normed_min, normed_max = 0, 1
    x_normed = (unify - current_min) / (current_max - current_min)
    x_normed = x_normed * (normed_max - normed_min) + normed_min
    return x_normed.astype(np.float16)

