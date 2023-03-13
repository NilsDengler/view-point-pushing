import torch
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import push_predictions
import os
import numpy as np
from utils import load_checkpoint, save_checkpoint, get_loaders, check_accuracy

learning_rate = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
num_epochs = 101
num_workers = 4
image_height = 400
image_width = 400
pin_memory = True
load_model = False
load_model_path = 'saved_models/my_checkpoint_9_on_vam_3k_bin.pth.tar'
save_model_path = 'saved_models/'
train_img_dir = './labels/transformed_images_2k.h5'
train_mask_dir = './labels/labels_2k.csv'

def train(loader, model, optimizer, loss_fn, scaler):
    loss_avg = 0
    num_batches = 0
    model.train()
    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(device=device)
        targets = targets.float().to(device=device)
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_avg += loss.item()
        num_batches += 1
    return loss_avg/num_batches
        #loop.set_postfix(loss=loss.item())

def main():
    train_transform = A.Compose([
         A.Resize(height=image_height, width=image_width),
         A.Normalize(
             mean=[0.0],
             std=[1.0],
             max_pixel_value=1.0
         ),
         ToTensorV2()
     ])
    model = push_predictions(image_width, image_height).to(device)
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = get_loaders(train_img_dir, train_mask_dir, batch_size, train_transform, num_workers, pin_memory)

    if load_model:
        load_checkpoint(torch.load(load_model_path), model)

    scaler = torch.cuda.amp.GradScaler()
    loop = tqdm(range(num_epochs))
    for epoch in loop:
        loss = train(train_loader, model, optimizer, loss_fn, scaler)
        loop.set_postfix(loss=loss)
        checkpoint = {
            "state_dict":model.state_dict(), 
            "optimizer":optimizer.state_dict(),
        }
        if (epoch+1) % 10 == 0:
            save_checkpoint(checkpoint, save_model_path + 'my_checkpoint_' + str(epoch) + '_on_sic_2k_bin.pth.tar')

        check_accuracy(train_loader, model, device=device)


if __name__ == "__main__":
    main()