import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import push_predictions
from utils import load_checkpoint, get_loaders, check_accuracy

learning_rate = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
num_epochs = 101
num_workers = 4
image_height = 400
image_width = 400
pin_memory = True
load_model = False
load_model_path = 'saved_models/my_checkpoint_49_on_vam_and_sic_5k_bin.pth.tar'
save_model_path = 'saved_models/'
train_img_dir = './testing/transformed_images_testing_fixed.h5'
train_mask_dir = './testing/labels_testing.csv'

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

    train_loader = get_loaders(train_img_dir, train_mask_dir, batch_size, train_transform, num_workers, pin_memory)

    load_checkpoint(torch.load(load_model_path), model)

    check_accuracy(train_loader, model, device=device)


if __name__ == "__main__":
    main()