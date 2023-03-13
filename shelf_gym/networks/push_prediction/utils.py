import torch
import torchvision
from dataset import PushDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print('=>saving checkpoint')
    torch.save(state, filename)

def load_checkpoint(checkpoint,model):
    print('=>loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])

def  get_loaders(train_img_dir, train_mask_dir, batch_size, train_transform, num_workers=4, pin_memory=True):
    train_ds = PushDataset(image_dir=train_img_dir, mask_dir=train_mask_dir, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)

    return train_loader

def check_accuracy(loader, model, device='cuda'):
    num_correct = 0
    num_cases = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.float().to(device)
            x_pred_tag = torch.round(torch.sigmoid(model(x)))
            correct_results_sum = (x_pred_tag == y).sum().float()
            num_correct += correct_results_sum
            num_cases += x_pred_tag.shape[0]
    acc = torch.true_divide(num_correct, num_cases)
    print(f'{num_correct}/{num_cases} with acc {acc:.4f}')

