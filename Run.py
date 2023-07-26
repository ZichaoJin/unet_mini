import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import helper
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets, models
from torchsummary import summary
import pytorch_unet
from tools import get_from_path, calc_loss, print_metrics, save_loss, reverse_transform
import configargparse

#Config
parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', required=False,
                        is_config_file=True, help='config file')
parser.add_argument('--root_train_img', required=False, type=str, default="train_input/",
                        help='data root path')
parser.add_argument('--root_train_mask', required=False, type=str, default="train_target/",
                        help='data root path')
parser.add_argument('--root_test_img', required=False, type=str, default="test_input/",
                        help='data root path')
parser.add_argument('--root_test_mask', required=False, type=str, default="test_target/",
                        help='data root path')
parser.add_argument('--workers', required=False, type=int, default=0,
                        help='number of workers for data loader')
parser.add_argument('--bsize', required=False, type=int, default=25,
                        help='batch size')
parser.add_argument('--epochs', required=False, type=int, default=5,
                        help='number of epochs')
parser.add_argument('--lr', required=False, type=float, default=1e-4,
                        help='learning rate')
parser.add_argument('--drop_prob', required=False, type=float, default=0.1,
                        help='dropblock dropout probability')
parser.add_argument('--block_size', required=False, type=int, default=5,
                        help='dropblock block size')
parser.add_argument('--weight_decay', required=False, type=float, default=1e-5,
                        help='weight decay')
parser.add_argument('--scheduler_step_size', required=False, type=int, default=100,
                        help='scheduler step size')
parser.add_argument('--scheduler_gamma', required=False, type=float, default=0.5,
                        help='scheduler gamma')
options = parser.parse_args()

root_train_img = options.root_train_img
root_train_mask = options.root_train_mask
root_test_img = options.root_test_img
root_test_mask = options.root_test_mask
batch_size = options.bsize
num_workers = options.workers
epochs = options.epochs
lr = options.lr
drop_prob = options.drop_prob
block_size = options.block_size
weight_decay = options.weight_decay
step_size = options.scheduler_step_size
gamma = options.scheduler_gamma
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define dataset
class SimDataset(Dataset):
    def __init__(self, images_path, target_path, transform=None):
        self.input_images = get_from_path(images_path)
        self.target_masks = get_from_path(target_path)
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)

        return [image, mask]
    
# use the same transformations for train and val images
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
])

data = SimDataset(root_train_img,root_train_mask,transform = trans)
train_set,val_set = random_split(
    dataset = data,
    lengths = [800,200],
)
image_datasets = {
    'train': train_set, 'val': val_set
}

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers),
    'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
}

# Unet with 6 output channels, 3 input channels, 3 layers, and 16 output channels for the 1st layer
model = pytorch_unet.UNet(6,3,3,16)
model = model.to(device)
# print model
summary(model, input_size=(3, 224, 224))


def train_model(model, optimizer, scheduler, num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    train_loss_list = []
    val_loss_list = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            save_loss(metrics, epoch_samples, phase, train_loss_list, val_loss_list)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))
    
    x=np.linspace(0,len(train_loss_list),len(val_loss_list))
    plt.plot(x,train_loss_list,label="train_loss",linewidth=1.5)
    plt.plot(x,val_loss_list,label="val_loss",linewidth=1.5)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Train Model

model = pytorch_unet.UNet(6,3,3,16).to(device)

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)

model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=epochs)


#Test Model

model.eval()   # Set model to evaluate mode

test_dataset = SimDataset(root_test_img,root_test_mask,transform = trans)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False, num_workers=num_workers)
        
inputs, labels = next(iter(test_loader))
inputs = inputs.to(device)
labels = labels.to(device)

pred = model(inputs)
pred = pred.data.cpu().numpy()

# Print out  results
# Change channel-order and make 3 channels for matplot
input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]

# Map each channel (i.e. class) to each color
target_masks_rgb = [helper.masks_to_colorimg(x) for x in labels.cpu().numpy()]
pred_rgb = [helper.masks_to_colorimg(x) for x in pred]

helper.plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb])