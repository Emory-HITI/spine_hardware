import numpy as np
import torch
from tqdm import tqdm
from datasets.spinal_hardware import SpinalHardwareDataset
from torchvision import transforms

# Function to get mean and standard deviation for normalization
def get_training_mean_std_bmav(dataset, **kwargs):
    size_val = kwargs.pop("size_val", 256)
    simple_transform = transforms.Compose([
            transforms.Resize((size_val, size_val)), # resize image
            transforms.ToTensor(),
        ])
    
    train_dataset = SpinalHardwareDataset(dataset, transform=simple_transform, **kwargs)
    
    
    means = []
    stds = []
    for i in tqdm(range(len(train_dataset))):
        image = train_dataset.__getitem__(i)[0][0].numpy()
        mean, std = np.mean(image), np.std(image)
        means.append(mean)
        stds.append(std)

    return np.mean(means), np.mean(stds)