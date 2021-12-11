import torch
from torch import nn
from tqdm import tqdm
import numpy as np

def evaluate_model(model, dataloaders, device, phase='test'):
    '''
    Function to evaluate on test set
    '''
    model.eval()   # Set model to evaluate mode        
    model.to(device)
    
    running_loss = 0.0
    running_corrects = 0
    running_incorrects = 0
    pred_list = []
    label_list = []
    output_list = []

    # Iterate over data.
    for inputs, labels in tqdm(dataloaders[phase]):
        inputs = list(map(lambda x: x.to(device), inputs))
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        # statistics
        running_corrects += torch.sum(preds == labels.data)
        running_incorrects += torch.sum(preds != labels.data)
        pred_list.extend(preds.cpu().tolist())
        output_list.extend(outputs.cpu().tolist())
        label_list.extend(labels.cpu().tolist())
    
    model.to("cpu")    
    print('Total Correct Predictions: ' + str(running_corrects))
    print('Total Incorrect Predictions: ' + str(running_incorrects))
    
    return np.array(pred_list), nn.Softmax(dim=-1)(torch.Tensor(output_list)).numpy(), np.array(label_list)