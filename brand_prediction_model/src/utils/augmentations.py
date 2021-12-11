from torchvision import transforms

def apply_augmentations_with_norm(mode, size_val, contrast_val, hue_val, grayscale_prob, hflip_prob, degree_val, shear_val, avg_pop_mean, avg_pop_std):
    
    '''
    This function performs augmentation operations on input image data.
    Inputs:
        mode:       string value from the following list: 'train','val', 'test',
        size_val:       dimensions to resize the image to, default value: 256,
        contrast_val:   value to adjust the contrast with (ideally less than +/- 3),
        hue_val:        value to adjust the hue with (ideally less than +/- 3),
        grayscale_prob: probability of applying grayscale (values must be between 0 and 1),
        hflip_prob:     probability of horizontally flipping images 9values must be between 0 and 1),
        degree_val:     degree to rotate the image by (default 0),
        shear_val:      shearing factor to be applied to the image,
        avg_pop_mean:   average mean,
        avg_pop_std:    average standard deviation
    Outputs:
        Various transforms chained together 
    '''
    
    #If data set is training or validation set, apply advanced augmentations
    if mode == 'train':
        aug_transform = transforms.Compose([
            transforms.Resize((size_val, size_val)), # resize image
            transforms.ColorJitter(contrast = contrast_val, hue = hue_val), #Change image contrast and hue
#             transforms.RandomGrayscale(p = grayscale_prob),  #Randomly covert image to grayscale
            transforms.RandomHorizontalFlip(p=hflip_prob), # Randomly flip images across horizontal axis
            transforms.RandomAffine(degrees = degree_val, shear = shear_val), # Random affine transformations
            transforms.ToTensor(), #Convert PIL Image to tensor
            transforms.Normalize([avg_pop_mean], [avg_pop_std]) # normalize images
        ])
    
    
    
    #If data set is testing or validation set, apply basic transformations ie resize, convert to tensor and normalize
    else:
        aug_transform = transforms.Compose([
            transforms.Resize((size_val, size_val)), # resize image to 256x256
            transforms.ToTensor(),
            transforms.Normalize([avg_pop_mean], [avg_pop_std]) # normalize images
        ])
    
    return aug_transform