import torch

def compute_weights(train_df, label_col_name):
    value_counts_ = train_df[label_col_name].value_counts().sort_index()
    weights = torch.Tensor(value_counts_.sum()/(len(value_counts_)*value_counts_).tolist())
    
    return weights