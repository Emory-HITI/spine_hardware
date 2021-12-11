import pandas as pd
import numpy as np

def compile_numeric(f1_scores, precision_scores, recall_scores, auc_scores):
    '''
    Compiles the numeric scores
    '''
    df = pd.DataFrame([f1_scores, precision_scores, recall_scores, auc_scores], index=["F1", "Precision", "Recall", "AUC"])
    df = df.T
    df.index.name = "Trial"
    
    return df

def compile_numeric_pats(f1_scores, precision_scores, recall_scores):
    '''
    Compiles the numeric scores by patient
    '''
    df = pd.DataFrame([f1_scores, precision_scores, recall_scores], index=["F1", "Precision", "Recall"])
    df = df.T
    df.index.name = "Trial"
    
    return df

def compile_curves(curve_data):
    '''
    Compile dictionary of train/acc curves by trial into dataframes
    '''
    df = pd.DataFrame(curve_data)
    
    return df

def compile_matrices(confusion_matrices):
    '''
    Compiles dictionary of confusion matrices
    '''
    list_ = []
    for key, value in confusion_matrices.items():
        list_.append(value)
    
    return np.array(list_)