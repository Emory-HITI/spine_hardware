import sklearn.metrics
from sklearn.preprocessing import OneHotEncoder
import numpy as np
    
def per_image_metrics_function(y_predicted, y_probs, y_true):
    '''
    This function takes an input of predictions and true values and returns weighted precision, recall, f1 scores,
    and AUC scores. 
    Inputs:
        y_predicted: NumPy array of shape (n_samples,) which contains predictions of categories
        y_probs: NumPy array of shape (n_samples, n_classes) which contains probabilities for each class
        y_true: NumPy array of shape (n_samples,) which contains actual labels for samples
    Outputs:
        f1_score: Weighted F1-score
        precision: Weighted Precision score
        recall: Weighted recall score
        auc: Weighted AUC score calculated using One-Versus-Rest Approach
        confusion_matrix: Confusion Matrix
    '''
    
    params = {
        'y_true': y_true,
        'y_pred': y_predicted,
        'average': 'weighted'
    }
    f1_score = sklearn.metrics.f1_score(**params)
    precision = sklearn.metrics.precision_score(**params)
    recall = sklearn.metrics.recall_score(**params)
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true = y_true, y_pred = y_predicted)
    
    encoder = OneHotEncoder()
    y_encoded = encoder.fit_transform(np.array(y_true).reshape(-1,1))
    auc = sklearn.metrics.roc_auc_score(y_true=y_encoded.toarray(), y_score=y_probs, average='weighted', multi_class='ovr')
    
    return f1_score, precision, recall, auc, confusion_matrix

def per_acc_metrics_function(accession_df, accession_column):
    '''
    This function takes an input dataframe of accession_ids, predictions, and true values and returns weighted precision, recall, f1 scores, and confusion matrix on a per patient basis. 
    Inputs:
        accession_df: DataFrame of shape (n_samples,3) which contains a accession_id column, a column called predictions, and a column called labels 
        accession_column: Name of the accession ID column
    Outputs:
        f1_score: Weighted F1-score
        precision: Weighted Precision score
        recall: Weighted recall score
        confusion_matrix: Confusion Matrix
    '''
    by_accession = accession_df.copy()
    
    accession_ids = []
    y_predicted = []
    y_true = []
    grouped = by_accession.groupby(accession_column)
    for accession_id, data in grouped:   
        if data['labels'].nunique() != 1:
            print(f"Problem with accession ground truth for {accession_id}")
            continue
        accession_ids.append(accession_id)
        label = data['labels'].unique()[0]
        y_true.append(label)

        if data['predictions'].nunique() == 1:
            prediction = data['predictions'].unique()[0]
        else:
            prediction = data['predictions'].mode()[0]
        y_predicted.append(prediction)
    
    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)
    
    params = {
        'y_true': y_true,
        'y_pred': y_predicted,
        'average': 'weighted'
    }
    
    f1_score = sklearn.metrics.f1_score(**params)
    precision = sklearn.metrics.precision_score(**params)
    recall = sklearn.metrics.recall_score(**params)
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true = y_true, y_pred = y_predicted)
    
    return f1_score, precision, recall, confusion_matrix