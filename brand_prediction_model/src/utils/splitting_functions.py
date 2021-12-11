from sklearn import model_selection

def split_data(train_val_dataset, patient_id_column, label_column, val_size=0.2, random_state=0):   
    """
    Function to split data into train, test, val with patient stratification
    You need 4 or more unique patients per brand for these functions to 
    ensure that each train, val and test set gets a brand
    Returns a dictionary
    """
    
    master = train_val_dataset.copy()
    
    subbed_data = master[[patient_id_column, label_column]].drop_duplicates().reset_index(drop=True)
    
    # Assigning the X (patient_ids) and the y (patient_labels)
    patient_ids = subbed_data[patient_id_column]
    patient_labels = subbed_data[label_column]

    # Get train and test
    Train_IDs_Strat, Val_IDs_Strat, Train_Labels_Strat, Val_Labels_Strat = model_selection.train_test_split(patient_ids, patient_labels, test_size=val_size, random_state=random_state, stratify=patient_labels)

    Train_DCMs_Strat = master[master[patient_id_column].isin(Train_IDs_Strat)].reset_index(drop=True)
    Val_DCMs_Strat = master[master[patient_id_column].isin(Val_IDs_Strat)].reset_index(drop=True)
    
    assert len(Train_DCMs_Strat) > 0 and len(Val_DCMs_Strat) > 0, "Not all the sets more than 1 row"
    assert len(Train_DCMs_Strat[label_column].unique()) == len(Val_DCMs_Strat[label_column].unique()), "Splits don't have the same number of unique labels"
    
    return dict(zip(["train","val"],[Train_DCMs_Strat, Val_DCMs_Strat]))