import torch
import time
import logging
import random
import copy
import pandas as pd
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from models.bmav import BMAV
from datasets.spinal_hardware import SpinalHardwareDataset
from utils import splitting_functions, metrics, evaluate, compute_weights, compiling, augmentations, aug_helper


class BMAVAgent:
    """
    BMAV agent that does n splits of train, val and test
    and stores metrics, model dictionaries, and training curves
    """
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger()
        
        self.f1_scores = {}
        self.precision_scores = {}
        self.recall_scores = {}
        self.auc_scores = {}
        self.confusion_matrices = {}
        self.model_dicts = {}
        self.train_losses = {}
        self.train_accs = {}
        self.confusion_pats = {}
    
    def run(self):
        for n in range(1, self.config.num_samples+1):
            self.logger.info(f"Split {n}/{self.config.num_samples}...")
            
            agent = BMAVAgentOneSample(self.config)
            agent.run_training()
            agent.test()
            
            # Appending results
            self.f1_scores[n] = agent.f1_score
            self.precision_scores[n] = agent.precision
            self.recall_scores[n] = agent.recall
            self.auc_scores[n] = agent.auc
            self.confusion_matrices[n] = agent.confusion_matrix
            self.model_dicts[n] = agent.best_model_wts
            self.train_losses[n] = agent.losses
            self.train_accs[n] = agent.accs           
            agent.temp.to_csv(f"{self.config.summary_dir}/accession_test_{n}.csv", index=False)
            agent.test.to_csv(f"{self.config.summary_dir}/test_{n}.csv", index=False)
            agent.probs.to_csv(f"{self.config.summary_dir}/probs_{n}.csv", index=False)
     
    def finalize(self):
        self.logger.info(f"Compiling and saving results from {self.config.num_samples} splits to {self.config.summary_dir}...")
        compiled_numeric = compiling.compile_numeric(self.f1_scores, self.precision_scores, self.recall_scores, self.auc_scores)
        compiled_matrices = compiling.compile_matrices(self.confusion_matrices)
        compiled_loss_curves = compiling.compile_curves(self.train_losses)
        compiled_acc_curves = compiling.compile_curves(self.train_accs)
        
        compiled_numeric.to_csv(f"{self.config.summary_dir}/numeric.csv", index=False)
        np.save(f"{self.config.summary_dir}/matrices.npy", compiled_matrices) 
        compiled_loss_curves.to_csv(f"{self.config.summary_dir}/loss.csv", index=False)
        compiled_acc_curves.to_csv(f"{self.config.summary_dir}/acc.csv", index=False)
        torch.save(self.model_dicts, f"{self.config.summary_dir}/state_dicts.pt")


class BMAVAgentOneSample:
    """
    An agent class that takes a Basic Append View Model and can do one train, validation, and test run.
    It takes a configuration file and can run one train/validation run
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger()
        
        self.train_val_dataset = pd.read_csv(self.config.train_val_dataset_fp)
        
        self.random_state = random.randint(0,1000)
        
        self.train_val_test_data = splitting_functions.split_data(self.train_val_dataset, config.patient_id_col, config.model_columns.labels_col, val_size=config.val_size, random_state=self.random_state)
        
        self.train_val_test_data['test'] = pd.read_csv(self.config.test_dataset_fp)
        
        self.logger.info("Getting mean and standard deviation of training set")
        self.mean_, self.std_ = aug_helper.get_training_mean_std_bmav(dataset=self.train_val_test_data['train'], size_val=config.aug_params.size_val, **config.model_columns)
    
        augs = {x: augmentations.apply_augmentations_with_norm(mode=x, avg_pop_mean=self.mean_,avg_pop_std=self.std_, **config.aug_params) for x in ['train', 'val', 'test']}
        
        self.datasets = {x: SpinalHardwareDataset(self.train_val_test_data[x], augs[x], **config.model_columns) for x in ['train','val','test']}
        
        self.dataset_sizes = {x: len(self.datasets[x]) for x in ['train', 'val', 'test']}
        
        self.dataloaders = {x: DataLoader(self.datasets[x], **config.dataloader_params) for x in ['train', 'val']}
        self.dataloaders['test'] = DataLoader(self.datasets['test'], batch_size=1, shuffle=False, num_workers=1) 
        
        self.num_classes = len(self.train_val_dataset[config.model_columns.labels_col].value_counts())

        self.model = BMAV(num_classes=self.num_classes, **config.bmav_params)
        self.optimizer = torch.optim.Adam([
            {'params': list(self.model.parameters())[:-6], 'lr': config.backbone_lr},
            {'params': list(self.model.parameters())[-6:], 'lr': config.modelhead_lr}
        ])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **config.scheduler_params)
                
        weights = compute_weights.compute_weights(self.train_val_test_data['train'], config.model_columns.labels_col) 
        self.criterion = torch.nn.CrossEntropyLoss(weight=weights)
        
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        self.num_epochs = config.num_epochs
        self.best_acc = 0
        self.current_epoch = 1
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.losses = {"train":[], "val": []}
        self.accs = {"train":[], "val": []}
    
    def run_training(self):
        try:
            since = time.time()
            self.model.to(self.device)
            self.criterion.to(self.device)

            for epoch in tqdm(range(1, self.num_epochs+1)):
                self.train()
                self.validate()
                self.current_epoch += 1

            time_elapsed = time.time() - since
            self.model.to("cpu")
            self.criterion.to("cpu")
            self.logger.info(f"Training Complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s")
            self.model.load_state_dict(self.best_model_wts)
            
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")
    
    def train(self):
        '''
        Runs one epoch of training
        '''
        self.model.train()
        
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in tqdm(self.dataloaders["train"]):
            inputs = list(map(lambda x: x.to(self.device), inputs))
            labels = labels.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                # backward + optimize
                loss.backward()
                self.optimizer.step()

            # statistics
            running_loss += loss.item() * inputs[0].size(0)
            running_corrects += torch.sum(preds == labels.data)        
            
        self.scheduler.step()
        epoch_loss = running_loss / self.dataset_sizes["train"]
        epoch_acc = running_corrects.double() / self.dataset_sizes["train"]

        print(f"\nEpoch {self.current_epoch} Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.4f}\n")
        self.losses["train"].append(epoch_loss)
        self.accs["train"].append(epoch_acc.cpu().item())
    
    def validate(self):
        '''
        Runs one epoch of validation
        '''
        self.model.eval()
        
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in tqdm(self.dataloaders["val"]):
            inputs = list(map(lambda x: x.to(self.device), inputs))
            labels = labels.to(self.device)

            # forward
            with torch.no_grad():
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs[0].size(0)
            running_corrects += torch.sum(preds == labels.data)           

        epoch_loss = running_loss / self.dataset_sizes["val"]
        epoch_acc = running_corrects.double() / self.dataset_sizes["val"]

        print(f"\nEpoch {self.current_epoch} Val Loss: {epoch_loss:.4f} Val Acc: {epoch_acc:.4f}\n")
        self.losses["val"].append(epoch_loss)
        self.accs["val"].append(epoch_acc.cpu().item())

        # deep copy the model
        if epoch_acc > self.best_acc:
            self.best_acc = epoch_acc
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
    
    def test(self):
        '''
        Computes metrics on test set
        '''
        # Per bounding box metrics
        preds, probs, labels = evaluate.evaluate_model(self.model, self.dataloaders, self.device, phase="test")
        f1_score, precision, recall, auc, confusion_matrix = metrics.per_image_metrics_function(preds, probs, labels)
        self.f1_score = f1_score
        self.precision = precision
        self.recall = recall
        self.auc = auc
        self.confusion_matrix = confusion_matrix
      
        # Per patient metrics
        acc_ids = self.train_val_test_data['test'][self.config.accession_id_col].values
        temp = np.vstack((acc_ids, preds, labels)).T
        self.temp = pd.DataFrame(temp, columns=[self.config.accession_id_col, "predictions", "labels"])
        self.test = self.train_val_test_data['test']
        self.probs = pd.DataFrame(probs)