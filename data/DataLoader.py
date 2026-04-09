from abc import abstractmethod
import random
import pandas as pd
import tensorflow as tf

class DataLoader():
    
    def __init__(self, val_size, mode, max_img=10000):
        """
        Parameters:
        val_size: Percentage of Patients to be used for validation. Use 0.0 when no validation run for training is made.
        mode: Loads different modes for training/testing. Further specified in subclasses.
        max_img: Maximum number of images to load. If no value is set, all images will be loaded.
        """
        self.mode = mode
        self.max_img = max_img
        self.val_size = val_size
        
        self.dataset = {}
            
        self._pull_data()    
            
        self.train_ids, self.validation_ids = self._train_val_split()
        self.current_ids = self.train_ids
            
        self.meta = pd.DataFrame()
        self.filtered_meta = pd.DataFrame()
        
        
    def _train_val_split(self):
        list_of_ids = list(self.dataset.keys())
        random.shuffle(list_of_ids)

        split_idx = int(len(list_of_ids) * (1 - self.val_size))
        train_ids = list_of_ids[:split_idx]
        val_ids = list_of_ids[split_idx:]

        return train_ids, val_ids
    
    
    @abstractmethod
    def _pull_data(self):
        pass 