import nibabel as nib
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, list_IDs, labels, image_path = "allregistered/", to_fit=True, batch_size=32, dim=(182, 218, 182), n_channels=1, n_classes=10, shuffle=True):
        self.list_IDs = list_IDs
        self.labels = labels
        self.image_path = image_path
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size: (index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs.iloc[k] for k in indexes]
        labels_temp = [self.labels.iloc[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_IDs_temp)

        if self.to_fit:
            y = self._generate_y(labels_temp)
            return X, y
        else:
            return X    
    
    def _generate_X(self, list_IDs_temp):
        
        # Initialization
        X1 = np.empty((self.batch_size, *self.dim))
        X2 = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            pet =nib.load(self.image_path + str(ID["PET_ID"]) + "_registered.nii.gz")
            mri = nib.load(self.image_path + str(ID["MRI_ID"]) + "_registered.nii.gz")
            X1[i,] = mri.get_fdata()
            X2[i,] = pet.get_fdata()
            mri.uncache()
            pet.uncache()
        return [[X1,X2]]
    
    def _generate_y(self, labels_temp):
        y = np.empty((self.batch_size, 1), dtype=int)

        # Generate data
        for i, label in enumerate(labels_temp):
            # Store sample
            y[i,] = label

        return y