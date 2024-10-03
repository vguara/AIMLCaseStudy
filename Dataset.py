import random
import pandas as pd

class Dataset:
    def __init__(self, DSpath):
        self.data = pd.read_csv(DSpath)
        self.fraud_rate = self.define_fraud_rate(self.data)

    def define_fraud_rate(self, data):
        # Define the fraud rate of a dataset
        # can be used for the whole dataset or for each 

        num_rows = data.shape[0]
        label = data.iloc[:'Class']
        fraud_count = (label == 1).sum()
        return fraud_count/num_rows

    def create_train_test(self, ratio, seed=None):
        num_rows = self.data.shape[0]
        shuffled_indices = list(range(num_rows))
        if seed:
            random.seed(seed)
        random.shuffle(shuffled_indices)
        self.train_set_size = int(num_rows * ratio)
        train_indices = shuffled_indices[:self.train_set_size]
        test_indices = shuffled_indices[self.train_set_size:]
        random.seed()  # resetting the seed after shuffling
        self.train_data = self.data.iloc[train_indices]
        self.test_data = self.data.iloc[test_indices]

    def define_label_features(self, label, features=None):
        self.train_label = self.train_data.loc[:,label]
        self.test_label = self.test_data.loc[:,label]
        if features: # if we want to test specific features
            self.train_features = self.train_data.loc[:,features]
            self.test_features = self.test_data.loc[:,features]
        else: # otherwise all features included
            self.train_features = self.train_data.drop(label, axis=1, inplace=False)
            self.test_features = self.test_data.drop(label, axis=1, inplace=False)