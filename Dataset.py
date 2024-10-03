import random
import pandas as pd
from openpyxl.utils.dataframe import dataframe_to_rows


class Dataset:
    def __init__(self, DSpath):
        self.data = pd.read_csv(DSpath)
        self.fraud_rate = Dataset.define_fraud_rate(self.data)
        self.subsets = {}
        self.validation_sets = {}

    def create_subsets(self, ratios, seed=None):
        '''
        Populates the subset attribute with subsets of the data for each given ratio.
        :param ratios: a list of ratios of fraud data over total data
        '''
        class_1_df = self.data[self.data['Class'] == 1]
        class_0_df = self.data[self.data['Class'] == 0]

        # Generate subsets for each ratio
        for ratio in ratios:
            if self.fraud_rate > ratio:  # Too much fraud, reduce fraud cases
                target_class_0_count = len(class_0_df)  # Use all non-fraud cases
                target_class_1_count = int(round((ratio * target_class_0_count)/(1 - ratio)))
                target_class_1_df = class_1_df.sample(n=target_class_1_count, random_state=seed)
                subset_df = pd.concat([target_class_1_df, class_0_df])

            elif self.fraud_rate < ratio:  # Too little fraud, reduce non-fraud cases
                target_class_1_count = len(class_1_df)  # Use all fraud cases
                target_class_0_count = int(round(target_class_1_count * (1 - ratio) / ratio))
                target_class_0_df = class_0_df.sample(n=target_class_0_count, random_state=seed)
                subset_df = pd.concat([class_1_df, target_class_0_df])

            else:  # Fraud rate matches the desired ratio
                # No sampling needed, use the full dataset
                subset_df = self.data.copy()

            # Ensure the total rows in subset match expected rows
            subset_size = target_class_1_count + target_class_0_count
            subset_df = subset_df.sample(frac=1, random_state=seed).reset_index(drop=True)

            # Store the subset in the dictionary with the ratio as the key
            self.subsets[ratio] = subset_df


    @staticmethod
    def define_fraud_rate(data):
        # Define the fraud rate of a dataset
        # can be used for the whole dataset or for each 

        num_rows = data.shape[0]
        label = data['Class']
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

    def create_cross_validation_subsets(self, k=5, keep_fraud_ratio=False):

        if not keep_fraud_ratio:
            for ratio, df in self.subsets.items():
                total_rows = df.shape[0]
                rows_k = int(total_rows/k)
                self.validation_sets[ratio] = []
                for j in range(0, total_rows, rows_k):
                    df_split = df.iloc[j:j+rows_k]
                    self.validation_sets[ratio].append(df_split)
        else:
            for ratio, df in self.subsets.items():
                fraud_cases = df[df['Class'] == 1]
                non_fraud_cases = df[df['Class'] == 0]
                total_rows = non_fraud_cases.shape[0]
                rows_k = int(total_rows/k)
                self.validation_sets[ratio] = []
                for j in range(0, total_rows, rows_k):
                    non_fraud_df_split = non_fraud_cases.iloc[j:j+rows_k]
                    df_concat = pd.concat([non_fraud_df_split, fraud_cases])
                    self.validation_sets[ratio].append(df_concat)

