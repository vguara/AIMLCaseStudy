import random
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV


class Dataset:
    def __init__(self, DSpath):
        self.data = pd.read_csv(DSpath)
        self.fraud_rate = Dataset.define_fraud_rate(self.data)
        self.subsets = {}
        self.validation_sets = {}
        self.train_data = None
        self.test_data = None

    def create_subsets(self, ratios, seed=None):
        '''
        Populates the subset attribute with subsets of the data for each given ratio.
        :param ratios: a list of ratios of fraud data over total data
        '''
        class_1_df = self.train_data[self.train_data['Class'] == 1]
        class_0_df = self.train_data[self.train_data['Class'] == 0]

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
                subset_df = self.train_data.copy()

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

    def reduce_test_data_ratio(self, ratio, seed=None):
        if self.test_data is None:
            print("No test data")
            return
        #Separate fraud from nonfraud
        fraud_data = self.test_data[self.test_data['Class'] == 1]
        non_fraud_data = self.test_data[self.test_data['Class'] == 0]


        current_ratio = len(fraud_data) / len(self.test_data)
        if current_ratio < ratio:
            print(f"cannot reduce test data ratio. current ratio: {current_ratio}")
            return


        fraud_desired = int((ratio / (1-ratio)) * len(non_fraud_data))

        new_fraud_data = resample(fraud_data, replace=False, n_samples=fraud_desired, random_state=seed)

        new_test_data = pd.concat([new_fraud_data, non_fraud_data])
        new_test_data = new_test_data.sample(frac=1, random_state=seed).reset_index(drop=True)
        print(f"Old test data - fraud size: {len(fraud_data)}, ratio {current_ratio}")
        self.test_data = new_test_data
        new_ratio = len(new_fraud_data)/len(self.test_data)
        print(f"New test data - fraud size: {len(new_fraud_data)}, ratio {new_ratio}")



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



####

def find_best_model(model, param_grid, cv, scoring, training_features, training_labels):
    """
    Perform cross-validation to find the best model.

    Parameters:
    model (estimator): The machine learning model to be optimized.
    param_grid (dict): The parameter grid to search over.
    cv (int): The number of folds for cross-validation. the k-fold cross validation is done for each parameter grid.
    scoring (str): The scoring method to evaluate the model.
    training_features: Training data features.
    training_labels: Training data labels.

    Returns:
    best_model (estimator): The best model found by GridSearchCV.
    best_params (dict): The best parameters found by GridSearchCV.
    best_score (float): The best score achieved by the best model.

    Example parameters:
    model = SVC()
    param_grid = {
        'C': [0.1, 1, 10, 100],
    }
    scoring = 'accuracy'
    cv = 5


    """





    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring)
    grid_search.fit(training_features, training_labels)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_model, best_params, best_score

