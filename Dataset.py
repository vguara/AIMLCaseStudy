import random
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV


class Dataset:
    def __init__(self, DSpath):
        self.data = pd.read_csv(DSpath)
        self.fraud_rate = Dataset.define_fraud_rate(self.data)
        self.subsets = {}
        # self.validation_sets = {}
        self.train_data = None
        self.test_data = None
        self.ratio = None

    def drop_id_or_time(self):
        """
        Remove 'Time' from the old dataset and id from the new one in order to make them comparable.
        """
        if self.fraud_rate > 0.1:
            self.data = self.data.drop('id', axis=1)
        else:
            self.data = self.data.drop('Time', axis=1)

    def reduce_data_set(self, ratio, data="Full"):
        if data == "Full":
            nrows = int(self.data.shape[0] * ratio)
            self.data = self.data.iloc[:nrows]
        elif data == "Train":
            nrows = int(self.train_data.shape[0] * ratio)
            self.train_data = self.train_data.iloc[:nrows]
        elif data == "Test":
            nrows = int(self.test_data.shape[0] * ratio)
            self.test_data = self.test_data.iloc[:nrows]
        else:
            print("Invalid data")



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

            # Call function to split feature from labels and store as tuple.
            # Store the subset in the dictionary with the ratio as the key
            # self.subsets[ratio] = subset_df
            self.subsets[ratio] = Dataset.split_feature_label(subset_df)


    @staticmethod
    def define_fraud_rate(data):
        # Define the fraud rate of a dataset
        # can be used for the whole dataset or for each 

        num_rows = data.shape[0]
        label = data['Class']
        fraud_count = (label == 1).sum()
        return fraud_count/num_rows

    def create_train_test(self, ratio, seed=None):
        '''
        Divides the data into train and test subsets.
        :param ratio: The ratio between training set and test set
        :param seed: seed for random splitting
        '''

        self.ratio = ratio
        if 0 < ratio < 1:
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
        elif ratio == 0:
            self.test_data = self.data
        else:
            self.train_data = self.data

    @staticmethod
    def split_feature_label(dataframe, label="Class"):
        '''

        :param dataframe: The dataframe to split between features and label
        :param label: the name of the label column in the dataframe
        :return: a dataframe of features and another of the frame.
        '''
        label_df = dataframe.loc[:, label]
        features = dataframe.drop(label, axis=1, inplace=False)
        return features, label_df

    def create_train_test_set_fraud(self, fraud_data_ratio, test_data_ratio, seed=None):
        '''
        Creates the train and test subsets. Mostly useful for imbalanced datasets.
        :param fraud_data_ratio: Ratio of distribution of fraud data between training and test set
        :param test_data_ratio: Ratio of fraud data inside the test set
        :param seed: seed for random splitting

        '''
        shuffle_data = self.data.sample(frac=1, random_state=seed).reset_index(drop=True)

        fraud_data = shuffle_data [shuffle_data ['Class'] == 1]
        legit_data = shuffle_data [shuffle_data ['Class'] == 0]

        #split fraud data
        rows_fraud = fraud_data.shape[0]
        fraud_data_train_rows = int(rows_fraud * fraud_data_ratio)
        fraud_data_train = fraud_data.iloc[:fraud_data_train_rows]
        fraud_data_test = fraud_data.iloc[fraud_data_train_rows:]

        #split non-fraud,but set a particular ratio for test data
        rows_fraud_test = fraud_data_test.shape[0]
        rows_legit_test = int((rows_fraud_test/test_data_ratio) - rows_fraud_test)
        legit_data_train = legit_data.iloc[rows_legit_test:]
        legit_data_test = legit_data.iloc[:rows_legit_test]

        # Combine both
        train_data_sorted = pd.concat([fraud_data_train, legit_data_train])
        test_data_sorted = pd.concat([fraud_data_test, legit_data_test])
        test_fraud_rate = Dataset.define_fraud_rate(test_data_sorted)

        # Shuffle the datasets
        self.train_data = train_data_sorted.sample(frac=1, random_state=seed).reset_index(drop=True)
        self.test_data = test_data_sorted.sample(frac=1, random_state=seed).reset_index(drop=True)


    def define_label_features(self, label, features=None):
        '''
        Divides the data into features and labels for the whole training set.
        :param label: name of the label column
        :param features: names of the features column
        '''
        if self.train_data is not None:
            self.train_label = self.train_data.loc[:,label]

            if features: # if we want to test specific features
                self.train_features = self.train_data.loc[:,features]
            else: # otherwise all features included
                self.train_features = self.train_data.drop(label, axis=1, inplace=False)

        if self.test_data is not None:
            self.test_label = self.test_data.loc[:, label]
            if features:
                self.test_features = self.test_data.loc[:, features]
            else:
                self.test_features = self.test_data.drop(label, axis=1, inplace=False)

    def reduce_test_data_ratio(self, ratio, seed=None):
        '''

        :param ratio: the ratio of fraud data desired in the test set
        :param seed: seed for random splitting
        '''
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
        self.test_data = new_test_data






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

