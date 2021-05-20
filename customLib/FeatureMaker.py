import pandas as pd
from tqdm import tqdm
import pickle
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch

class FeatureMaker():
    def __init__(self, input_ind_df=None, input_con_df=None, load_from_file=False):
        """
        This constructor initializes the internal DataFrames performing the
        necessary pre-processing.

        Args:
            input_ind_df (DataFrame): the original individuals DataFrame dataset.
            input_con_df (DataFrame): the original connections DataFrame dataset.
            load_from_file (bool): whether the data should be loaded from file.
        """

        self.__data_file = 'data/processed_dataset.pkl'

        if load_from_file:
            self.__load_data_from_file()
        else:
            self.__ind_df = input_ind_df.copy()
            self.__con_df = input_con_df.copy()

            self.__pre_proc_ind_df()
            self.__pre_proc_con_df()

            self.__save_data_to_file()

    def prepare_sets(self):
        """
        This method takes the dataset and prepares the train, valid and test sets.
        Besides that, the entries to be predicted are also separated in a group.
        """

        all_indices = np.arange(len(self.__targets))

        features = np.array(self.__features)
        targets = np.array(self.__targets)
        # indices = np.array(self.__indices)

        predict_set_indices_on_array = np.argwhere(np.isnan(targets))

        predict_set_features = features[predict_set_indices_on_array]
        predict_set_targets = targets[predict_set_indices_on_array]
        # self.__predict_set_indices = indices[predict_set_indices_on_array]

        set_indices = np.delete(all_indices, predict_set_indices_on_array)

        complete_set_features = features[set_indices]
        complete_set_targets = targets[set_indices]

        complete_set_size = len(complete_set_targets)

        start = 0
        stop = int(0.7*complete_set_size)
        train_set_features = complete_set_features[start:stop]
        train_set_targets = complete_set_targets[start:stop]

        start = stop
        stop += int(0.15*complete_set_size)
        valid_set_features = complete_set_features[start:stop]
        valid_set_targets = complete_set_targets[start:stop]

        start = stop
        test_set_features = complete_set_features[start:]
        test_set_targets = complete_set_targets[start:]

        self.__train_loader = self.__batch_data(train_set_features, train_set_targets)
        self.__valid_loader = self.__batch_data(valid_set_features, valid_set_targets)
        self.__test_loader = self.__batch_data(test_set_features, test_set_targets)
        self.__predict_loader = self.__batch_data(predict_set_features, predict_set_targets, shuffle=False)

    def get_loader(self, loader):
        """
        This method returns the desired DataLoader.

        Args:
            loader (str): the DataLoader to be returned ('train', 'valid',
                          'test' or 'predict').

        Returns:
            DataLoader: the requested DataLoader.
        """

        if loader == 'train':
            return self.__train_loader
        elif loader == 'valid':
            return self.__valid_loader
        elif loader == 'test':
            return self.__test_loader
        elif loader == 'predict':
            # return self.__predict_loader, self.__predict_set_indices
            return self.__predict_loader
        else:
            return None

    def get_ind_df(self):
        """
        Test function for debugging purposes

        Returns:
            DataFrame: the individuals DataFrame after pre-processing
        """

        return self.__ind_df

    def get_con_df(self):
        """
        Test function for debugging purposes: get the connections Dataframe.

        Returns:
            DataFrame: the connections DataFrame after pre-processing
        """

        return self.__con_df

    def get_dataset(self):
        """
        Test function for debugging purposes: get the processed dataset.

        Returns:
            list: the dataset with features and targets
        """

        return self.__features, self.__targets

    def fill_nan_values(self, predictions):
        """
        This method fill the missing values with predictions.

        Args:
            predictions (list): the predictions that will fill the missing values.
        
        Returns:
            numpy.array: all features.
            numpy.array: all targets.
        """

        features = np.array(self.__features)
        targets = np.array(self.__targets)

        i = 0
        for idx, target in enumerate(targets):
            if np.isnan(target):
                targets[idx] = predictions[i]
                i += 1

        return features, targets

    def get_columns(self):
        """
        This method returns the columns names of the conections structure.
        
        Returns:
            list: the columns names.
        """

        return self.__columns

    def get_norm_params(self):
        """
        This method returns the normalizations parameters.
        
        Returns:
            dict: the normalization parameters.
        """

        return self.__norm_params

    # Private methods
    def __batch_data(self, features, targets, shuffle=True, batch_size=1000):
        """
        This method creates the DataLoader for the input data.

        Args:
            features (numpy.array): the features to be loaded.
            targets (numpy.array): the targets to be loaded.
            shuffle (bool): whether the data should be shuffled.
            batch_size (int): the batch size.

        Returns:
            DataLoader: the DataLoader of the input data.
        """

        # Create Tensor dataset
        data = TensorDataset(torch.from_numpy(features), torch.from_numpy(targets))

        # DataLoader
        loader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)

        return loader

    def __load_data_from_file(self):
        """
        This method loads the processed dataset from a file.
        """

        with open(self.__data_file, 'rb') as f:
            # self.__features, self.__targets, self.__indices = pickle.load(f)
            self.__features, self.__targets, self.__columns, self.__norm_params = pickle.load(f)

    def __save_data_to_file(self):
        """
        This method saves the processed dataset to a file for future use.
        """

        with open(self.__data_file, 'wb') as f:
            # pickle.dump([self.__features, self.__targets, self.__indices], f)
            pickle.dump([self.__features, self.__targets, self.__columns, self.__norm_params], f)

    def __pre_proc_ind_df(self):
        """
        This method performs the pre-processing steps on the individuals dataset,
        i.e., normalization and encoding.
        """

        self.__norm_params = {}
        self.__str_and_num = {}

        for col in self.__ind_df.columns:
            if col != 'name':
                if self.__ind_df[col].dtype != 'object':
                    self.__col_minmax_norm(col)
                else:
                    self.__encode_col(col)

    def __col_minmax_norm(self, col):
        """
        This method performs the min-max normalization method.

        Args:
            col (str): the name of the column to be normalized in the
                       self.__ind_df DataFrame
        """

        maxim = self.__ind_df[col].max()
        minim = self.__ind_df[col].min()
        self.__norm_params[f'max_{col}'] = maxim
        self.__norm_params[f'min_{col}'] = minim

        self.__ind_df[col] = (self.__ind_df[col] - minim)/(maxim - minim)

    def __encode_col(self, col, which_df='ind'):
        """
        This method performs the one-hot-encoding process in a column of
        the specified DataFrame.

        Args:
            col (str): the name of the column to be encoded in the
                       self.__ind_df DataFrame
            which_df (str): which DataFrame to be encoded. If 'ind',
                       then it is self.__ind_df; if 'con', then it is
                       self.__con_df; o.w. nothing is done. Default: 'ind'
        """

        if which_df == 'ind':
            encoding = pd.get_dummies(self.__ind_df[col], prefix=col, prefix_sep='-')
            self.__ind_df = self.__ind_df.drop(col, axis=1)
            self.__ind_df = self.__ind_df.join(encoding)
        elif which_df == 'con':
            encoding = pd.get_dummies(self.__con_df[col], prefix=col, prefix_sep='-')
            self.__con_df = self.__con_df.drop(col, axis=1)
            self.__con_df = self.__con_df.join(encoding)

    def __pre_proc_con_df(self):
        # Encoding
        for col in self.__con_df.columns:
            if self.__con_df[col].dtype == 'object':
                self.__encode_col(col, 'con')

        # Feature making
        con_list = self.__con_df.values.tolist()
        # indices = self.__con_df.index.tolist()

        self.__columns = None

        features = []
        targets = []
        for con in tqdm(con_list):
            v1_name = con[0]
            v2_name = con[1]
            target = con[2]

            v1 = self.__ind_df[self.__ind_df['name'] == v1_name].drop(['name'], axis=1).values.tolist()
            v2 = self.__ind_df[self.__ind_df['name'] == v2_name].drop(['name'], axis=1).values.tolist()

            if self.__columns == None:
                self.__columns = list(self.__ind_df[self.__ind_df['name'] == v1_name].drop(['name'], axis=1).columns)
                self.__columns += list(self.__ind_df[self.__ind_df['name'] == v2_name].drop(['name'], axis=1))
                self.__columns += list(self.__con_df.columns)[3:]

            feature = v1[0] + v2[0] + con[3:]

            features.append(feature)
            targets.append(target)

            ##### deletar
        #     break

        # with open(self.__data_file, 'rb') as f:
        #     self.__features, self.__targets, _ = pickle.load(f)
        #     # a = pickle.load(f)
        #     # self.__features = a[0]
        #     # self.__targets = a[1]

        # with open(self.__data_file, 'wb') as f:
        #     # pickle.dump([self.__features, self.__targets, self.__indices], f)
        #     pickle.dump([self.__features, self.__targets, self.__columns, self.__norm_params], f)
            #########################################################################################

        # self.__features = features
        # self.__targets = targets
        # # self.__indices = indices