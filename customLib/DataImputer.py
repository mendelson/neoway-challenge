import pandas as pd
import math
import numpy as np
from tqdm import tqdm
from sklearn.impute import KNNImputer

# DataImputer: performs Nearest Neighbor Imputation on
# the input_df variable.
class DataImputer():
    def __init__(self, input_df):
        """
        This constructor initializes the internal DataFrame performing the
        necessary pre-processing.

        Args:
            input_df (DataFrame): the original DataFrame dataset.
        """

        self.__batch_size = 10000

        # Shuffling data
        self.__df = input_df.sample(frac=1).copy()

        self.__pre_proc_df()

        # self.__n_neighbors = int(math.sqrt(self.__df.dropna(axis=0).shape[0]))
        self.__n_neighbors = int(math.sqrt(self.__batch_size))

    def impute(self):
        """
        This method performs the Nearest Neighbor Imputation.
        """

        batches = self.__get_batches()
        # return batches

        # The weight parameter was set to distance in order to increase the influence of closer elements
        imputer = KNNImputer(n_neighbors=self.__n_neighbors, weights='distance')

        df = pd.DataFrame(columns=list(self.__df.columns))

        print('Performing imputations...')
        for batch in tqdm(batches):
            data = imputer.fit_transform(batch.drop(['name'], axis=1))
            batch.iloc[:, 1:] = data
            df = pd.concat([df, batch])

        self.__update_df(df)

    def get_data(self):
        """
        This method transforms the full dataset back to the original
        format as in the .csv files and returns the dataframe.

        Returns:
            DataFrame: the transformed DataFrame as in the .csv format
        """

        self.__get_readable_df()

        for col in self.__df:
            if (self.__df[col].dtype != 'object' and col != 'IMC') or col == 'name':
                self.__df[col] = self.__df[col].astype('int32')

        return self.__df

    # Private methods
    def __pre_proc_df(self):
        """
        This method performs the pre-processing steps on the dataset, i.e.,
        normalization and categorical values manipulation.
        """

        self.__norm_params = {}
        self.__str_and_num = {}
        self.__categorical_cols = []

        for col in self.__df.columns:
            if col != 'name':
                if self.__df[col].dtype != 'object':
                    self.__col_minmax_norm(col)
                else:
                    self.__categorical_cols.append(col)
                    self.__categorical_to_num(col)
                    self.__col_minmax_norm(col)

    def __col_minmax_norm(self, col):
        """
        This method performs the min-max normalization method.

        Args:
            col (str): the name of the column to be normalized in the
                       self.__df DataFrame
        """

        maxim = self.__df[col].max()
        minim = self.__df[col].min()
        self.__norm_params[f'max_{col}'] = maxim
        self.__norm_params[f'min_{col}'] = minim

        self.__df[col] = (self.__df[col] - minim)/(maxim - minim)

    def __categorical_to_num(self, col):
        """
        This method converts categorical values to a numeric value (not fit for
        predictions, only for imputation).

        Args:
            col (str): the name of the column to be processed in the
                       self.__df DataFrame
        """

        original_values = list(self.__df[col].unique())
        original_values.remove(np.nan)

        for idx, original_value in enumerate(original_values):
            self.__str_and_num[f'{col}_{original_value}'] = float(idx)
            self.__str_and_num[f'{col}_{idx}.0'] = original_value

        self.__str_and_num[f'{col}_{np.nan}'] = np.nan
        self.__str_and_num[f'{col}_{np.nan}'] = np.nan

        self.__df[col] = self.__df[col].apply(lambda x: self.__str_and_num[f'{col}_{x}'])

    def __num_to_categorical(self, col):
        """
        This method converts numeric values to its categorical value.

        Args:
            col (str): the name of the column to be processed in the
                       self.__df DataFrame
        """

        self.__df[col] = self.__df[col].apply(lambda x: self.__str_and_num[f'{col}_{x}'])

    def __get_readable_df(self):
        """
        This method transforms the dataset back to human-readable values.
        """

        for col in self.__df.columns:
            if col != 'name':
                if col not in self.__categorical_cols:
                    self.__denorm_data(col)
                else:
                    self.__denorm_data(col, True)
                    self.__num_to_categorical(col)

    def __denorm_data(self, col, round_values=False):
        """
        This method reverts the min-max normalization process back to its original values.

        Args:
            col (str): the name of the column to be processed in the
                       self.__df DataFrame
            round_values (bool): indicates if the reconstructed values should
                          be rounded (used for categorical columns)
        """

        maxim = self.__norm_params[f'max_{col}']
        minim = self.__norm_params[f'min_{col}']

        if round_values:
            self.__df[col] = (self.__df[col]*(maxim - minim) + minim).round()

        else:
            self.__df[col] = self.__df[col]*(maxim - minim) + minim

    def __get_batches(self):
        """
        This method splits the dataset into batches of max size self.__batch_size
        in order to speed up the KNN Imputation process.

        Returns:
            list: the list of DataFrame batches
        """

        train_df = self.__df.dropna(axis=0).copy()
        missing_df = self.__df[self.__df.isna().any(axis=1)].copy()

        batches = []

        start = 0
        end = int(start + self.__batch_size/2)

        while start < missing_df.shape[0]:
            batches.append(pd.concat([train_df[start:end].copy(), missing_df[start:end].copy()]))

            start = end
            end += int(self.__batch_size/2)

        return batches

    def __update_df(self, df):
        """
        This method receives the imputed data frame and uses it to update self.__df. This is
        necessary because not all the rows are used in the imputation process, so not all
        original rows are present in the imputed DataFrame.

        Args:
            df (DataFrame): the imputed DataFrame.
        """

        present = list(df['name'])
        to_be_checked = list(self.__df['name'])

        to_be_added = np.setdiff1d(to_be_checked, present)

        to_add = self.__df[self.__df['name'].isin(to_be_added)]
        df = pd.concat([df, to_add])

        self.__df = df
