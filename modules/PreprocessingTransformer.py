from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import numpy as np
import pandas as pd
import pickle


class PreprocessingTransformer():
    def __init__(self, categorical_features:list = [], numerical_features:list = []):
        """
        Fit and apply  transformations on the data,
        StandardScaler for numerical and OneHotEncoder for categorical values
        :param categorical_features: list of categorical columns names
        :param numerical_features: list of numerical columns names
        """
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder()
        self.label_encoder = LabelEncoder()
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.fitted_categorical_columns = categorical_features

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the transformers on the given data
        :param X: features data frame
        :param y: target value
        """
        self.scaler.fit(X[self.numerical_features])
        self.encoder.fit(X[self.categorical_features])
        self.label_encoder.fit(y)

    def split_fit_transform(self, X: pd.DataFrame, y: pd.Series, test_size=0.2):
        """
        Split the data to train and test with respect to the given `test_size` parma.
        Fits the transformers on the train data and then transforms the train and test data
        :param X: features data frame
        :param y: target value
        :param test_size: portion of data to take as test data
        :return: X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_size,
                                                            stratify=y)
        self.fit(X_train, y_train)
        X_train, y_train = self.transform(X_train, y_train)
        X_test, y_test = self.transform(X_test, y_test)
        return X_train, X_test, y_train, y_test

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the transformers on the given data and then transforms it
        :param X: DataFrame of features to fit on
        :param y: Series of the target class
        :return:
        """

        self.fit(X,y)
        return self.transform(X,y)

    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None):
        """
        Transforms the given data
        :param X: DataFrame of features to transform
        :param y: (Optional) Series of the target class
        :return:
        """
        X_train_num = self.scaler.transform(X[self.numerical_features])
        X_train_cat = self.encoder.transform(X[self.categorical_features]).toarray()
        if y is not None:
            y = self.label_encoder.transform(y)

        self.fitted_categorical_columns = self.encoder.get_feature_names_out()

        X_train_num_pd = pd.DataFrame(data=X_train_num, columns=self.numerical_features)
        X_train_cat_pd = pd.DataFrame(data=X_train_cat, columns=self.fitted_categorical_columns)

        return pd.concat((X_train_num_pd,X_train_cat_pd), axis=1), y

    def save(self, path: str):
        """
        Saves the current object to the given path as pickle format
        :param path: The path to save the object
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        """
        Loads the current class
        :param path: Path to the saved pickle file
        :return: The loaded file
        """
        with open(path, "rb") as f:
            instance = pickle.load(f)
        return instance

