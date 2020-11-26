import pandas as pd
from functools import reduce
from sklearn import base
from sklearn import compose
from sklearn import preprocessing
from sklearn import impute


class ColumnSelector(compose.make_column_selector):

    def __init__(self, pattern=None, *, dtype_include=None, dtype_exclude=None, columns=None):
        if columns is not None:
            pattern = '|'.join(columns)
        super().__init__(pattern, dtype_include=dtype_include, dtype_exclude=dtype_exclude)


class FeatureUnion(base.TransformerMixin):

    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def fit(self, X, y=None):
        for (name, t) in self.transformer_list:
            t.fit(X, y)
        return self

    def transform(self, X):
        Xts = [t.transform(X) for _, t in self.transformer_list]
        Xunion = reduce(lambda X1, X2: pd.merge(X1, X2, left_index=True, right_index=True), Xts)
        return Xunion


class FunctionTransformer(preprocessing.FunctionTransformer):

    def transform(self, X):
        Xf = super().transform(X)
        if isinstance(X, pd.DataFrame):
            Xf = pd.DataFrame(Xf, index=X.index, columns=X.columns)
        return Xf


class OrdinalEncoder(preprocessing.OrdinalEncoder):

    def transform(self, X):
        Xf = super().transform(X)
        if isinstance(X, pd.DataFrame):
            Xf = pd.DataFrame(Xf, index=X.index, columns=X.columns)
        return Xf


class OneHotEncoder(preprocessing.OneHotEncoder):

    def transform(self, X):
        Xf = super().transform(X)
        if isinstance(X, pd.DataFrame):
            Xf = pd.DataFrame(Xf.toarray(), index=X.index, columns=self.get_feature_names(X.columns))
        return Xf


class SimpleImputer(impute.SimpleImputer):

    def transform(self, X):
        Xf = super().transform(X)
        if isinstance(X, pd.DataFrame):
            Xf = pd.DataFrame(Xf, index=X.index, columns=X.columns)
        return Xf


class StandardScaler(preprocessing.StandardScaler):

    def transform(self, X, copy=None):
        Xf = super().transform(X)
        if isinstance(X, pd.DataFrame):
            Xf = pd.DataFrame(Xf, index=X.index, columns=X.columns)
        return Xf


class MinMaxScaler(preprocessing.MinMaxScaler):

    def transform(self, X):
        Xf = super().transform(X)
        if isinstance(X, pd.DataFrame):
            Xf = pd.DataFrame(Xf, index=X.index, columns=X.columns)
        return Xf


class RobustScaler(preprocessing.MinMaxScaler):

    def transform(self, X):
        Xf = super().transform(X)
        if isinstance(X, pd.DataFrame):
            Xf = pd.DataFrame(Xf, index=X.index, columns=X.columns)
        return Xf


class ColumnTransformer(compose.ColumnTransformer):

    def _hstack(self, Xs):
        try:
            # assumes Xs is DataFrame list
            if type(Xs) == list and all(isinstance(x, pd.DataFrame) for x in Xs):
                return pd.concat(Xs, axis=1)
            else:
                return super()._hstack(Xs)
        except RuntimeError:
            return super()._hstack(Xs)
