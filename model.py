# -*- coding: utf-8 -*-

# Guardando el modelo usando pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv("banking.csv")
dataframe=data[['job','balance','age','marital','y']]

# Dividiendo la data en conjuntos train y test 
X_train, X_test, y_train ,y_test = train_test_split(
    dataframe.drop(columns = ['y']),
    dataframe['y'],
    test_size=0.25,
    random_state=42,
    stratify=dataframe["y"]
)

# Definiendo el estimador de apilamiento

from sklearn.base import BaseEstimator, TransformerMixin, is_classifier
from sklearn.utils import check_array
class StackingEstimator(BaseEstimator, TransformerMixin):

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):

        self.estimator.fit(X, y, **fit_params)
        return self

    def transform(self, X):

        X = check_array(X)
        X_transformed = np.copy(X)
        # agregar probabilidades de clase como una característica sintética

        if is_classifier(self.estimator) and hasattr(self.estimator, 'predict_proba'):
            y_pred_proba = self.estimator.predict_proba(X)
            # check all values that should be not infinity or not NAN
            if np.all(np.isfinite(y_pred_proba)):
                X_transformed = np.hstack((y_pred_proba, X))

        # agregar predicción de clase como una característica sintética
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))
        return X_transformed

# Creando el pipeline

from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
st = StackingEstimator(estimator=DecisionTreeClassifier(max_depth=10,
                                                   min_samples_leaf=11,
                                                   min_samples_split=4,
                                                   random_state=42))
gnb = GaussianNB()

pipeline = make_pipeline(st, gnb)

# Ajuste el modelo en el conjunto de entrenamiento(train)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)


# Guardar el modelo en el disco con pickle
filename = 'model.pkl'
#pickle.dump(pipeline, open('model.pkl','wb'))
joblib.dump(pipeline, filename)


# algunos minutos despúes...

#Cargar el modelo desde el disco con pickle

#pipeline = pickle.load('model.pkl')
