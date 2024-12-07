import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from api.exception import CustomException


def save_object(file_path: str, obj: object) -> None:
    """
    Save an object to a file using pickle.

    Args:
        file_path (str): The path where the object will be saved.
        obj (object): The object to be saved.

    Raises:
        CustomException: If there's an error during the saving process.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, models: dict, param: dict) -> dict:
    """
    Evaluate multiple models using GridSearchCV and return their performance.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Testing features.
        y_test (np.ndarray): Testing labels.
        models (dict): Dictionary of models to evaluate.
        param (dict): Dictionary of parameters for GridSearchCV.

    Returns:
        dict: A report of model performance on the test set.

    Raises:
        CustomException: If there's an error during model evaluation.
    """
    try:
        report = {}

        for model_name, model in models.items():
            gs = GridSearchCV(model, param[model_name], cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str) -> object:
    """
    Load an object from a file using pickle.

    Args:
        file_path (str): The path to the file containing the object.

    Returns:
        object: The object loaded from the file.

    Raises:
        CustomException: If there's an error during the loading process.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
