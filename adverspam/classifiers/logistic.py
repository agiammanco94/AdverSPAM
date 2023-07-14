# -*- coding: utf-8 -*-
"""
    References:
        [1] https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
"""
from typing import List

import numpy as np
from sklearn.linear_model import LogisticRegression

import adverspam.classifiers.classifier as classifier
import adverspam.utilities.miscellaneous as misc_util


class LogisticRegressionClassifier(classifier.Classifier):

    def __init__(self, classifier_name: str, C: float = 1.0, dataset_name: str = None,
                 subset_of_features: List[str] = None, seed: int = 42, debug_path: str = None,
                 debug_flag: bool = True) -> None:
        """
            Params are taken verbatim from [1]

            Args:
                C: Inverse of regularization strength; must be a positive float. Like in support vector machines,
                    smaller values specify stronger regularization.

                seed: An integer representing the random seed for obtaining deterministic procedures.

        """
        super().__init__()
        self.C = C
        self.dataset_name = dataset_name
        self.subset_of_features = subset_of_features
        self.model = LogisticRegression(C=C, random_state=seed)
        self.debug_path = debug_path
        self.classifier_name = classifier_name
        self.debug_flag = debug_flag

    def __str__(self) -> str:
        return f'Logistic Regression with {self.C} regularization factor.'

    def train(self, x_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        if self.debug_flag:
            misc_util.print_with_timestamp(f"Training {self.brief_description()}...", file_path=self.debug_path)
        training_set_hash = self.get_hash_of_train_set(x_train)
        if not self.load_trained_model(training_set_hash=training_set_hash):
            self.model.fit(x_train, y_train)
            self.save_trained_model(training_set_hash=training_set_hash)
        self.trained = True

    def predict(self, x_test, y_test=None, **kwargs) -> ...:
        try:
            prediction = self.model.predict(x_test)
        except ValueError:
            return np.array([-999])
        return prediction

