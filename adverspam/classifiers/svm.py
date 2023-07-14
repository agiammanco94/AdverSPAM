# -*- coding: utf-8 -*-
"""
    References:
        [1] https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
"""

from typing import Union, List

import numpy as np
from sklearn.svm import SVC, SVR

import adverspam.classifiers.classifier as classifier
import adverspam.utilities.miscellaneous as misc_util


class SupportVectorMachine(classifier.Classifier):

    def __init__(self, classifier_name: str, C: float = 1.0, kernel: str = 'rbf', gamma: Union[str, float] = 'scale',
                 tol: float = 1e-3, dataset_name: str = None, subset_of_features: List[str] = None,
                 seed: int = 42, probability: bool = True, regressor: bool = False, debug_path: str = None) -> None:
        """
            Params are taken verbatim from [1]

            Args:
                C: Regularization parameter. The strength of the regularization is inversely proportional to C.
                    Must be strictly positive. The penalty is a squared l2 penalty.

                kernel: Specifies the kernel type to be used in the algorithm.
                    It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable.

                gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.

                tol: Tolerance for stopping criterion.

                seed: An integer representing the random seed for obtaining deterministic procedures.

                regressor: A boolean which if True instantiates a Regressor instead of a Classifier.

        """
        super().__init__()
        self.classifier_name = classifier_name
        self.C = C
        self.kernel = kernel
        self.dataset_name = dataset_name
        self.subset_of_features = subset_of_features
        self.gamma = gamma
        if not regressor:
            self.model = SVC(kernel=kernel, gamma=gamma, C=C, tol=tol, random_state=seed, probability=probability)
        else:
            self.model = SVR(kernel=kernel, gamma=gamma, C=C, tol=tol)
        self.debug_path = debug_path

    def __str__(self) -> str:
        return f'Support Vector Machine with {self.kernel} kernel, C={self.C}, gamma={self.gamma}'

    def train(self, x_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        misc_util.print_with_timestamp(f"Training {self.brief_description()}...", file_path=self.debug_path)
        training_set_hash = self.get_hash_of_train_set(x_train)
        if not self.load_trained_model(training_set_hash=training_set_hash):
            self.model.fit(x_train, y_train)
            self.save_trained_model(training_set_hash=training_set_hash)
        self.trained = True

    def predict(self, x_test, **kwargs) -> np.ndarray:
        try:
            prediction = self.model.predict(x_test)
        except ValueError:
            return np.array([-999])
        return prediction
