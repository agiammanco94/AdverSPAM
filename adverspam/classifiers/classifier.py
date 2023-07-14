# -*- coding: utf-8 -*-
import hashlib
import pickle
from os.path import exists

import numpy as np

import adverspam.utilities.miscellaneous as misc_util


class Classifier:
    """
        This abstract class models a generic Classifier.
    """

    def __init__(self) -> None:
        self.model = None
        self.trained = False
        self.subset_of_features = None
        self.dataset_name = None
        self.classifier_name = None

    def __str__(self) -> str:
        return "Generic Classifier"

    def train(self, train_x, train_y, **kwargs) -> None:
        pass

    def predict(self, test_x, **kwargs) -> np.ndarray:
        pass

    def predict_proba(self, test_x, **kwargs) -> np.ndarray:
        pass

    def brief_description(self) -> str:
        return self.classifier_name

    @staticmethod
    def get_trained_models_path() -> str:
        prefix = misc_util.get_relative_path()
        results_path = f'{prefix}adverspam/classifiers/trained_classifiers'
        misc_util.create_dir(results_path)
        return results_path

    def get_classifier_path(self, dataset_name: str, **kwargs) -> str:
        results_path = self.get_trained_models_path()
        dataset_path = f'{results_path}/{dataset_name}'
        misc_util.create_dir(dataset_path)
        dataset_path = f'{dataset_path}/all'
        misc_util.create_dir(dataset_path)
        if 'training_set_hash' in kwargs:
            dataset_path = f'{dataset_path}/{kwargs["training_set_hash"]}'
            misc_util.create_dir(dataset_path)
        model_path = f'{dataset_path}/{self.brief_description()}'
        misc_util.create_dir(model_path)
        return model_path

    def save_trained_model(self, training_set_hash: str) -> None:
        model_path = self.get_classifier_path(self.dataset_name, training_set_hash=training_set_hash)
        pickle.dump(self.model, open(model_path + '/model.pkl', "wb"))

    def load_trained_model(self, training_set_hash: str) -> bool:
        model_path = self.get_classifier_path(self.dataset_name, training_set_hash=training_set_hash)
        model_f_name = model_path + '/model.pkl'
        if exists(model_f_name):
            misc_util.print_with_timestamp(f'Loading pre-trained {self.brief_description()}')
            with open(model_f_name, "rb") as f:
                self.model = pickle.load(f)
            self.trained = True
            return True
        else:
            return False

    @staticmethod
    def get_hash_of_train_set(training_x: np.array) -> str:
        return hashlib.sha256(np.ascontiguousarray(training_x)).hexdigest()
