# -*- coding: utf-8 -*-
"""
    References:
        [1] https://stackoverflow.com/questions/874017/python-load-words-from-file-into-a-set
"""
import datetime
import os
import sys
import subprocess
import time
from itertools import chain, combinations
from os import path
from typing import List
import numpy as np
import random


def get_relative_path() -> str:
    """
        Returns the prefix to come back to the root folder.

        Returns:
            prefix: A string representing the path to the root folder of the project.
    """

    s = str(subprocess.check_output(['pwd']))
    s = s[:-2].split("/")[1:]
    s[-1] = s[-1][:-1]

    prefix = "/"

    root_dir = 'AdverSPAM'
    while s[-1] != root_dir:
        prefix += '../'
        s = s[:-1]

    current_path = os.getcwd()

    return current_path + prefix


def get_current_timestamp(format_str: str = '%d-%m-%Y %H-%M-%S-%f') -> str:
    """
        Returns the current timestamp with the specified format.

        Args:
            format_str: A string representing the format for the current timestamp.

        Returns::
            now: a string representing the current timestamp according to the desired format.
    """
    now = str(datetime.datetime.now().strftime(format_str))
    return now


def print_with_timestamp(message: str, file_path: str = None, format_str: str = '%d-%m-%Y %H-%M-%S-%f') -> None:
    """
        Utility function to print a string with the current date and time.

        Args:
            message: A string representing the message to output with the current timestamp.

            file_path: A string representing the file path on which to redirect the stdout.

            format_str: A string representing the format for the current timestamp.
    """
    import sys
    ts = get_current_timestamp(format_str=format_str)
    if file_path is None or sys.gettrace() is not None:  # debug mode
        print(f'({ts}) - ', end='')
        print(message)
    else:
        with open(file_path, 'a') as sys.stdout:
            print(f'({ts}) - ', end='')
            print(message)


def get_subdirs(dir_path: str) -> list:
    """
        Returns the list of subdirectories inside the input folder.

        Args:
            dir_path: String representing the directory to navigate.

        Returns:
            subdirs: The list of subdirectories in the input dir_path.
    """
    subdirs = [x for x in os.listdir(dir_path) if os.path.isdir(dir_path + '/' + x)]
    return subdirs


def create_dir(dir_path: str) -> None:
    """
        Utility function which creates a directory if it is not already present in the system.

        Args:
            dir_path: A string representing the path of the directory to create.
    """
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)


def get_list_of_paths(dir_path: str) -> List[str]:
    """
        Given a certain directory, this function returns a list of all the sub-directories contained inside it.

        Args:
            dir_path: A string representing the path of the directory to scan.

        Returns:
            list_of_paths: A list of strings containing the paths of the sub-directories inside dir_path.
    """
    list_of_paths = [dir_path + '/' + x for x in os.listdir(dir_path) if os.path.isdir(dir_path + '/' + x)]
    return list_of_paths


def all_subsets(ss, max_size=5):
    x = list(chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1))))
    res = [list(i) for i in x if len(i) <= max_size and len(i) > 0]
    return res


def from_list_of_strings_to_string(list_of_strings: List[str], separator: str = '-') -> str:
    """
        Returns a string concatenating all the strings contained in the first parameter list.

        Args:
            list_of_strings: a list of strings to convert in a single string.
            separator: a string which will be used as separator for all the strings contained in the first parameter.

        Returns:
            result: a string concatenating all the strings in the first parameter, separated with the second parameter.
    """
    result = separator.join(list_of_strings)
    return result


def exists(filename: str) -> bool:
    """
        Utility function which checks if directory exists or not in the system.

        Args:
            filename: A string representing the path of the directory to create.
    """
    return path.exists(filename)


def get_results_path() -> str:
    """
        Returns the path to the results folder.

        Returns::
            results_path: A string representing the path to the results folder of the project.
    """
    prefix = get_relative_path()
    results_path = prefix + 'results'
    create_dir(results_path)
    return results_path


def debugger_is_active() -> bool:
    """
        Return True if the debugger is currently active.
    """
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None


def normalize_array(arr: np.array) -> np.array:
    """
        This function normalize an input vector by columns, so that all the values lie in the range [0, 1].

        Args:
            arr: A numpy input array.

        Returns::
            normalized: A numpy array holding the normalized values in the range [0, 1].
    """
    # get minimum and maximum values of each column
    min_vals = np.min(arr, axis=0)
    max_vals = np.max(arr, axis=0)

    # subtract minimum value from each column and divide by the range
    normalized = (arr - min_vals) / (max_vals - min_vals)

    return normalized


def standardize_array(arr: np.array) -> np.array:
    """
    This function standardizes an input vector by columns, so that all the values have zero mean and unit variance.

    Args:
        arr: A numpy input array.

    Returns:
        standardized: A numpy array holding the standardized values with zero mean and unit variance.
    """
    # calculate mean and standard deviation of each column
    mean_vals = np.mean(arr, axis=0)
    std_vals = np.std(arr, axis=0)

    # subtract mean and divide by standard deviation
    standardized = (arr - mean_vals) / std_vals

    return standardized


def is_array_in_list(arr_to_check: np.array, list_of_arrays: list) -> tuple:
    """
        This function checks if a numpy array is contained inside a list of numpy arrays.

        Args:
            arr_to_check: A numpy array which we need to check if it is contained in a list.

            list_of_arrays: A list of numpy arrays to check for membership.

        Returns::
            is_contained: A boolean which if True signals that the parameter arr_to_check is contained inside the
                list_of_arrays passed as input.

            equal_indices_flags: A list of boolean flags which signal whether the indexed element in the list_of_arrays
                is equal to the input arr_to_check.
    """
    equal_indices_flags = [np.array_equal(arr_to_check, arr) for arr in list_of_arrays]
    is_contained = any(equal_indices_flags)
    return is_contained, equal_indices_flags


def shuffle_samples(samples: np.array, seed: int) -> np.array:
    """
        This function shuffle the input samples according to a seed.
        Args:
            samples: the numpy array to shuffle.
            seed: the seed for prng.

        Returns:
            shuffled_samples: the shuffled numpy array.
    """
    random.seed(seed)
    shuffled_samples = np.copy(samples)
    current_indices = [i for i in range(samples.shape[0])]
    random.shuffle(current_indices)
    shuffled_samples = shuffled_samples[current_indices]
    return shuffled_samples


class Timer:
    """
        This class models a simple timer, which can be used to monitor the execution time of code fragments.

        Attributes:
            start_time: A float representing the number of seconds elapsed from "epoch", i.e.,
                January 1, 1970, 00:00:00, when the timer is started.

            end_time: A float representing the number of seconds elapsed from "epoch", when the timer is ended.

            seconds_elapsed: A float representing the amount of seconds elapsed between the call to start and
                stop methods.

            minutes_elapsed: A float representing the amount of minutes elapsed between the call to start and
                stop methods.
    """
    def __init__(self) -> None:
        """
            Inits a Timer instance with 0 values for the attributes, and then starts it.
        """
        self.start_time = 0
        self.end_time = 0
        self.seconds_elapsed = 0
        self.minutes_elapsed = 0
        self.start()

    def start(self) -> None:
        """
            Starts the Timer recording the current time.
        """
        self.start_time = time.time()

    def stop(self) -> float:
        """
            Stops the Timer, returning the amount of minutes elapsed.

            Returns::
                minutes_elapsed: A float representing the amount of minutes elapsed between the start and stop of the
                    timer.
        """
        self.end_time = time.time()
        self.seconds_elapsed = (self.end_time - self.start_time)
        self.minutes_elapsed = self.seconds_elapsed / 60
        return self.minutes_elapsed
