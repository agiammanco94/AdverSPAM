# -*- coding: utf-8 -*-
import math
import time
from scipy.stats import linregress
from scipy.optimize import minimize, LinearConstraint
import pandas as pd
import numpy as np
from typing import List, Tuple
import adverspam.classifiers.classifier as classifier
from adverspam.utilities.adverspam_utilities import project_point_on_line, new_adverspam_distance, \
    estimating_psi_margin_beyond_decision_boundary


def adverspam_attack(surrogate_model: classifier.Classifier,
                     feature_names: List[str],
                     train_x: np.ndarray,
                     train_y: np.ndarray,
                     test_x_positive_examples: np.ndarray,
                     y_desired: int,
                     y_start: int,
                     desired_ratio_of_included_samples: float,
                     blocks_of_semantic_dependent_features: List[List[str]],
                     correction_for_correlation_constraints: float = 0,
                     lambda_param: float = 0.5,
                     return_statistics: bool = False) -> Tuple[np.ndarray, pd.DataFrame]:
    """
        Params:
            surrogate_model: A Classifier object representing the surrogate model for crafting the adversarial examples.

            feature_names: A list of string containing the name of the features in the dataset.

            train_x: The set of train_x is useful for computing the linear regression for the positive examples.

            train_y: The same holds for the corresponding labels.

            test_x_positive_examples: This is the set of input examples that we want to perturb.
                The associated labels of this set of examples is a list [y_start, y_start, ...]

            y_desired: The label of the desired class.

            y_start: The label of the positive class which we want to be flipped.

            desired_ratio_of_included_samples: This is the ratio of samples desired to include between psi line and
                the decision boundary.

            blocks_of_semantic_dependent_features: This is a list of lists, where every sublist contains the string
                representing a block of features which stands in a semantic relationship dependence between them.
                Two features are defined as semantic dependent when there is a particular relationship between them,
                for example, in the paper we suppose this relationship to consist in the sharing of a raw variable
                in the computations of two semantic dependent features.

            correction_for_correlation_constraints: This parameter shrinks (if negative) or widens (if positive)
                the constraints relative to the correlation by acting on the margin surrounding the regression line.
                This parameter is optional and its default value is zero, meaning that the margin around the regression
                line is set to the TSS as in the original paper. If there are no good results for the particular dataset
                at hand, modifying this value can help in having better results, as the admissible region may be too
                tight (or too wide).

            lambda_param: The parameter used to control which of the two distances is more important. When lambda
                approaches 1, the euclidean distance from the input is predominant. Conversely, when lambda approaches
                0, the euclidean distance from the parallel to the decision boundary is predominant.

            return_statistics: Returns a dataframe with some statistics about the attack.

        Returns:
            adversarial_examples: The set of crafted perturbed examples. If the constraint satisfaction problem is
                infeasible for a single input sample, that particular sample is copied from the ground truth inside
                the set of adversarial examples.

            statistics_df: A dataframe containing some information on the attack.
    """
    adversarial_examples = []
    n_features = len(feature_names)

    train_positive_examples_mask = (train_y == y_start)
    if isinstance(train_positive_examples_mask, pd.DataFrame):
        train_positive_examples_mask = train_positive_examples_mask.to_numpy().flatten().tolist()
    elif isinstance(train_positive_examples_mask[0], list) or isinstance(train_positive_examples_mask, np.ndarray):
        train_positive_examples_mask = train_positive_examples_mask.flatten()
    train_positive_examples = train_x[train_positive_examples_mask]

    train_negative_examples_mask = (train_y == y_desired)
    if isinstance(train_negative_examples_mask, pd.DataFrame):
        train_negative_examples_mask = train_negative_examples_mask.to_numpy().flatten().tolist()
    elif isinstance(train_negative_examples_mask[0], list) or isinstance(train_negative_examples_mask, np.ndarray):
        train_negative_examples_mask = train_negative_examples_mask.flatten()
    train_negative_examples = train_x[train_negative_examples_mask]

    # Precomputations for saving time
    TSS_for_positive_examples = []
    for feature_idx in range(len(feature_names)):
        mu_feature = train_positive_examples[:, feature_idx].mean()
        TSS = 0
        for sample in train_positive_examples[:, feature_idx]:
            TSS += ((sample - mu_feature) ** 2)
        TSS_for_positive_examples.append(TSS)

    psi_for_negative_examples = dict()
    for feature_i in feature_names:
        psi_for_negative_examples[feature_i] = dict()
        for feature_j in feature_names:
            if feature_i == feature_j:
                continue
            psi_for_negative_examples[feature_i][feature_j] = None

    statistics_df_list_of_dicts = []

    model_weights = surrogate_model.model.coef_[0]
    model_intercept = surrogate_model.model.intercept_[0]

    test_for_semiplane = 0
    for weight, feature in zip(model_weights, test_x_positive_examples[0]):
        test_for_semiplane += (weight * feature)
    test_for_semiplane += model_intercept

    y_hat = surrogate_model.predict([test_x_positive_examples[0]])[0]

    # positive_verse is True if the semiplane holding the desired region is the positive one... in our example is False
    positive_verse = None
    if test_for_semiplane >= 0:
        if y_hat != y_desired:  # the correct semiplane is the left one
            positive_verse = False
        elif y_hat == y_desired:  # the correct semiplane is the right one
            positive_verse = True
    else:
        if y_hat != y_desired:  # the correct semiplane is the right one
            positive_verse = True
        elif y_hat == y_desired:  # the correct semiplane is the left one
            positive_verse = False

    psi = estimating_psi_margin_beyond_decision_boundary(surrogate_model, positive_verse,
                                                         train_negative_examples,
                                                         desired_ratio_of_included_samples)

    for input_sample_idx, input_sample in enumerate(test_x_positive_examples):

        statistics_for_input_sample = dict()
        statistics_for_input_sample['Sample #'] = input_sample_idx
        statistics_for_input_sample['Feature vector'] = str(input_sample)
        statistics_for_input_sample['psi'] = psi
        for i, feature_name in enumerate(feature_names):
            statistics_for_input_sample[f'x_{feature_name}'] = input_sample[i]
        start_time_of_processing = time.time()

        # coefficients of linear inequality on the decision variables; in our case, the \tilde{x}
        A = []
        # the inequality upper bounds
        b = []

        # Decision boundary constraint

        decision_boundary_constraint = np.zeros(n_features)
        for i in range(len(input_sample)):
            decision_boundary_constraint[i] = model_weights[i]
        # the half-plane containing the desired region is the positive one -> major constraint ->
        # the sign of the weights must be inverted
        if positive_verse:
            decision_boundary_constraint = [-1 * x for x in decision_boundary_constraint]
        A.append(decision_boundary_constraint)

        # Psi constraint

        coefficient_for_strict_minor_boundary = 0.001
        psi_coefficient_for_strict_minor_boundary = 2 * coefficient_for_strict_minor_boundary

        if psi < 0:
            coefficient_for_strict_minor_boundary *= -1
            psi_coefficient_for_strict_minor_boundary *= -1

        psi_constraint = np.zeros(n_features)
        psi_constraint_sign = 1
        for i in range(len(input_sample)):
            psi_constraint[i] = psi_constraint_sign * model_weights[i]
        # if the desired half-plane is the negative one, I draw a parallel going down... I want the opponent's sample to
        # be above the parallel -> major constraint -> I have to reverse the signs
        if not positive_verse:
            psi_constraint = [-1 * x for x in psi_constraint]
        A.append(psi_constraint)

        # Decision boundary b
        b_decision_boundary = -1 * (model_intercept + coefficient_for_strict_minor_boundary)
        if positive_verse:
            b_decision_boundary *= -1
        b.append(b_decision_boundary)

        # Psi b
        b_psi = -1 * (model_intercept + psi_coefficient_for_strict_minor_boundary) - psi
        b_psi *= psi_constraint_sign
        if isinstance(b_psi, np.ndarray):
            b_psi = b_psi[0]
        if not positive_verse:
            b_psi *= -1
        b.append(b_psi)

        line_describing_psi = list(psi_constraint)
        line_describing_psi.append(-1 * b_psi)
        line_describing_psi = np.array(line_describing_psi)

        for i, feature_i in enumerate(feature_names):

            for j, feature_j in enumerate(feature_names):

                if feature_i == feature_j:
                    continue

                exist_semantic_correlation = False
                for block_of_features in blocks_of_semantic_dependent_features:
                    if feature_i in block_of_features and feature_j in block_of_features:
                        exist_semantic_correlation = True
                        break

                if not exist_semantic_correlation:
                    continue

                linear_regression = linregress(train_positive_examples[:, i], train_positive_examples[:, j])
                alpha = linear_regression.slope
                beta = linear_regression.intercept
                r_square = linear_regression.rvalue ** 2

                TSS_j = TSS_for_positive_examples[j]
                margin_j_i = math.sqrt(abs(1 - r_square) * TSS_j)

                #################
                # A computation #
                #################

                # Correlation constraints
                correlation_constraint_1 = np.zeros(n_features)
                correlation_constraint_1[i] = -1 * alpha
                correlation_constraint_1[j] = 1
                A.append(correlation_constraint_1)

                correlation_constraint_2 = np.zeros(n_features)
                correlation_constraint_2[i] = alpha
                correlation_constraint_2[j] = -1
                A.append(correlation_constraint_2)

                # Domain constraints
                domain_constraint_1 = np.zeros(n_features)
                domain_constraint_1[j] = 1
                A.append(domain_constraint_1)

                domain_constraint_2 = np.zeros(n_features)
                domain_constraint_2[j] = -1
                A.append(domain_constraint_2)

                #################
                # b computation #
                #################

                # Correlation b
                b_correlation_1 = beta + (margin_j_i + correction_for_correlation_constraints)
                b.append(b_correlation_1)

                b_correlation_2 = -1 * beta + (margin_j_i + correction_for_correlation_constraints)
                b.append(b_correlation_2)

                # Domain b
                b_domain_1 = 1
                b.append(b_domain_1)

                b_domain_2 = 0
                b.append(b_domain_2)

        A = np.array(A)
        b = np.array(b)

        constraints = []
        for i in range(len(A)):
            a = A[i]
            ub = b[i]
            lb = -np.inf
            cons = LinearConstraint(a, ub=ub, lb=lb)
            constraints.append(cons)

        n = len(feature_names)
        projection_on_psi = project_point_on_line(input_sample, line_describing_psi)

        res = minimize(new_adverspam_distance, input_sample, method='COBYLA',
                       constraints=constraints,
                       args=(input_sample, lambda_param, projection_on_psi))

        if res is not None and res.success:
            adversarial_example = res.x
        else:
            adversarial_example = input_sample
        adversarial_examples.append(adversarial_example)

        end_time_of_processing = time.time()
        time_of_processing = end_time_of_processing - start_time_of_processing
        statistics_for_input_sample['Seconds for processing'] = time_of_processing

        try:
            statistics_for_input_sample['# Iterations for Optimization Problem'] = res.nit if res is not None else -1
        except AttributeError:
            statistics_for_input_sample['# Iterations for Optimization Problem'] = -1
        statistics_for_input_sample['Success'] = res.success if res is not None else False

        statistics_for_input_sample['Adversarial example'] = str(adversarial_example)
        for i, feature_name in enumerate(feature_names):
            statistics_for_input_sample[f'x_adv_{feature_name}'] = adversarial_example[i]

        delta = adversarial_example - input_sample
        statistics_for_input_sample['Delta'] = str(delta)
        for i, feature_name in enumerate(feature_names):
            statistics_for_input_sample[f'delta_{feature_name}'] = delta[i]

        statistics_df_list_of_dicts.append(statistics_for_input_sample)

    adversarial_examples = np.array(adversarial_examples)
    statistics_df = pd.DataFrame(statistics_df_list_of_dicts)
    if return_statistics:
        return adversarial_examples, statistics_df
    else:
        return adversarial_examples

