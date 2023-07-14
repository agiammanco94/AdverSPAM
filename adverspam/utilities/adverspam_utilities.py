import numpy as np
import adverspam.classifiers.classifier as classifier


def euclidean_distance(a: np.ndarray, b: np.ndarray):
    """
        This utility function computes the euclidean distance among two input vectors.

        Args:
            a: The first input array.

            b: The second input array.

        Returns:
            dist: The euclidean distance between the two input vectors.
    """
    if not isinstance(a, np.ndarray):
        a = np.ndarray(a)
    if not isinstance(b, np.ndarray):
        b = np.ndarray(b)
    return np.linalg.norm(a-b)


def project_point_on_line(point: np.array, line: np.array) -> np.array:
    """
        Utility function for obtaining the orthogonal projection of a point into a line.

        Args:
            point: A numpy array representing the point to project.

            line: The line in its implicit form.

        Returns:
            projection: The projected point over the input line.
    """
    x = np.array(point)
    direction_vector = line[:-1]

    c = line_implicit_to_point(line, len(direction_vector))
    # Determine if the point lies above or below the line
    # using the dot product
    vector_from_point_to_line = x - c

    # Set the sign of the projection depending on
    # whether the point lies above or below the line
    sign = -1

    projection = x + sign * np.dot(vector_from_point_to_line, direction_vector) / np.dot(direction_vector,
                                                                                         direction_vector) * direction_vector

    return projection


def line_implicit_to_point(line_equation: np.array, dimensions: int) -> np.array:
    """
        Converts an implicit line equation to a corresponding point in a given number of dimensions.

        Args:
            line_equation: The implicit line equation represented as a 1D numpy array.
                                      The last element of the array is the intercept of the line.
                                      The array's shape must be (dimensions + 1,).
            dimensions: The number of dimensions in the coordinate system.

        Returns:
            np.array: The corresponding point on the line in the coordinate system.

        Notes:
            - If the line equation is not valid (incorrect shape), None is returned.
            - If the line is degenerate (i.e., it represents a point), the point itself is returned.
            - The function assumes that the line equation is given in implicit form:
              a*x_1 + b*x_2 + ... + n*x_n + c = 0, where (x_1, x_2, ..., x_n) represents
              the coordinates of a point on the line.
    """
    intercept = line_equation[-1]
    # Check if the line equation is valid
    if line_equation.shape != (dimensions + 1,):
        return None

    # Check if the line is degenerate (i.e. it is a point)
    if np.count_nonzero(line_equation[:-1]) == 0:
        return tuple(line_equation[:-1])

    # Find the first non-zero coefficient in the line equation
    for i in range(dimensions):
        if line_equation[i] != 0:
            break

    # Set the point as the origin, except for the coordinate
    # corresponding to the first non-zero coefficient
    point = np.zeros(dimensions)

    point[i] = -1 * intercept / line_equation[i]

    return point


def new_adverspam_distance(adversarial: np.array, input_sample: np.array, lambda_param: float,
                           projection_on_psi: np.array) -> float:
    """
        This function implements the new distance measure introduced by our method AdverSPAM.
        The idea is to separate two distances for the adversarial sample: the distance from the original input,
        and the distance from its projection on the parallel to the decision boundary.

        Args:
            adversarial: The adversarial sample.

            input_sample: The input sample.

            lambda_param: The parameter used to control which of the two distances is more important. When lambda
                approaches 1, the euclidean distance from the input is predominant. Conversely, when lambda approaches
                0, the euclidean distance from the parallel to the decision boundary is predominant.

            projection_on_psi: The projection of the input sample over the parallel to the decision boundary.

        Returns:
            dist: The weighted euclidean distance.
    """
    f = euclidean_distance(adversarial, input_sample)
    g = euclidean_distance(adversarial, projection_on_psi)
    dist = lambda_param * f + (1 - lambda_param) * g
    return dist


def measure_percentage_of_negative_examples_included_in_psi(weights: np.array, intercept: float, psi: float,
                                                            positive_verse: bool, negative_examples: np.array) -> float:
    """
        Measures the percentage of negative examples included in a given Psi threshold above the decision boundary.

        Args:
            weights: The weights of the model.
            intercept: The intercept of the model.
            psi: The Psi threshold beyond the decision boundary.
            positive_verse: Flag indicating if the positive classification region lays in the positive semiplane
                according to the division made by the classifier decision boundary.
            negative_examples: The negative examples to be evaluated.

        Returns:
            float: The percentage of negative examples included in the Psi threshold.

        Notes:
            - The function iterates through each negative example and evaluates if it is included in the Psi threshold.
            - The inclusion criteria depends on the positive_verse flag.
                - If positive_verse is True, a negative example is included if the model output (test) is less than or
                    equal to 0.
                - If positive_verse is False, a negative example is included if the model output (test) is greater than
                    or equal to 0.
            - The function calculates the percentage of negative examples included in the Psi threshold by dividing
              the number of included negative examples by the total number of negative examples.
    """
    n_samples = len(negative_examples)
    n_samples_included_in_psi = 0
    for example in negative_examples:
        test = 0
        for weight, feature in zip(weights, example):
            test += (weight * feature)
        test += intercept
        test += psi

        if positive_verse and test <= 0:
            n_samples_included_in_psi += 1
        elif not positive_verse and test >= 0:
            n_samples_included_in_psi += 1
        else:
            continue
    percentage_of_negative_examples_included_in_psi = n_samples_included_in_psi / n_samples
    return percentage_of_negative_examples_included_in_psi


def estimating_psi_margin_beyond_decision_boundary(model: classifier.Classifier, positive_verse: bool,
                                                   negative_examples: np.array,
                                                   desired_ratio_of_included_samples: float = 0.10):
    """
        Estimates the Psi margin beyond the decision boundary for a given model and desired ratio of included samples.

        Args:
            model: The trained model.
            positive_verse: Flag indicating if the positive classification region lays in the positive semiplane
                according to the division made by the classifier decision boundary.
            negative_examples: The negative examples.
            desired_ratio_of_included_samples: The desired ratio of included samples in the Psi threshold.

        Returns:
            float: The estimated Psi margin beyond the decision boundary.

        Notes:
            - The function estimates the Psi margin beyond the decision boundary by incrementing the Psi value until the
              percentage of negative examples included in the Psi threshold reaches the desired ratio.
            - The Psi threshold is evaluated using the `measure_percentage_of_negative_examples_included_in_psi`
                function.
    """
    weights = model.model.coef_[0]
    intercept = model.model.intercept_

    psi = 0
    step = 0.01
    if positive_verse:
        step = -0.01
    percentage_of_negative_examples_included_in_psi = 0
    iteration = 0

    while percentage_of_negative_examples_included_in_psi < desired_ratio_of_included_samples:
        psi += step
        iteration += 1
        percentage_of_negative_examples_included_in_psi = \
            measure_percentage_of_negative_examples_included_in_psi(weights, intercept, psi, positive_verse,
                                                                    negative_examples)
    return psi
