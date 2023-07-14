import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from adverspam.classifiers.logistic import LogisticRegressionClassifier
from adverspam.adverspam_attack import adverspam_attack


SEED = 43

iris = datasets.load_iris()
# we restrict the dataset to the first 100 samples in order to deal with a binary classification problem
X = iris.data[:100, :]
y = iris.target[:100]
target_names = ['setosa', 'versicolor']
features_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# the features relative to the sepal are semantic dependant
sepal_features = ['sepal length (cm)', 'sepal width (cm)']
# the same holds for the features relative to the petal
petal_features = ['petal length (cm)', 'petal width (cm)']
blocks_of_semantic_dependent_features = [sepal_features, petal_features]


# we need to train a surrogate model for using it as source of adversarial samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

# a requirement of adverspam is to deal with surrogates having linear decision boundary
# we can thus use a logistic regression model
surrogate_model = LogisticRegressionClassifier(classifier_name='LR', C=1, dataset_name='IRIS',
                                               subset_of_features=features_names, seed=SEED)

surrogate_model.train(X_train, y_train)

# suppose we want to perturb the versicolor samples (label 1) in order to let the classifier recognize them as
# setosa samples (label 0)
y_start = 1  # versicolor
y_desired = 0  # setosa

target_test_positive_examples_mask = (y_test == y_start)
target_test_positive_examples = X_test[target_test_positive_examples_mask]

adversarial_examples, statistics_df = adverspam_attack(surrogate_model=surrogate_model,
                                                       feature_names=features_names,
                                                       train_x=X_train,
                                                       train_y=y_train,
                                                       test_x_positive_examples=target_test_positive_examples,
                                                       y_desired=y_desired,
                                                       y_start=y_start,
                                                       correction_for_correlation_constraints=0,
                                                       desired_ratio_of_included_samples=0.2,
                                                       blocks_of_semantic_dependent_features=
                                                       blocks_of_semantic_dependent_features,
                                                       lambda_param=0.5,
                                                       return_statistics=True)

# we now compare the performances of the surrogate model in recognizing the original versicolor samples,
# and the perturbed versicolor samples, that we want to be recognized as setosa samples
y_pred = surrogate_model.model.predict(target_test_positive_examples)
y_hat_pred = surrogate_model.model.predict(adversarial_examples)

accuracy_on_ground = np.count_nonzero(y_pred == 1) / y_pred.shape[0]
accuracy_on_adversarial = np.count_nonzero(y_hat_pred == 1) / y_pred.shape[0]

print(f'Accuracy on ground versicolor samples: {accuracy_on_ground}')
print(f'Accuracy on adversarial versicolor samples: {accuracy_on_adversarial}')
