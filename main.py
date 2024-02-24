from sklearn.datasets import load_iris

import pso_optimizer.pso as pso

# Load sample dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Example usage with KNN estimator
best_hyperparameters, best_score = pso.pso_hyperparameter_optimization(estimator="DT", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, num_iterations=50, num_particles=100, c1=2, c2=2)

print(f"Best Score: {best_score}")

pso.evaluate_model(estimator="DT", best_hyperparameters=best_hyperparameters)



