from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split

from pso_optimizer.pso import  PSOOptimizer


# Load sample dataset
load_digits= load_iris()
X = load_digits.data
y = load_digits.target

# Split dataset into train and test sets
optimizer = PSOOptimizer(estimator="DT",random_state=42, random_seed=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Example usage with KNN estimator
best_hyperparameters, best_score = optimizer.pso_hyperparameter_optimization(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, num_iterations=50, num_particles=100, c1=2, c2=2)

print(f"Best Score: {best_score}")

optimizer.get_hyperparameters(best_hyperparameters=best_hyperparameters)

report = optimizer.get_report(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, best_hyperparameters=best_hyperparameters)

print(f"Classification Report: {report}")