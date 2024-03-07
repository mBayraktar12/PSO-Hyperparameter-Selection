import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from pso_optimizer.hyperparameter_mappings import (kernel_map,
                                                   gamma_map,
                                                   shrinking_map,
                                                   decision_function_shape_map,
                                                   algorithm_map,
                                                   criterion_map_dt,
                                                   criterion_map_rf,
                                                   distance_metric_map,
                                                   max_features_map_dt,
                                                   max_features_map_rf,
                                                   min_samples_split_map_dt,
                                                   min_samples_split_map_rf,
                                                   splitter_map,
                                                   weighting_method_map)

class PSOOptimizer:

    def __init__(self, estimator, random_state, random_seed):
        self.estimator = estimator
        self.random_state = random_state
        self.random_seed = random_seed
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        print("\n*** Default Values for PSO Optimization ***")
        print(f"The default value for c1 and c2 is 2.05, and for w is 0.72894 according to the paper 'The Particle Swarm — Explosion, Stability, and Convergence in a Multidimensional Complex Space' by Clerc and Kennedy.\n")

    def pso_hyperparameter_optimization(self, X_train, X_test, y_train, y_test, num_particles, num_iterations, c1 = 2.05, c2 = 2.05, num_jobs=-1, w=0.72984):
        """
        Perform hyperparameter optimization using Particle Swarm Optimization (PSO).

        Parameters:
            - estimator: The estimator object (e.g., KNeighborsClassifier).
            - data: The dataset.
            - target_column_index: Index of the target column in the dataset.
            - num_particles: Number of particles in the population.
            - num_iterations: Number of iterations for the PSO algorithm.
            - c1: Acceleration constant. Default value is c1 = 2.05
            - c2: Acceleration constant. Default value is c2 = 2.05
            - num_jobs: Number of parallel jobs for fitness evaluation.
            - inertia weight: Inertia constant. Default value is w=0.72984 according to the paper by M. Clerc and J. Kennedy

        Returns:
            - global_best_position: The best set of hyperparameters found.
            - global_best_fitness: The best accuracy found.
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        hyperparameter_space = self._get_hyperparameter_space()

        progress_bar = tqdm(total=num_iterations, desc="PSO Progress")

        # Initialize the population of particles
        population = []
        for _ in range(num_particles):
            hyperparameters = [np.random.choice(hyperparameter_space[param]) for param in hyperparameter_space]
            population.append(hyperparameters)

        # Initialize velocity and best position
        velocity = [[0] * len(hyperparameter_space) for _ in range(num_particles)]
        best_position = population.copy()
        global_best_fitness = -float("inf")
        global_best_position = []

        # PSO optimization loop
        for _ in range(num_iterations):
            fitness = Parallel(n_jobs=num_jobs)(
                delayed(self.evaluate_fitness)(X_train, X_test, y_train, y_test, particle)
                for particle in population
            )

            for j, particle in enumerate(population):
                if fitness[j] > self.evaluate_fitness(X_train, X_test, y_train, y_test, best_position[j]):
                    best_position[j] = particle

            if max(fitness) > global_best_fitness:
                global_best_fitness = max(fitness)
                global_best_position = population[fitness.index(max(fitness))]

            for j, particle in enumerate(population):
                r1 = np.random.uniform(0, 1)
                r2 = np.random.uniform(0, 1)
                velocity[j] = [w * velocity[j][k] + c1 * r1 * (best_position[j][k] - particle[k]) + c2 * r2 * (global_best_position[k] - particle[k]) for k in range(len(hyperparameter_space))]

                for k in range(len(hyperparameter_space)):
                    particle[k] += velocity[j][k]
                    # particle[k] = max(min(particle[k], max(hyperparameter_space[param])), min(hyperparameter_space[param]))
                    particle[k] = max(min(particle[k], max(hyperparameter_space[list(hyperparameter_space.keys())[k]])), 
                        min(hyperparameter_space[list(hyperparameter_space.keys())[k]]))
        

            # Update progress bar
            progress_bar.update(1)

            # w = self.update_inertia_weight(w=w)

        # Close progress bar
        progress_bar.close()

        return global_best_position, global_best_fitness

    # def update_inertia_weight(self, w):

    #     return w - 0.01 # Linear decay
    
    def evaluate_fitness(self, X_train, X_test, y_train, y_test, hyperparameters):
        """
        Evaluate the fitness of a set of hyperparameters.

        Parameters:
            - estimator: The estimator object.
            - X_train: Training features.
            - X_test: Testing features.
            - y_train: Training labels.
            - y_test: Testing labels.
            - hyperparameters: The set of hyperparameters to evaluate.

        Returns:
            - score: The accuracy score of the estimator with the given hyperparameters.
        """
        # Unpack hyperparameters
        estimator_instance = self._create_estimator(hyperparameters)

        estimator_instance.fit(X_train, y_train)
        y_pred = estimator_instance.predict(X_test)
        accuracy_pso = accuracy_score(y_test, y_pred)
        return accuracy_pso

    def _create_estimator(self, hyperparameters):
        # Train the estimator using the hyperparameters
        estimator_instance = None
        if self.estimator == "KNN":
            k, distance_metric, weighting_method, algorithm, leaf_size, p_val = hyperparameters
            estimator_instance = KNeighborsClassifier(n_neighbors=int(k), metric=distance_metric_map[round(distance_metric)],
                                                    weights=weighting_method_map[round(weighting_method)],
                                                    algorithm=algorithm_map[round(algorithm)],
                                                    leaf_size=int(leaf_size), p=int(p_val))
        elif self.estimator == "RF":
            n_estimators_values, max_depth_values, criterion_values, min_samples_split_values, min_samples_leaf_values, min_weight_fraction_leaf_values, max_features_values = hyperparameters
            estimator_instance = RandomForestClassifier(random_state = self.random_state, n_estimators=int(n_estimators_values), max_depth=int(max_depth_values),
                                                        criterion=criterion_map_rf[round(criterion_values)], 
                                                        min_samples_split= min_samples_split_map_rf[round(min_samples_split_values)],
                                                        min_samples_leaf=int(min_samples_leaf_values),
                                                        min_weight_fraction_leaf = min_weight_fraction_leaf_values,
                                                        max_features=max_features_map_rf[round(max_features_values)]
                                                        )

        elif self.estimator == "DT":
            splitter_values, max_depth_values, criterion_values, min_samples_split_values, min_samples_leaf_values, min_weight_fraction_leaf_values, max_features_values = hyperparameters
            estimator_instance = DecisionTreeClassifier(random_state = self.random_state, splitter=splitter_map[round(splitter_values)], max_depth=int(max_depth_values),
                                                        criterion=criterion_map_dt[round(criterion_values)],min_samples_split=min_samples_split_map_dt[round(min_samples_split_values)],
                                                        min_samples_leaf=int(min_samples_leaf_values), 
                                                        min_weight_fraction_leaf=min_weight_fraction_leaf_values, max_features=max_features_map_dt[round(max_features_values)])
        elif self.estimator == "SVC":
            c, kernel_values, degree_values, gamma_values, shrinking_values, decision_function_shape_values = hyperparameters
            estimator_instance = SVC(random_state = self.random_state, C=int(c), kernel=kernel_map[round(kernel_values)],
                                    degree=int(degree_values), shrinking=shrinking_map[round(shrinking_values)], gamma=gamma_map[round(gamma_values)],
                                    decision_function_shape=decision_function_shape_map[round(decision_function_shape_values)])
        else:
            raise ValueError("Estimator not supported.")

        return estimator_instance

    def get_hyperparameters(self, best_hyperparameters):

        if self.estimator == "KNN":
            print("Optimal hyperparameters:")
            print(f"K: {best_hyperparameters[0]}")
            print(f"Distance Metric: {distance_metric_map[best_hyperparameters[1]]}")
            print(f"Weighting Method: {weighting_method_map[best_hyperparameters[2]]}")
            print(f"Algorithm: {algorithm_map[best_hyperparameters[3]]}")
            print(f"Leaf Size: {best_hyperparameters[4]}")
            print(f"P Value: {best_hyperparameters[5]}")

        elif self.estimator == "RF":
            print("Optimal hyperparameters:")
            print(f"n_estimators_value: {best_hyperparameters[0]}")
            print(f"max_depth_value: {best_hyperparameters[1]}")
            print(f"Criterion: {criterion_map_rf[best_hyperparameters[2]]}")
            print(f"min_samples_split_values: {min_samples_split_map_rf[best_hyperparameters[3]]}")
            print(f"min_samples_leaf_values: {best_hyperparameters[4]}")
            print(f"min_weight_fraction_leaf_values: {best_hyperparameters[5]}")
            print(f"max_features_values: {max_features_map_rf[best_hyperparameters[6]]}")
        
        elif self.estimator == "DT":
            print("Optimal hyperparameters:")
            print(f"Splitter: {splitter_map[best_hyperparameters[0]]}")
            print(f"max_depth_value: {best_hyperparameters[1]}")
            print(f"Criterion: {criterion_map_dt[best_hyperparameters[2]]}")
            print(f"min_samples_split_values: {min_samples_split_map_dt[best_hyperparameters[3]]}")
            print(f"min_samples_leaf_values: {best_hyperparameters[4]}")
            print(f"min_weight_fraction_leaf_values: {best_hyperparameters[5]}")
            print(f"max_features_values: {max_features_map_dt[best_hyperparameters[6]]}")

        elif self.estimator == "SVC":
            print("Optimal hyperparameters:")
            print(f"C: {[best_hyperparameters[0]]}")
            print(f"Kernel: {kernel_map[best_hyperparameters[1]]}")
            print(f"Degree: {best_hyperparameters[2]}")
            print(f"Gamma: {gamma_map[best_hyperparameters[3]]}")
            print(f"Shrinking: {shrinking_map[best_hyperparameters[4]]}")
            print(f"Decision Function Shape: {decision_function_shape_map[best_hyperparameters[5]]}")
        
        else:
            raise ValueError("Estimator not supported")

    def get_report(self, X_train, X_test, y_train, y_test, best_hyperparameters):

        estimator_instance = None
        if self.estimator == "KNN":
            k, distance_metric, weighting_method, algorithm, leaf_size, p_val = best_hyperparameters
            estimator_instance = KNeighborsClassifier(n_neighbors=int(k), metric=distance_metric_map[round(distance_metric)],
                                                    weights=weighting_method_map[round(weighting_method)],
                                                    algorithm=algorithm_map[round(algorithm)],
                                                    leaf_size=int(leaf_size), p=int(p_val))
        elif self.estimator == "RF":
            n_estimators_values, max_depth_values, criterion_values, min_samples_split_values, min_samples_leaf_values, min_weight_fraction_leaf_values, max_features_values = best_hyperparameters
            estimator_instance = RandomForestClassifier(random_state = self.random_state, n_estimators=int(n_estimators_values), max_depth=int(max_depth_values),
                                                        criterion=criterion_map_rf[round(criterion_values)], 
                                                        min_samples_split= min_samples_split_map_rf[round(min_samples_split_values)],
                                                        min_samples_leaf=int(min_samples_leaf_values),
                                                        min_weight_fraction_leaf = min_weight_fraction_leaf_values,
                                                        max_features=max_features_map_rf[round(max_features_values)]
                                                        )

        elif self.estimator == "DT":
            splitter_values, max_depth_values, criterion_values, min_samples_split_values, min_samples_leaf_values, min_weight_fraction_leaf_values, max_features_values = best_hyperparameters
            estimator_instance = DecisionTreeClassifier(random_state = self.random_state, splitter=splitter_map[round(splitter_values)], max_depth=int(max_depth_values),
                                                        criterion=criterion_map_dt[round(criterion_values)],min_samples_split=min_samples_split_map_dt[round(min_samples_split_values)],
                                                        min_samples_leaf=int(min_samples_leaf_values), 
                                                        min_weight_fraction_leaf=min_weight_fraction_leaf_values, max_features=max_features_map_dt[round(max_features_values)])
        elif self.estimator == "SVC":
            c, kernel_values, degree_values, gamma_values, shrinking_values, decision_function_shape_values = best_hyperparameters
            estimator_instance = SVC(random_state = self.random_state, C=int(c), kernel=kernel_map[round(kernel_values)],
                                    degree=int(degree_values), shrinking=shrinking_map[round(shrinking_values)], gamma=gamma_map[round(gamma_values)],
                                    decision_function_shape=decision_function_shape_map[round(decision_function_shape_values)])
        else:
            raise ValueError("Estimator not supported.")

        estimator_instance.fit(X_train, y_train)
        y_pred = estimator_instance.predict(X_test)
        # accuracy_pso = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return report    

    def _get_hyperparameter_space(self):

        # Define hyperparameter space based on the estimator
        if self.estimator == "KNN":
            hyperparameter_space = {
                "k": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "distance_metric": [0, 1, 2, 3, 4, 5],
                "weighting_method": [0, 1],
                "algorithm": [0, 1, 2, 3],
                "leaf_size": [10, 20, 30, 40, 50],
                "p_val": [1, 2, 3, 4, 5]
            }
        elif self.estimator == "RF":
            hyperparameter_space = {
                "n_estimators_values": [10, 50, 100, 200, 500],
                "max_depth_values": [5, 10, 20],
                "criterion_values": [0, 1, 2],
                "min_samples_split_values": [0, 1, 2],
                "min_samples_leaf_values": [1, 2, 3, 4, 5],
                "min_weight_fraction_leaf_values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                "max_features_values": [0, 1, 2]
            }
        elif self.estimator == "DT":
            hyperparameter_space = {
                "splitter_values": [0,1],
                "max_depth_values": [5, 10, 20],
                "criterion_values": [0, 1, 2],
                "min_samples_split_values": [0, 1, 2],
                "min_samples_leaf_values": [1, 2, 3, 4, 5],
                "min_weight_fraction_leaf_values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                "max_features_values": [0, 1]
            }
        elif self.estimator == "SVC":
            hyperparameter_space = {
                "c": [1, 10, 100],
                "kernel_values": [0, 1, 2, 3],
                "degree_values": [2, 3, 4],
                "gamma_values": [0, 1],
                "shrinking_values": [0,1],
                "decision_function_shape_values": [0, 1]
            }
        else:
            raise ValueError("Estimator not supported.")
        return hyperparameter_space
