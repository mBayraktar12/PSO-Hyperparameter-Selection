# KNN
distance_metric_map = {
    0: "euclidean",
    1: "manhattan",
    2: "chebyshev",
    3: "minkowski",
    4: "l1",
    5: "l2",
}

weighting_method_map = {
    0: "uniform",
    1: "distance"
}

algorithm_map = {
    0: "brute",
    1: "kd_tree",
    2: "auto",
    3: "ball_tree"
}

# RANDOM FOREST
criterion_map_rf = {
     0: "gini",
     1: "entropy",
     2: "log_loss"
 }

min_samples_split_map_rf = {
     0: 2,
     1: 5,
     2: 10
 }

max_features_map_rf = {
    0: "sqrt",
    1: "log2",
    2: None
}

# DECISION TREE
criterion_map_dt = {
     0: "gini",
     1: "entropy",
     2: "log_loss"
 }

splitter_map = {
    0: "best",
    1: "random"
}

min_samples_split_map_dt = {
     0: 2,
     1: 5,
     2: 10
 }

max_features_map_dt = {
    0: "sqrt",
    1: "log2"
}

# SVC
kernel_map = {
    0:"linear",
    1:"poly",
    2:"rbf",
    3:"sigmoid"
}

gamma_map = {
    0:"scale",
    1:"auto"
}

shrinking_map = {
    0:True,
    1:False
}

decision_function_shape_map = {
    0:"ovo",
    1:"ovr"
}