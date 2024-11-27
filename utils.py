from scipy.spatial import KDTree
import numpy as np

def compute_density(points, radius):
    """Compute density for each point based on neighbors within a given radius."""
    tree = KDTree(points)
    densities = []
    for point in points:
        neighbors = tree.query_ball_point(point, radius)
        densities.append(len(neighbors))
    return np.array(densities)

def compute_entropy(points, features, radius):
    """Compute entropy for each point based on neighboring feature distributions."""
    tree = KDTree(points)
    entropies = []
    for point in points:
        neighbors = tree.query_ball_point(point, radius)
        neighbor_features = features[neighbors]
        probabilities = np.mean(neighbor_features, axis=0)
        probabilities = probabilities / probabilities.sum()  # Normalize
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
        entropies.append(entropy)
    return np.array(entropies)
