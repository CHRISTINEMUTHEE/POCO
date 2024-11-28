from scipy.spatial import KDTree
import numpy as np

# def compute_density(points, radius):
#     """Compute density for each point based on neighbors within a given radius."""
#     tree = KDTree(points)
#     densities = []
#     for point in points:
#         neighbors = tree.query_ball_point(point, radius)
#         densities.append(len(neighbors))
#     return np.array(densities)

# def compute_entropy(points, features, radius):
#     """Compute entropy for each point based on neighboring feature distributions."""
#     tree = KDTree(points)
#     entropies = []
#     for point in points:
#         neighbors = tree.query_ball_point(point, radius)
#         neighbor_features = features[neighbors]
#         probabilities = np.mean(neighbor_features, axis=0)
#         probabilities = probabilities / probabilities.sum()  # Normalize
#         entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
#         entropies.append(entropy)
#     return np.array(entropies)

def compute_density(points, k):
    """
    Compute density for each point in a point cloud.
    Density = k / (4 * pi * r^3), where r is the distance to the k-th neighbor.
    """
    tree = KDTree(points)
    densities = []
    for point in points:
        distances, _ = tree.query(point, k=k)
        r = distances[-1]  # Farthest neighbor distance
        density = k / (4 * np.pi * (r ** 3) + 1e-8)  # Avoid division by zero
        densities.append(density)
    return np.array(densities)

def compute_entropy(points, k):
    """
    Compute entropy for each point based on feature distributions in its neighborhood.
    """

    tree = KDTree(points)
    entropies = []

    for point in points:
        distances, _ = tree.query(point, k=k)
        # print("Negative distances:", np.where(distances < 0))
        probabilities = distances / (np.sum(distances) + 1e-8)  # Normalize
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
        entropies.append(entropy)
    # print("Negative entropies:", np.where(np.array(entropies) < 0))
    return np.array(entropies)


def normalize_values(values):
    """
    Normalize values to the range [0, 1].
    """
    min_value = np.min(values)
    max_value = np.max(values)
    normalized_values = (values - min_value) / (max_value - min_value + 1e-8)

    return normalized_values

def compute_adaptive_score(points, k, alpha=0.5, beta=0.5):
    """
    Compute adaptive scores for a point cloud based on density and entropy.
    """
    densities = compute_density(points, k)
    entropies = compute_entropy(points, k)
    
    # Normalize densities and entropies
    normalized_densities = normalize_values(densities)
    normalized_entropies = normalize_values(entropies)

    # print("normalized densities:", normalized_densities)
    # print("normalized entropies:", normalized_entropies)
    
    # Compute adaptive score
    adaptive_scores = alpha * normalized_densities + beta * normalized_entropies

    return adaptive_scores

