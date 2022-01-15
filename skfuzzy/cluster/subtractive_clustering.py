"""Subtractive Clustering Algorithm
"""
__author__ = 'Bhavesh Kumar'
__author__ = 'Hélder Seixas Lima'
# Improvement of original implementation https://github.com/bhaveshkr/Subtractive-Clustering-Algorithm
# Refereces:
# Chiu, S., "Fuzzy Model Identification Based on Cluster Estimation," Journal of Intelligent & Fuzzy Systems, Vol. 2, No. 3, Sept. 1994.

import numpy as np
import pandas as pd

def subtractive_clustering(dataset, ra = 0.5, rejection_ratio = 0.15, acceptance_ratio = 0.5):
    # Squash factor
    rb = ra * 1.5
    # Acceptance ratio
    Eup = acceptance_ratio
    # Rejection ratio
    Edown = rejection_ratio
    alfa = 4 / ra ** 2
    beta = 4 / rb ** 2
    cluster_center = []
    
    if type(dataset) == np.ndarray:
        data_matrix = pd.DataFrame(dataset)
    else:
        data_matrix = dataset

    # Step 1: Calculate the likelihood that each data point would define a cluster center, based on the density of surrounding data points.
    size = len(data_matrix)
    number_dimensions = len(data_matrix[0])
    potential = [0.0] * size
    for i in range(size):
        Xi = data_matrix[i]
        for j in range(i + 1, size):
            Xj = data_matrix[j]
            sum_diff_to_square = 0
            for k in range(number_dimensions):
                sum_diff_to_square += (Xi[k] - Xj[k]) ** 2
            value = np.exp(-1.0 * alfa * sum_diff_to_square)
            potential[i] += value
            potential[j] += value

    # Step 2: Choose the data point with the highest potential to be the first cluster center.
    max_potential_value = max(potential)  # p1
    max_potential_index = potential.index(max_potential_value)

    # Repeat steps 3 and 4 until all the data is within the influence range of a cluster center.
    current_max_value = max_potential_value
    criteria = 1
    while criteria and current_max_value:
        criteria = 0
        current_potential_vector = data_matrix[max_potential_index]  # x1

        # Check criterias to flow the process
        if current_max_value > (Eup * max_potential_value):
            criteria = 1
        elif current_max_value < (Edown * max_potential_value):
            break
        else:
            distance_among_potential_cluster_end_effective_cluters = []
            for cc in cluster_center:
                distance_between_potential_cluster_and_cc = 0
                for k in range(number_dimensions):
                    distance_between_potential_cluster_and_cc += (current_potential_vector[k] - cc[k]) ** 2
                distance_among_potential_cluster_end_effective_cluters.append(distance_between_potential_cluster_and_cc)
            dmin = np.min(distance_among_potential_cluster_end_effective_cluters)
            if ((dmin / ra) + (current_max_value / max_potential_value)) >= 1:
                criteria = 1
            else:
                criteria = 2

        if criteria == 1:
            # Step 2.1 and 4.1: transform the potential cluster center to an effective cluste center
            cluster_center.append(current_potential_vector)
            # Step 3: Remove all data points near the current cluster center. The vicinity is determined using clusterInfluenceRange (ra).
            for i in range(size):
                Xj = data_matrix[i]
                potential_value = potential[i]
                sum_diff_to_square = 0
                for k in range(number_dimensions):
                    sum_diff_to_square += (current_potential_vector[k] - Xj[k]) ** 2
                potential_value = potential_value - (current_max_value * np.exp(-1.0 * beta * sum_diff_to_square))
                if potential_value < 0:
                    potential_value = 0
                potential[i] = potential_value
            # Step 4: Choose the remaining point with the highest potential as the next cluster center.
            current_max_value = max(potential)  # p1
            max_potential_index = potential.index(current_max_value)
        elif criteria == 2:
            potential[max_potential_index] = 0
            current_max_value = max(potential)  # p1
            max_potential_index = potential.index(current_max_value)

    return np.array(cluster_center)