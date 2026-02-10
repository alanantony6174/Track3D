import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def iou_3d(bboxes1, bboxes2):
    """
    Calculate 3D IoU between two sets of bounding boxes.
    Args:
        bboxes1: (N, 7) [h, w, l, x, y, z, theta]
        bboxes2: (M, 7) [h, w, l, x, y, z, theta]
    Returns:
        iou_matrix: (N, M)
    """
    # For simplicity, we approximate 3D IoU as BEV IoU * Height IoU
    # In a full implementation, this should use a polygon clipping library like shapely for BEV
    # Here we simplify to axis-aligned for basic function, or we can use the IoU logic from AB3DMOT
    # Given the complexity of rotating 3D IoU without CUDA/C++ extensions, we will use
    # a method that approximates the boxes as axis-aligned if rotation is small, 
    # OR we use Euclidean distance as a robust fallback.
    
    # Placeholder: Using Euclidean Distance as similarity for now to ensure robustness
    # AB3DMOT uses custom C++ extensions for 3D IoU which we can't easily compile here.
    # We will use Center Distance and Volume IoU (Axis Aligned) as a proxy.
    
    # TODO: Implement full Polygon3D IoU if needed.
    pass

def euclidean_distance_3d(bboxes1, bboxes2):
    """
    Args:
        bboxes1: (N, 7)
        bboxes2: (M, 7)
    """
    centers1 = bboxes1[:, 0:3]
    centers2 = bboxes2[:, 0:3]
    
    # Compute dists: (N, M)
    dists = cdist(centers1, centers2)
    return dists

def linear_assignment_3d(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > thresh:
            unmatched_a.append(r)
            unmatched_b.append(c)
        else:
            matches.append((r, c))
            
    # Add unmatched rows
    for r in range(cost_matrix.shape[0]):
        if r not in row_ind:
            unmatched_a.append(r)
            
    # Add unmatched cols
    for c in range(cost_matrix.shape[1]):
        if c not in col_ind:
            unmatched_b.append(c)
            
    return np.array(matches), np.array(unmatched_a), np.array(unmatched_b)
