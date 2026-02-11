import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def iou_3d(bboxes1, bboxes2):
    """
    Calculate 3D IoU between two sets of bounding boxes.
    Args:
        bboxes1: (N, 7) [x, y, z, theta, l, w, h]  # Note: Check if theta is 4th or last index in caller. Assuming index 3 based on context.
                   Actually, bot_sort_3d.py L12 says: [x, y, z, theta, l, w, h]
                   Wait, looking at bot_sort_3d.py L12 comment: [x, y, z, theta, l, w, h]
                   But in update/activate L30: x, y, z, theta, l, w, h, vx, vy, vz
                   Let's assume input is (N, 7) with [x, y, z, theta, l, w, h]
        bboxes2: (M, 7) [x, y, z, theta, l, w, h]
    Returns:
        iou_matrix: (N, M)
    """
    import shapely.geometry
    import shapely.affinity
    
    N = len(bboxes1)
    M = len(bboxes2)
    iou_matrix = np.zeros((N, M), dtype=np.float32)
    
    if N == 0 or M == 0:
        return iou_matrix
        
    for i, box1 in enumerate(bboxes1):
        # Create Polygon for box1
        x1, y1, z1, theta1, l1, w1, h1 = box1
        # Create rectangle centered at (0,0) with (l, w) then rotate and translate
        # shapely box is (minx, miny, maxx, maxy) -> we need polygon from standard box
        # (l, w) are dimensions. 
        # Points relative to center: (+l/2, +w/2), (-l/2, +w/2), (-l/2, -w/2), (+l/2, -w/2)
        p1 = shapely.geometry.Polygon([
            (l1/2, w1/2), (-l1/2, w1/2), (-l1/2, -w1/2), (l1/2, -w1/2)
        ])
        # Rotate by theta (radians or degrees? Standard in ROS/Autoware is radians)
        # Assuming radians. Shapely rotate takes degrees by default, need use_radians=True
        p1 = shapely.affinity.rotate(p1, theta1, use_radians=True)
        p1 = shapely.affinity.translate(p1, x1, y1)
        
        # Height range
        z_min1 = z1 - h1/2
        z_max1 = z1 + h1/2
        vol1 = l1 * w1 * h1
        
        for j, box2 in enumerate(bboxes2):
             x2, y2, z2, theta2, l2, w2, h2 = box2
             p2 = shapely.geometry.Polygon([
                 (l2/2, w2/2), (-l2/2, w2/2), (-l2/2, -w2/2), (l2/2, -w2/2)
             ])
             p2 = shapely.affinity.rotate(p2, theta2, use_radians=True)
             p2 = shapely.affinity.translate(p2, x2, y2)
             
             # BEV Intersection
             if not p1.intersects(p2):
                 continue
                 
             intersection_area = p1.intersection(p2).area
             
             # Height Intersection
             z_min2 = z2 - h2/2
             z_max2 = z2 + h2/2
             
             z_overlap_min = max(z_min1, z_min2)
             z_overlap_max = min(z_max1, z_max2)
             
             h_overlap = max(0.0, z_overlap_max - z_overlap_min)
             
             intersection_vol = intersection_area * h_overlap
             
             vol2 = l2 * w2 * h2
             union_vol = vol1 + vol2 - intersection_vol
             
             if union_vol > 0:
                 iou_matrix[i, j] = intersection_vol / union_vol
                 
    return iou_matrix

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
