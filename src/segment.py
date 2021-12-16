import math
import numpy as np
from scipy.ndimage import maximum_filter


def preprocess(pts, threshold=2):
    processed_pts = np.zeros((0, 2))
    
    prev_idx = 0
    processed_pts = np.array([pts[0]]).reshape(-1, 2)
    for idx in range(1, len(pts)):
        dist = np.linalg.norm(pts[idx] - pts[prev_idx])
        if dist >= threshold:
            processed_pts = np.append(processed_pts, pts[idx].reshape(-1, 2), axis=0)
            prev_idx = idx

    return processed_pts


def computeCurvature(pts, skip=2):
    curvature_list = np.zeros(0)
    for idx in range(pts.shape[0]):
        if idx - skip < 0 or idx + skip > len(pts) - 1:
            curvature_list = np.append(curvature_list, 0)
            continue

        pt1 = pts[idx - skip]
        pt2 = pts[idx]
        pt3 = pts[idx + skip]
        
        v1 = pt2 - pt1
        v2 = pt3 - pt2
        
        curvature = np.dot(v1, v2)
        curvature /= np.linalg.norm(v1)
        curvature /= np.linalg.norm(v2)
        # Clipping to ensure arccos does not fail with invalid
        # values. Ex: it might happen due to float comutations that
        # the dot product results in 1.000000000000002 instead of
        # 1.00.
        curvature = np.arccos(np.clip(curvature, -1, 1))
        
        curvature_list = np.append(curvature_list, curvature)
        
    return curvature_list


def computeAbnormality(curvature_list, window=4):
    abnormality_list = np.zeros((0))
    n = curvature_list.shape[0]
    
    for idx in range(n):
        mi = max(0, idx - window)
        ma = min(n, idx + window)
        
        ci = curvature_list[idx]
        cimi = np.sum(curvature_list[mi : idx+1])
        cima = np.sum(curvature_list[idx+1 : ma])
        
        a = max(1, idx + 1 - window)
        b = min(n, idx + 1 + window)
        
        abnormality = ci - ((cimi + cima) / (b - a))
        abnormality_list = np.append(abnormality_list, abnormality)
        
    return abnormality_list


def findAbnormalityAboveThreshold(abnormality_list, kinv=0.5):
    abnormality_list = np.copy(abnormality_list)
    abnormality_list[abnormality_list < kinv] = 0
    zero_val_idx = np.where(abnormality_list == 0)
    
    abnormality_list_max = maximum_filter(abnormality_list, size=(3))
    max_idx = np.where(abnormality_list == abnormality_list_max)
    
    return np.setdiff1d(max_idx[0], zero_val_idx[0])


def findSegmentationPoints(pts):
    if type(pts) != np.ndarray:
        pts = np.array(pts)

    processed_pts = preprocess(pts)
    c_list = computeCurvature(processed_pts, skip=3)
    a_list = computeAbnormality(c_list, window=12)
    seg_pt_idx = findAbnormalityAboveThreshold(a_list, kinv=0.833)
    out_points = []
    prev = 0
    
    for i in range(len(seg_pt_idx)):
        out_points.append(processed_pts[prev:seg_pt_idx[i]])
        prev = seg_pt_idx[i]
    if prev < len(processed_pts) - 1:
        out_points.append(processed_pts[prev:])

    # If no segmentation points (corners), return all the points. Applicable in cases
    # of an edge or node where there are no corners.
    if len(out_points) == 0:
        out_points = np.array([processed_pts])

    return out_points, c_list, a_list, processed_pts[seg_pt_idx]


def findSegmentationPointsSpeedBased(pts, timestamps_in_ns, threshold=50):
    if type(pts) != np.ndarray:
        pts = np.array(pts)

    s_list = np.zeros((1))

    for idx in range(1, len(pts) - 1):
        dist = math.dist(pts[idx], pts[idx - 1])
        t = timestamps_in_ns[idx] - timestamps_in_ns[idx - 1]
        s_list = np.append(s_list, dist / t)

    # For first and last index, speed will be 0
    s_list = np.append(s_list, 0)

    # Convert from pixels/ns to pixels/s
    s_list *= 1e9

    seg_idx = np.where(s_list < threshold)
    seg_idx = seg_idx[0]
    
    # Remove first and last index as speed is undefined at those points
    seg_idx = np.setdiff1d(seg_idx, [0, len(pts) - 1])
    seg_pts = pts[seg_idx]
    seg_pts = preprocess(seg_pts, threshold=10)

    return s_list, seg_pts
