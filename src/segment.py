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
        curvature = np.arccos(curvature)
        
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


def findSegmenationPoints(pts):
    if type(pts) != np.ndarray:
        pts = np.array(pts)
    
    processed_pts = preprocess(pts)
    c_list = computeCurvature(processed_pts, skip=3)
    a_list = computeAbnormality(c_list, window=15)
    seg_pt_idx = findAbnormalityAboveThreshold(a_list, kinv=0.833)
    out_points = []
    prev = 0
    
    for i in range(len(seg_pt_idx)):
        out_points.append(processed_pts[prev:seg_pt_idx[i]])
        prev = seg_pt_idx[i]
    if prev < len(processed_pts):
        out_points.append(processed_pts[prev:])
        
    return out_points, c_list, a_list
