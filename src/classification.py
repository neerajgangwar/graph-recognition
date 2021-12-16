'''
    Assuming we are representing segmented strokes as a numpy array of 2-D points, in chronological order
    Find the parameters given these representations
'''

import numpy as np
import scipy
from scipy.spatial import ConvexHull
import alphashape
import shapely

'''
    Adjust stroke to limit the range of distances for all points.
    Remove any segments below the minimum distance and linearly interpolate any segments above the maximum distance 
'''
def fix_stroke_distance (stroke, min_dist, max_dist):
    # Convert stroke to segments
    start = stroke[0]
    end = stroke[stroke.shape[0] - 1]
    
    segments = np.zeros((stroke.shape[0] - 1, 2))
    for n in range(1, stroke.shape[0]):
        segments[n - 1] = stroke[n] - stroke[n - 1]
        
    # Remove any strokes below minimum distance
    minsegments = np.zeros((0,2))
    running_segment = np.zeros((1,2))
    
    for n in range(0, segments.shape[0] - 1):
        running_segment = running_segment + segments[n]
        if np.linalg.norm(running_segment) < min_dist:
            continue
            
        minsegments = np.append(minsegments, running_segment, axis = 0)
        running_segment = np.zeros((1,2))

    
    # Divide any strokes above maximum distance
    maxsegments = np.zeros((0,2))

    for n in range(0, minsegments.shape[0]):
        if np.linalg.norm(minsegments[n]) < max_dist:
            maxsegments = np.append(maxsegments, minsegments[n])
            continue
        
        d = np.ceil(np.linalg.norm(minsegments[n]) / max_dist)
        
        divs = minsegments[n] // d
        
        maxsegments = np.append(maxsegments, np.tile(divs, (int(d), 1)))
        
    maxsegments = maxsegments.reshape((-1,2))
        
    newstroke = np.array([start])
        
    for segment in maxsegments:
        newstroke = np.append(newstroke, np.array([newstroke[-1] + segment]), axis = 0)
        
    return newstroke
    
    
'''
    Find the alpha shape ratio of a stroke 
    Perimeter of A(stroke, 25) / Area of A(stroke, 25) / 500
'''
def alpha_shape_ratio(stroke, alpha):
    A = alphashape.alphashape(np.asarray(stroke), alpha=alpha)
    return A.length / A.area / 500
    
'''
    Find the circumscribed circle ratio of a stroke 
    Area of CH(S) / Circle(CH(S))
'''
def circumscribed_circle_ratio(stroke):
    CH = ConvexHull(stroke)
    
    CH_points = stroke[CH.vertices]
    
    point_distances = scipy.spatial.distance.cdist(CH_points, CH_points, metric='euclidean')
        
    idx1, idx2 = np.unravel_index(np.argmax(point_distances, axis=None), point_distances.shape)
    
    diameter = point_distances.max()
    
    carea = (diameter / 2) ** 2 * np.pi
    
    return CH.volume / carea, CH_points[idx1] + CH_points[idx2] / 2, diameter / 2
    
'''
    Find the inscribed triangle ratio of a stroke 
    Area of ABC(S) / Area of CH(S)
    https://stackoverflow.com/questions/1621364/how-to-find-largest-triangle-in-convex-hull-aside-from-brute-force-search/1621913#1621913
'''
def triangle_ratio(stroke):
    # find largest triangle
    CH = ConvexHull(stroke)
    
    CH_points = stroke[CH.vertices]
    
    A, B, C = 0, 1, 2
    max_area = np.cross(CH_points[A] - CH_points[C], CH_points[B] - CH_points[C])
    
    while A < len(CH_points) - 1: 
        while True:
            while True:
                C = (C + 1) % len(CH_points)
                side1 = CH_points[A] - CH_points[C]
                side2 = CH_points[B] - CH_points[C]
                area = np.cross(side1, side2)
                angle = np.arccos(np.dot(side1, side2) / np.linalg.norm(side1) / np.linalg.norm(side2))
                if area > max_area and angle > 0.35:
                    max_area = area
                else:
                    break
            B = (B + 1) % len(CH_points)
            side1 = CH_points[A] - CH_points[C]
            side2 = CH_points[B] - CH_points[C]
            area = np.cross(CH_points[A] - CH_points[C], CH_points[B] - CH_points[C])
            angle = np.arccos(np.dot(side1, side2) / np.linalg.norm(side1) / np.linalg.norm(side2))
            if area > max_area and angle > 0.35:
                max_area = area
            else:
                break
        A = A + 1
        side1 = CH_points[A] - CH_points[C]
        side2 = CH_points[B] - CH_points[C]
        area = np.cross(CH_points[A] - CH_points[C], CH_points[B] - CH_points[C])
        angle = np.arccos(np.dot(side1, side2) / np.linalg.norm(side1) / np.linalg.norm(side2))
        if area > max_area and angle > 0.35:
            max_area = area
            
    return max_area / CH.volume / 2, CH_points[A], CH_points[B], CH_points[C]
  

def convex_hull_parameter(stroke):
    CH = ConvexHull(stroke)
    return CH.area / 500

'''
    Find the number of disjoint regions in the alpha shape that are over 50 pixels apart - 1.
    -1 is so that a stroke with 1 disjoint region does not disqualify the stroke.
'''  
def disjoint_shapes(stroke, alpha):
    A = alphashape.alphashape(np.asarray(stroke), alpha=alpha)
    if type(A) is shapely.geometry.multipolygon.MultiPolygon:
        points = [point for polygon in A.geoms for point in polygon.exterior.coords[:-1]]
    else:
        points = A.exterior.coords[:-1]
    ptrs = [-1,] * len(points)
        
    stack = [0]
    
    point_distances = scipy.spatial.distance.cdist(points, points, metric='euclidean')
        
    for point in point_distances:
        point[point == 0] = 51
        if point.min() > 50:
            return 1

    if point_count < len(points):
        return 1
                    
    return 0
    
def target_points(stroke, distance):
    start = stroke[0]
    end = stroke[stroke.shape[0] - 1:stroke.shape[0]]
    
    ret = np.array([start])
    
    
    cumdist = 0
    for n in range(1, stroke.shape[0] - 1):
        cumdist += np.linalg.norm(stroke[n] - stroke[n - 1])
        if (cumdist > distance):
            ret = np.append(ret, stroke[n:n+1], axis = 0)
            cumdist %= distance
    
    ret = np.append(ret, end, axis = 0)
    return ret
    
    

'''
    Using parameters pulled from original paper
   
    vertex, arrow, edge, self-loop
    
    Return np.array of probabilities and dict of specifications

'''
def classify(stroke):      
    stroke = fix_stroke_distance(stroke, 2, 10)

    parameters = np.array([[1.20, 21.20, -14.89, -8.61, -5.88, -1000],
        [-4.13, -8.45, 21.40, -3.98, -27.19, -1000],
        [0.67, -18.85, -3.07, 2.44, 17.78, -1000],
        [-0.28, 23.74, -16.93, -25.10, 6.40, -1000]])
    
    specifications = {}
    
    circle_ratio, center, radius = circumscribed_circle_ratio(stroke)
        
    specifications['vertex'] = {'center': center, 'radius': radius}
    specifications['edge'] = target_points(stroke, 40)
    specifications['loop'] = target_points(stroke, 20)
    
    tri_ratio, a, b, c = triangle_ratio(stroke)
    specifications['triangle'] = {'a' : a, 'b' : b, 'c' : c}
    
    
    features = np.array([1, circle_ratio, tri_ratio, \
                alpha_shape_ratio(stroke, 0.04), convex_hull_parameter(stroke), disjoint_shapes(stroke, 0.04)])
    
    return 1 / (1 + np.exp(-np.matmul(parameters, features.T))), specifications
