import numpy as np
from numpy import linalg as LA
import math
import itertools
import random

from .segment import findSegmentationPoints
from .classification import classify

'''
-----------Some Preliminary Definitions------------
Machine Specifications:
- Vertex is specified by a center c(x,y) and radius r : {t,c=(x,y),r}
- Edge is specified by a sequence of target points e_i(x,y) : {t,E=[e_1,e_2,...,e_n]}
- Arrow is specified by a triangle ABC : {t,(A_x,A_y),(B_x,B_y),(C_x,C_y)}
- Self-loop is specified by a sequence of target points s_i(x,y) : {t,S=[s_1,s_2,...,s_n]}

- Interpreted subgraph is given by the best candidate graph
- Cost (loss) of parameters A, B & C depends on the intended interpretation
'''

#---------Domain Interpretation----------

def recog(S_i,L,A,B,C):
  # return list of candidate graphs sorted by score
  # exhaustive search for given stokes and their possible interpretations
  # S_i is a set of strokes, i.e., {(a_1,b_1),(a_2,b_2),...,(a_m,b_m)}
  # where a_i is the starting segment and b_i is the last segment
  # number of candidates = 4^(number of seg_pts)
  candidates = []
  probabilities = []

  S_i = S_i.reshape((-1,2))

  seg_pts = findSegmentationPoints(S_i)[0]
  all_segs = []
  all_probs = []

  for seg in seg_pts:
    curr_seg = []
    probs,specs = classify(seg)
    all_probs.append(list(probs))
    for comp in specs:
      v = []
      e = []
      t = []
      l = []
      if comp=='vertex':
        v.append('vertex')
        v.append(list(specs[comp]['center']))
        v.append(specs[comp]['radius'])
        curr_seg.append(v)
      if comp=='edge':
        e.append('edge')
        e.append(list(specs[comp]))
        curr_seg.append(e)
      if comp=='loop':
        l.append('loop')
        l.append(list(specs[comp]))
        curr_seg.append(l)
      if comp=='triangle':
        t.append('triangle')
        t.append(list(specs[comp]['a']))
        t.append(list(specs[comp]['b']))
        t.append(list(specs[comp]['c']))
        curr_seg.append(t)

    all_segs.append(curr_seg)
  
  for element in itertools.product(*all_segs):
    candidates.append(L+list(element))
  
  for element in itertools.product(*all_probs):
    # all component probabilities are greater than 0.2
    if all(i >= 0.2 for i in element):
      probabilities.append(sum(element))
    else:
      probabilities.append(0.0)
  
  scores = []
  for cand in candidates:
    scores.append(score(cand,A,B,C))
  
  final_scores = [sum(x) for x in zip(scores, probabilities)]
  sorted_candidates = [x for _, x in sorted(zip(final_scores, candidates), reverse=True)]
  
  return sorted_candidates

def score(p,A=-1.3,B=0.2,C=-0.6):
  return A*num_components(p) + B*num_connections(p) + C*num_missing_connections(p)

def num_components(p):
  return len(p)

def num_connections(p):
  #print('vertex-edge: ',len(vertex_edge(p)))
  #print('arrow-edge: ', len(arrow_edge(p))) 
  return 2*(len(vertex_edge(p)) + len(vertex_edge(p,loop=True)) + len(arrow_edge(p)) + len(arrow_edge(p,loop=True)))

def num_missing_connections(p):
  arrows = []
  edges = []
  loops =[]
  for comp in p:
    if comp[0] == 'triangle':
      arrows.append(comp)
    elif comp[0] == 'edge':
      edges.append(comp)
    elif comp[0] == 'loop':
      loops.append(comp)
  
  # 1 edge = 2 edge endpoints
  mis_edge_pts = 2*(len(edges)+len(loops))-len(vertex_edge(p))-len(vertex_edge(p,loop=True))
  mis_arrows = len(arrows)-len(arrow_edge(p))-len(arrow_edge(p,loop=True))
  mis_loops=0
  sl_conns = vertex_edge(p,loop=True)
  for loop in loops:
    v1 = None
    v2 = None
    fl = 0
    for conn in sl_conns:
      if conn[1]==loop and fl==0:
        v1 = conn[0]
        fl+=1
      if conn[1]==loop and fl==1:
        v2 = conn[0]
        fl+=1
    if fl==2 and v1!=v2:
      mis_loops+=1

  return mis_edge_pts + mis_arrows + 1000*mis_loops

def ccw(A,B,C):
  return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
# https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
def intersect(A,B,C,D):
  return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

# intersection point of line1 and line2
# https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
def line_intersection(line1, line2):
  xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
  ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

  div = det(xdiff, ydiff)
  if div == 0:
    raise Exception('lines do not intersect')

  d = (det(*line1), det(*line2))
  x = det(d, xdiff) / div
  y = det(d, ydiff) / div
  return x, y

def det(a, b):
  return a[0] * b[1] - a[1] * b[0]

def arrow_edge(p,loop=False):
  connections = []
  gamma = 20
  arrows = []
  edges = []
  for comp in p:
    if comp[0] == 'triangle':
      arrows.append(comp)
    elif not loop:
      if comp[0] == 'edge':
        edges.append(comp)
    else:
      if comp[0] == 'loop':
        edges.append(comp)

  for e in edges:
    # endpoint 1
    e0 = []
    e1 = e[1][0]
    e2 = e[1][1]
    slope = (e2[1]-e1[1])/(e2[0]-e1[0])
    if e2[0]-e1[0] >= 0:
      e0.append(e1[0]-(gamma/math.sqrt(slope**2 + 1)))
    else:
      e0.append(e1[0]+(gamma/math.sqrt(slope**2 + 1)))
    if e2[1]-e1[1] >= 0:
      e0.append(e1[1]-((slope*gamma)/math.sqrt(slope**2 + 1)))
    else:
      e0.append(e1[1]+((slope*gamma)/math.sqrt(slope**2 + 1)))
    e[1].insert(0,e0)

    q = np.inf
    conn = []
    point1 = None
    point2 = None
    point3 = None
    for a in arrows:
      for i in range(len(e[1]),2,-1):
        if intersect(e[1][i-1],e[1][i-2],a[1],a[2]):
          point1 = a[1]
          point2 = a[2]
          point3 = a[3]
          break
        elif intersect(e[1][i-1],e[1][i-2],a[2],a[3]):
          point1 = a[2]
          point2 = a[3]
          point3 = a[1]
          break
        elif intersect(e[1][i-1],e[1][i-2],a[3],a[1]):
          point1 = a[3]
          point2 = a[1]
          point3 = a[2]
          break
        else:
          pass

        if (point1 != None) and (point2 != None):
          x = line_intersection((point1,point2),(e[1][i-1],e[1][i-2]))
          if (dist(e[1][0],x) < gamma) or (dist(e[1][0],point3) < gamma):
            if dist(midpoint(point1,point2),x) < q:
              q = dist(midpoint(point1,point2),x)
              conn = [a,e,e[1][1]]
    e[1].pop(0)
    if len(conn)!=0:
      connections.append(conn)

    # endpoint 2
    e0 = []
    e1 = e[1][-1]
    e2 = e[1][-2]
    slope = (e2[1]-e1[1])/(e2[0]-e1[0])
    if e2[0]-e1[0] >= 0:
      e0.append(e1[0]-(gamma/math.sqrt(slope**2 + 1)))
    else:
      e0.append(e1[0]+(gamma/math.sqrt(slope**2 + 1)))
    if e2[1]-e1[1] >= 0:
      e0.append(e1[1]-((slope*gamma)/math.sqrt(slope**2 + 1)))
    else:
      e0.append(e1[1]+((slope*gamma)/math.sqrt(slope**2 + 1)))
    e[1].append(e0)

    q = np.inf
    conn = []
    point1 = None
    point2 = None
    point3 = None
    for a in arrows:
      for i in range(len(e[1])-1):
        if intersect(e[1][i],e[1][i+1],a[1],a[2]):
          point1 = a[1]
          point2 = a[2]
          point3 = a[3]
          break
        elif intersect(e[1][i],e[1][i+1],a[1],a[2]):
          point1 = a[2]
          point2 = a[3]
          point3 = a[1]
          break
        elif intersect(e[1][i],e[1][i+1],a[3],a[1]):
          point1 = a[3]
          point2 = a[1]
          point3 = a[2]
          break
        else:
          pass

        if (point1 != None) and (point2 != None):
          x = line_intersection((point1,point2),(e[1][i-1],e[1][i-2]))
          if (dist(e[1][-1],x) < gamma) or (dist(e[1][-1],point3) < gamma):
            if dist(midpoint(point1,point2),x) < q:
              q = dist(midpoint(point1,point2),x)
              conn = [a,e,e[1][-2]]
    e[1].pop()
    if len(conn)!=0:
      connections.append(conn)
  
  return connections

def vertex_edge(p,loop=False):
  connections = []
  thres = 20
  vertices = []
  edges = []
  for comp in p:
    if comp[0] == 'vertex':
      vertices.append(comp)
    elif not loop:
      if comp[0] == 'edge':
        edges.append(comp)
    else:
      if comp[0] == 'loop':
        edges.append(comp)

  for e in edges: # e = {E=[e_1,...,e_n],t_i}
    # endpoint 1
    q = np.inf
    conn = []
    for v in vertices: # v = {center,radius,t_i}
      d = dist(v[1],e[1][0])
      if d-v[2] < thres and max(d-v[2],0) < q:
        q = max(d-v[2],0)
        conn = [v,e,e[1][0]]
    if len(conn)!=0:
      connections.append(conn)
   
    # endpoint 2
    q = np.inf
    conn = []
    for v in vertices: # v = {center,radius,t_i}
      d = dist(v[1],e[1][-1])
      if d-v[2] < thres and max(d-v[2],0) < q:
        q = max(d-v[2],0)
        conn = [v,e,e[1][-1]]
    if len(conn)!=0:
      connections.append(conn)  
  
  return connections

def dist(a,b):
  return math.sqrt((a[1]-b[1])**2 + (a[0]-b[0])**2)

def midpoint(a,b):
  return [(a[0]+b[0])/2,(a[1]+b[1])/2]

def is_isomorphic(z,Gi):
  '''
  - Isomorphism for directed graph (arrows) not considered because
  it is very unlikely that graphs are not isomorphic after their
  undirected version are isomprhic in our case after classification.
  - We did not check whether z and Gi had the same 
  adjacent vertex degrees because of complexity & time constraints.
  - Our algorithm checks the number of vertices, number of edges
  and the degreee sequence of z and Gi.
  '''
  vertices_z = 0
  vertices_g = 0
  edges_z = 0
  edges_g = 0
  deg_z = []
  deg_g = []

  connections_z = vertex_edge(z) + vertex_edge(z, loop=True)
  connections_g = vertex_edge(Gi) + vertex_edge(Gi, loop=True)

  for comp in z:
    if comp[0] == 'vertex':
      vertices_z+=1
      deg = 0
      for i in range(len(connections_z)):
        if connections_z[i][0] == comp:
          deg+=1
      deg_z.append(deg) 
    if comp[0] == 'edge':
      edges_z+=1
  deg_z = sorted(deg_z)

  for comp in Gi:
    if comp[0] == 'vertex':
      vertices_g+=1
      deg = 0
      for i in range(len(connections_g)):
        if connections_g[i][0] == comp:
          deg+=1
      deg_g.append(deg)
    if comp[0] == 'edge':
      edges_g+=1
  deg_g = sorted(deg_g)
  
  # same # of vertices?
  # same # of edges?
  # same degree sequence?
  if vertices_g == vertices_z:
    if edges_g == edges_z:
      if deg_g == deg_z:
        return True
    else:
      return False
  else:
    return False

#----------Cost Calculation------------

# Y = sequences of stroke sets = sketched graphs Q_1, Q_2, ...
def Cost(Y,A,B,C):
  totalcost = 0
  n = len(Y)
  for i in range(n):
    Q_i = Y[i]
    L_i = [] # locked-in graph for Q_i
    totalcost += CostSequence(Q_i,A,B,C,L_i)
  return totalcost

# Q = set of strokes introduced by the user = sketched graph
# G_i is the real (intended) representation of subgraph S_i
def CostSequence(Q,A,B,C,L):
  cost = 0
  m = len(Q)
  for i in range(m):
    S_i = Q[i][0] # set of added strokes
    G_i = Q[i][1] # intended subgraph after adding segments in S_i
    if not L[i-1]:
      cost += 1
      L[i] = False
    else:
      cost += CostPair(S_i,G_i,L,i,A,B,C)
  return cost

def CostPair(S_i,G_i,L,i,A,B,C):
  Z_i = recog(S_i,L[i-1],A,B,C) # set of sorted candidate graphs at step i
  L[i] = LockIn(Z_i,G_i)
  if not is_isomorphic(Z_i[0],G_i):
    return 1
  else:
    return 0

def LockIn(Z_i,G_i):
  for j in range(len(Z_i)):
    if is_isomorphic(Z_i[j],G_i):
      return Z_i[j]
  return False

#---------Training----------
def train_interpretation(Y):
  # Y = sketched graphs
  # code tested on sample classification probabilities
  As = np.arange(-1.6,-0.8,0.1)
  Bs = np.arange(0.2,0.7,0.1)
  Cs = np.arange(-0.6,-0.1,0.1)

  mincost = np.inf
  minparam = []
  for A, B, C in zip(As,Bs,Cs):
    currcost = Cost(Y,A,B,C)
    if currcost < mincost:
      mincost = currcost
      minparam = [A,B,C]

  print(minparam)