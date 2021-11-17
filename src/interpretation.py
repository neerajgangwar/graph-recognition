import numpy as np
from numpy import linalg as LA

'''
-----------All sections of this script not yet tested------------
Machine Specifications:
- Vertex is specified by a center c(x,y) and radius r : {c=(x,y),r=(x,y),t}
- Edge is specified by a sequence of target points e_i(x,y) : {E=[e_1,e_2,...,e_n],t}
- Arrow is specified by a triangle ABC : {(A_x,A_y),(B_x,B_y),(C_x,C_y),t}
- Self-loop is specified by a sequence of target points s_i(x,y) : {S=[s_1,s_2,...,s_n],t}
'''

#---------Domain Interpretation----------

def machine_spec(p):
  # TODO -- represent graph p in terms of machine specifications
  pass

def recog(S_i,l,A,B,C):
  # return list of candidate graphs sorted by score
  # S_i is a set of strokes, i.e., {(a_1,b_1),(a_2,b_2),...,(a_m,b_m)}
  # where a_i is the starting segment and b_i is the last segment
  # TODO -- how to identify candidate graphs ???
  sorted_candidates = None
  return sorted_candidates

def num_components(p):
  return len(p)

def num_connections(p):
  return 2*(len(vertex_edge(p)) + len(vertex_edge(p,loop=True)) + \
    len(arrow_edge(p)) + len(arrow_edge(p,loop=True)))

def num_missing_connections(p):
  arrows = []
  edges = []
  loops =[]
  for comp in p:
    if comp[3] == 'arrow':
      arrows.append(comp)
    elif comp[1] == 'edge':
      edges.append(comp)
    elif comp[1] == 'self-loop':
      loops.append(comp)
  
  # 1 edge = 2 edge endpoints
  mis_edge_pts = 2*(len(edges)+len(loops))-len(vertex_edge(p))-len(vertex_edge(p,loops=True))
  mis_arrows = len(arrows)-len(arrow_edge(p))-len(arrow_edge(p,loop=True))
  mis_loops=0
  sl_conns = vertex_edge(p,loops=True)
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

def arrow_edge(p,loop=False):
  # p = machine_spec(p)
  connections = []
  gamma = 20
  arrows = []
  edges = []
  for comp in p:
    if comp[3] == 'arrow':
      arrows.append(comp)
    elif not loop:
      if comp[1] == 'edge':
        edges.append(comp)
    else:
      if comp[1] == 'self-loop':
        edges.append(comp)
  # TODO -- check my arrow-edge implementation
  pass

def vertex_edge(p,loop=False):
  # p = machine_spec(p)
  connections = []
  thres = 20
  vertices = []
  edges = []
  for comp in p:
    if comp[2] == 'vertex':
      vertices.append(comp)
    elif not loop:
      if comp[1] == 'edge':
        edges.append(comp)
    else:
      if comp[1] == 'self-loop':
        edges.append(comp)

  for e in edges: # e = {E=[e_1,...,e_n],t_i}
    # endpoint 1
    q = np.inf
    conn = []
    for v in vertices: # v = {center,radius,t_i}
      d = dist(v[0],e[0][0])
      if d-v[1] < thres and max(d-v[1],0) < q:
        q = max(d-v[1],0)
        conn = [v,e,e[0][0]]
    if len(conn)!=0:
      connections.append(conn)
   
    # endpoint 2
    q = np.inf
    conn = []
    for v in vertices: # v = {center,radius,t_i}
      d = dist(v[0],e[0][-1])
      if d-v[1] < thres and max(d-v[1],0) < q:
        q = max(d-v[1],0)
        conn = [v,e,e[0][-1]]
    if len(conn)!=0:
      connections.append(conn)  
  
  return connections

def classification_score(p):
  # TODO -- get probabilities from classification section
  score = None
  flag = None # all component probabilities are greater than 0.2
  return score,flag

def score(p,A,B,C):
  if classification_score(p)[1]:
    return classification_score(p)[0] + A*num_components(p) + \
      B*num_connections(p) + C*num_missing_connections(p)
  else:
    return 0

def dist(x,y):
  return LA.norm(x-y)

def is_isomorphic(z,Gi):
  # TODO -- check my isomorphic graphs implementation
  pass

#----------Training Algorithm------------

# Y = sequences of stroke sets = sketched graphs Q_1, Q_2, ...
def Cost(Y,A,B,C,):
  totalcost = 0
  for i in range(len(Y)):
    Q_i = Y[i]
    totalcost += CostSequence(Q_i,A,B,C)
  return totalcost

# Q = set of strokes introduced by the user = sketched graph
def CostSequence(Q,A,B,C):
  cost = 0
  L = [] # locked-in graph
  for i in range(len(Q)):
    S_i = Q[i][0] # set of added strokes
    G_i = Q[i][1] # subgraph after adding segments in S_i
    if i==0:
      cost += CostPair(S_i,G_i,L,i,A,B,C)
    else:
      if not L[i-1]:
        cost += 1
        L[i] = False
      else:
        cost += CostPair(S_i,G_i,L,i,A,B,C)
  return cost

def CostPair(S_i,G_i,L,i,A,B,C):
  Z_i = recog(S_i,L[i-1],A,B,C) # sorted candidate graphs at step i
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

#---------Retrieve Best Parameters----------
if __name__ == '__main__':
  Y = [] # sketched graphs
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