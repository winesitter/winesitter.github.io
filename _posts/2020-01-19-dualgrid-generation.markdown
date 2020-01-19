---
layout: post
title:  "Generation of Median Dual Grids"
date:   2020-01-19 16:15:35 +0100
categories: jekyll update
---
This post is about the median dual grid representation of a triangular mesh.
Dual grids belong to the mathematical discipline of [graph theory](https://en.wikipedia.org/wiki/Graph_theory) and are often used in numerical solution procedures for differential equations, for example in [computational fluid dynamics](https://en.wikipedia.org/wiki/Computational_fluid_dynamics).
Read more on dual representations of graphs on [wikipedia](https://en.wikipedia.org/wiki/Dual_graph).

Here are all necessary packages: 

```python
import numpy as np, matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from copy import deepcopy
from math import pi, sin
```

In this block the class representing the dual grid is defined:

```python
class Dualmesh:
    '''
    This class represents the median dual structure of a Delaunay triangulation (referred to as primary grid)
    
    @param Primesh: scipy.spatial.Delaunay mesh object - the triangular mesh to process
    @param k: float - centroid weighting factor according to 
        Nishikawa, Journal of Computational Physics 401 (2020): 109001
    
    '''
    def __init__(self, Primesh, k = 0):
        '''
        @attribute points_p
        @attribute edges_p
        @attribute edges_p_dict
        @attribute elements_p
        @attribute neighbors_p
        
        @attribute edge_p_centroids
        @attribute element_p_centroids
        
        @attribute points_d
        @attribute elements_d
        @attribute element_volumes_d
        
        @attribute n_points_p
        @attribute n_edges_p
        @attribute n_elements_p
        
        '''
        # Copy necessary data from primary grid
        self.points_p = deepcopy(Primesh.points)
        self.elements_p = deepcopy(Primesh.simplices)
        self.neighbors_p = deepcopy(Primesh.neighbors)
        
        self.n_elements_p = len(self.elements_p)
        self.n_points_p = len(self.points_p)
        
        # Construct primary grid edge connectivity
        self.construct_edges_p()
        self.edge_p_centroids = self.calc_edge_centroids(self.edges_p)
        self.calc_element_p_centroids(k=k)
        
        # Construct median dual grid connectivity 
        self.construc_dual_elements()
        self.points_d = np.vstack((self.points_p,
                                   self.edge_p_centroids, 
                                   self.element_p_centroids))
        
        # Compute median dual grid element areas
        self.calc_element_areas_d()
        
    def construct_edges_p(self):
        '''
        Function to construct primary grid edge structure
        
        '''
        edges_p_tmp = dict()
        i_edge = 0
        
        for cur_elem in self.elements_p:
            for i_node_loc in range(3):
                cur_edge = tuple(sorted((cur_elem[i_node_loc],
                                         cur_elem[(i_node_loc+1)%3])))
                try: edge_exists = edges_p_tmp[cur_edge]
                except:
                    edges_p_tmp[cur_edge] = i_edge
                    i_edge += 1
                
        self.n_edges_p = len(edges_p_tmp.keys())
        edges_p = [None] * self.n_edges_p
        
        for cur_edge, i_edge in edges_p_tmp.items():
            edges_p[i_edge] = cur_edge
        
        self.edges_p = np.array(edges_p)
        self.edges_p_dict = edges_p_tmp
        
    def calc_edge_centroids(self, edges):
        '''
        Calculates centroids for a given numpy array of edges
        
        @param edges: (2xN) numpy-array - list of vertex indices, 
            defining N edges
        @return: (2xN) array - edge-centroids coordinates 
        
        '''
        return np.mean(self.points_p[edges],1)
    
    def calc_edge_lengths(self, edges):
        '''
        Calculates edgelengths for a given numpy array of edges
        
        @param edges: (2xN) numpy-array - list of vertex indices, 
            defining N edges
        @return: float - edge length
        '''
        return np.linalg.norm(self.points_p[edges[:,1]] - self.points_p[edges[:,0]], 
                              axis = 1)
    
    def calc_dualtri_area(self, tri):
        '''
        Calculates the area of a triangle from its vertex coordiantes
        
        @param tri: (3x1) array - vertex indices defining triangle 
        @return: float - triangle area 
        '''
        
        A = self.points_d[tri[0]]
        B = self.points_d[tri[1]]
        C = self.points_d[tri[2]]
        return abs(A[0]*(B[1]-C[1])
                  +B[0]*(C[1]-A[1])
                  +C[0]*(A[1]-B[1])) / 2.
    
    def calc_element_p_centroids(self, k = 0):
        '''
        Calcuates primary grid element centroids according to equation of 
        Nishikawa, Journal of Computational Physics 401 (2020): 109001
        
        @param k: float - centroid weighting factor
        
        '''
        element_p_centroids = [None] * self.n_elements_p
        
        for i_elem, cur_elem in enumerate(self.elements_p):
            edges = np.array([(cur_elem[i], cur_elem[(i+1)%3]) for i in range(3)])
            edge_centroids = self.calc_edge_centroids(edges)
            edge_lengths = self.calc_edge_lengths(edges)
            elem_centroid = np.sum(np.array([edge_lengths**k * edge_centroids[:,0], 
                                             edge_lengths**k * edge_centroids[:,1]]), 1)
            element_p_centroids[i_elem] = elem_centroid / np.sum(edge_lengths**k)
            
        self.element_p_centroids = np.array(element_p_centroids)

    def construc_dual_elements(self):
        '''
        Constructs the median dual representation of the primary grid.
        Every median dual grid element consinsts of several sub-triangles.
        Each of those is defined through its three vertex indices of the 
        median dual grid vertices.
        For the dual grid representation, every element consists of (3x1)
        tuples, defining its sub-triangles
        '''
        elements_d = [()] * self.n_points_p
        
        for i_elem, cur_elem in enumerate(self.elements_p):
            for i_node_loc in range(3):
                cur_edge = (cur_elem[i_node_loc],
                            cur_elem[(i_node_loc+1)%3])
                i_edge = self.edges_p_dict[tuple(sorted(cur_edge))]
                tri_1 = (cur_edge[0], 
                         i_edge + self.n_points_p,
                         i_elem + self.n_points_p + self.n_edges_p)
                tri_2 = (cur_edge[1],
                         i_elem + self.n_points_p + self.n_edges_p, 
                         i_edge + self.n_points_p)
                
                elements_d[cur_edge[0]] += tri_1,
                elements_d[cur_edge[1]] += tri_2,
                
        self.elements_d = elements_d
            
    def calc_element_areas_d(self):
        '''
        Calculates the median dual grid element areas from their sub-triangle
        areas
        '''
        vol = [0.0] * self.n_points_p
        
        for i_elem, subtris in enumerate(self.elements_d):
            for tri in subtris: vol[i_elem] += self.calc_dualtri_area(tri)
        
        self.element_areas_d = np.array(vol)
        
```

These are some auxiliary functions to generate the primary grid and plot the grids:

```python
def create_primary_grid(Nx, Ny, a=0.7, f=4.0):
    '''
    Create a set of points
    
    @param Nx, Ny: integer - number of vertices 
    @param a, f: float - grid deformation parameters
    @return: scipy.spatial.Delaunay mesh object
    '''
    x = [i + a * sin(f * pi * (j/Ny)) for i in range(Nx) for j in range(Ny)]
    y = [j + a * sin(-f * pi * (i/Nx)) for i in range(Nx) for j in range(Ny)]
    xy = np.vstack((x,y)).T.astype(float)
    return Delaunay(xy)

def plot_pimary_grid(primesh, ax):
    '''
    Plot primary grid structure to an matplotlib axis object
    
    @param primesh: primary grid object
    @param ax: matplotlib axis object
    '''
    tri_patches = []
    for tri in primesh.simplices: 
        tri_patches.append(Polygon([primesh.points[tri[i]] for i in range(3)]))

    tri_col = PatchCollection(tri_patches, color='', edgecolor='k', lw=1.5)
    ax.add_collection(tri_col)
    
def plot_dual_grid(dualmesh, ax):
    '''
    Plot median dual grid structure to an matplotlib axis object
    
    @param dualmesh: median dual grid object
    @param ax: matplotlib axis object
    '''
    dual_patches = []
    dual_areas = []
    for i_elem, dualtris in enumerate(dualmesh.elements_d):
        for tri in dualtris:
            dual_areas.append(dualmesh.element_areas_d[i_elem])
            dual_patches.append(Polygon([dualmesh.points_d[tri[i]] for i in range(3)]))

    dual_col = PatchCollection(dual_patches, edgecolors='k', alpha=1.0, lw=0.5,
                               cmap='coolwarm')

    dual_col.set_array(np.array(dual_areas))
    ax.add_collection(dual_col)
```


Finally, everythin is plotted here:

```python
# Create primary grid structure
primesh = create_primary_grid(12, 12, a=0.9)

# Create plots 
fig, ax = plt.subplots(1,3,figsize=(36,12))

# Create dualgrids for various k and plot them
for i, k in enumerate([0,2,8]):
    ax[i].set_aspect('equal')
    # Plot median dualmesh
    plot_dual_grid(Dualmesh(primesh, k = k), ax[i])
    # Plot primary mesh as wireframes
    plot_pimary_grid(primesh, ax[i])
    # Plot vertices
    ax[i].scatter(primesh.points[:,0], 
                  primesh.points[:,1],
                  s=40, marker='o', c = 'k')

    ax[i].set_title(r'$k = %d$' % k, fontsize = 26)
    ax[i].set_axis_off()
    ax[i].set_xlim((3,7))
    ax[i].set_ylim((3,7))

plt.show()

```

These are the resulting median dual representations for the same primary grid as input:
![png](/images/2020-01-19-dualgrid-generation_files/dualgrid-generation.png)
