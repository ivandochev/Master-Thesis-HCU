#Version 12.3.2017. MST function adopted from Andreas Mueller:
#http://peekaboo-vision.blogspot.de/2012/02/simplistic-minimum-spanning-tree-in.html
#It requires ALKIS, addresses and Clusters

import math 
import timeit
import itertools
import operator
import numpy as np
from scipy.spatial.distance import pdist, squareform
from itertools import groupby


#INPUT PARAMETERS
BuildFuncList = [1000,1010,1110,1120,1121,1122,1123,1100,1131]
StrWeight = 0.6 #Weigth of the street connection

#INPUT FIELD SPECIFICATION
BuildingFunction = 'gebaeudefu'
#BuildNumberofFloors = "anzahlDerO"
Bauweise = 'bauweise'
AddrPointField = 'zeigtAuf'
BaublockField = 'baublockbe'
StreetNameField = 'strname' #'strname'
HausNummer = 'hausnr'
NumZusatz = 'zusatz'
FlurID = 'FlurstID'
Cluster = 'Group' #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
VisOrder ='VisOrder'

tic=timeit.default_timer() # Timer begins

#PREPARE FUNCTIONS
def MST(X, copy_X=True):
    """X are edge weights of fully connected graph"""
    if copy_X:
        X = X.copy()
 
    if X.shape[0] != X.shape[1]:
        raise ValueError("X needs to be square matrix of edge weights")
    n_vertices = X.shape[0]
    spanning_edges = []
     
    # initialize with node 0:                                                                                         
    visited_vertices = [0]                                                                                            
    num_visited = 1
    # exclude self connections:
    diag_indices = np.arange(n_vertices)
    X[diag_indices, diag_indices] = np.inf
     
    while num_visited != n_vertices:
        new_edge = np.argmin(X[visited_vertices], axis=None)
        # 2d encoding of new_edge from flat, get correct indices                                                      
        new_edge = divmod(new_edge, n_vertices)
        new_edge = [visited_vertices[new_edge[0]], new_edge[1]]                                                       
        # add edge to tree
        spanning_edges.append(new_edge)
        visited_vertices.append(new_edge[1])
        # remove all edges inside current tree
        X[visited_vertices, new_edge[1]] = np.inf
        X[new_edge[1], visited_vertices] = np.inf                                                                     
        num_visited += 1
    return np.vstack(spanning_edges)

def MSTPLine(edge_list,points):
    geoms = QgsGeometry.fromWkt('GEOMETRYCOLLECTION()')
    for edge in edge_list:
        i, j = edge
        P1 = QgsPoint(points[i, 0], points[i, 1])
        P2 = QgsPoint(points[j, 0], points[j, 1])
        Line = (QgsGeometry.fromPolyline([P1,P2]))
        geoms = geoms.combine(Line)
    return geoms

def Densify(line,dist):
    L = line.length()
    currentDist = dist
    points = []
    while currentDist < L:
        point = line.interpolate(currentDist).asPoint()
        points.append(point)
        currentDist += dist
    return points

#PREPARE DICTIONARIES
tic=timeit.default_timer() # Timer begins
canvas = qgis.utils.iface.mapCanvas()
layers = canvas.layers()
#Layer order in the canvas becomes important
BuildDict = {b.id(): b for b in layers[0].getFeatures()}
StreetDict = {s.id(): s for s in layers[1].getFeatures()}
GridsLayer = layers[2]

#PREPARE STREET AND BUILD INDEX
SpIndex = QgsSpatialIndex()
BSpIndex = QgsSpatialIndex()
for s in StreetDict.values():
    SpIndex.insertFeature(s)

#0. PREPARE BUILDNG LIST
feature_list = []
for i in BuildDict.values():
    if i[BuildingFunction] in BuildFuncList\
        and i[StreetNameField] != NULL:
            BSpIndex.insertFeature(i)
            geom = i.geometry()
            feature_list.append([
            i,                                   #[0]
            i[FlurID],                         #[1]
            i[StreetNameField],           #[2]
            int(i[HausNummer]),         #[3]
            i[NumZusatz],  #[4]
            i[Cluster],                          #[5]
            i[VisOrder],                         #[6]
            "HzuHGroup",                        #[7]
            str(i[BaublockField])+str(i[StreetNameField]) #[8]
            ])
            '''I need the BB_Str since the TSP numbering
            does not include added groups to a cluster'''
 
#GROUP BASED ON CLUSTER
'''Based on cluster in order to have a segment for each
cluster. All buildings/HzuH and BBStr in the cluster will try
to reach this segment only'''
feature_list = sorted(feature_list, 
    key=operator.itemgetter(5,2,6))
ClusterGrouping = []
for Cl, Clgroup in groupby(feature_list, lambda x: x[5]):
    ClusterGrouping.append(list(Clgroup))
    

#FIND THE NEAREST STREET TO EACH CLUSTER
#And the cluster centroid is also added
NearStrDict = {}
for Cl in ClusterGrouping:
    MultiGeom = QgsGeometry.fromWkt('GEOMETRYCOLLECTION()')
    for b in Cl:
        MultiGeom = MultiGeom.combine(b[0].geometry())
#    print MultiGeom.exportToWkt()
    ClCentr = MultiGeom.centroid().asPoint()
    NearStrList = SpIndex.nearestNeighbor(ClCentr,4)
#    print "%s_%s" % (NearStrList, Cl[0][5])
#    print MultiGeom.centroid().exportToWkt()
    NearestDist = 10000 #Initial distance
    for near in NearStrList:
        NearStrGeom = StreetDict[near].geometry()
        NearStrSeg = NearStrGeom.closestSegmentWithContext(ClCentr)
        if math.sqrt(NearStrSeg[0]) < NearestDist:
            NearestDist = math.sqrt(NearStrSeg[0])
            NearestID = near
            NearP= NearStrSeg[1]
#    print QgsGeometry.fromPoint(NearP).exportToWkt()
    NearStrDict[Cl[0][5]] = NearestID

#CONNECT HzuH Clusters
OutputFeatureList = []
for Cl in ClusterGrouping:
    NearStrID = NearStrDict[Cl[0][5]]
    NearStrGeom = StreetDict[NearStrID].geometry()

    #If more than one build in Cluster
    if len(Cl) > 1:
        Points = Densify(NearStrGeom,5)
        #Street Points Count
        StrPNum = len(Points)
        #Add Buildings
        for b in Cl:
            Points.append(b[0].geometry().centroid().asPoint())
        #Building Points Count and TotalNum Count
        BPNum = len(Points) - StrPNum
        TotalNum = len(Points)
        
        #Generate distances
        p = np.array(Points)
        x = squareform(pdist(p))
        
        #Weights Matrix - unweighted part first(mask)
        M = np.empty([BPNum,BPNum])
        M.fill(1)
        
        #Then weighted base matrix
        W = np.ones_like(x)
        W.fill(StrWeight)
        
        #Change the bottom part of the matrix with 1s
        W[StrPNum: ,StrPNum: ] = M
        
        #Modify the distance matrix with the weights
        W2 = x*W
        
        #Compute Edges
        edge_list = MST(W2)
        Clgrid = MSTPLine(edge_list,p)
#    print Clgrid.exportToWkt()

    else:
        BuilCen = Cl[0][0].geometry().centroid()
        Con = BuilCen.shortestLine(NearStrGeom)
        Clgrid = NearStrGeom.combine(Con)
    
    #Length
    L = Clgrid.length()
    
    
    feat = QgsFeature(GridsLayer.pendingFields())
    feat.setAttribute('Cluster', Cl[0][5])
    feat.setAttribute('StrSeg', NearStrID)
    feat.setAttribute('Length', L)
    feat.setAttribute('StrSegLen', NearStrGeom.length())
    feat.setGeometry(Clgrid)

        
    OutputFeatureList.append(feat)


#print OutputFeatureList
GridsLayer.dataProvider().addFeatures(OutputFeatureList)

toc=timeit.default_timer()
print "Minutes elapsed : " + str((toc - tic)/60)

