# ClusterAnalysis

Minimum Requirements:

    Python 3.9.4

    --> numpy==1.20.2

    --> matplotlib==3.4.1

**Synopsis:**

There are many robust libraries with a variety of clustering methods that have been patched over the years to maximize efficiency. But sometimes, it can be a fruitful exercise to "reinvent the wheel". Without resorting to these established routines, this code contains a small variety of methods to perform a cluster analysis. Understanding ones dataset can help determine which clustering method would work "best". 
These methods work for data in n-dimensions. But first, let's consider 2-d data along two annular rings.

![Unclassified Data - 2d Annuli](https://github.com/mikeysflix/ClusterAnalysis/blob/master/Figures/annulus_2d/Unclassified_Data_dims-0_1.png?raw=true)

The first type of method we will look at is barycentric clustering, which includes k-means and k-medians. (I would like to add k-medoids via PAM to these methods.) If the number of clusters `k` is not known a priori, then the optimal value of `k` can be estimated using the elbow method or the silhouette method. Juding from first glance, it would appear that there are two clusters - the inner ring and the outer ring. Because these two rings share a common center, clustering based on their respective barycenters is likely not the optimal approach. Considering `k={2, 3, ..., 9, 10} for this example, the elbow method finds that the optimal number of clusters is `k=3` while the silhouette method finds `k=10`. 

![EX1: Comparison of Optimal K](https://github.com/mikeysflix/ClusterAnalysis/blob/master/Figures/annulus_2d/barycentric/by_elbow/Barycentric_K-Means_Cost_%26_Silhouette_ByK_Random_Centroids_vertical.png?raw=true)

Using `k=3` via the elbow method, the classified data and related specs are shown below.

![EX1: Full Result - Barycentric](https://github.com/mikeysflix/ClusterAnalysis/blob/master/Figures/annulus_2d/barycentric/by_elbow/Barycentric_K-Means_FullOptimizationResult_k3_dims-0_1_Random_Centroids.png?raw=true)

A better method for this dataset is clustering based on density. Only one density-based method - DBSCAN - is currently coded, but I would like to add mean-shift in the near future; this result is shown below.

![EX1: Full Result - Density](https://github.com/mikeysflix/ClusterAnalysis/blob/master/Figures/annulus_2d/density/DensityDBSCAN_FullOptimizationResult_k2_by_m0_dims-0_1.png?raw=true)

Another approach is to use a dendrogram to agglomerate points into clusters. The locations of branches and leaves on a dendrogram are relative to the ordering of the points; for this reason, a dendrogram is sometimes shown alongaide a distance matrix (see `Figures`). The dendrogram shown below employs the "single-linkage" (or shortest link) criterion.

![EX1: Dendrogram](https://github.com/mikeysflix/ClusterAnalysis/blob/master/Figures/annulus_2d/agglomerative/AgglomerativeDendrogram_Single-Linkage_Row0_pos.png?raw=true)

Using this approach, one can make out two distinct groups (that become more distinct as the number of points decreases) - which agrees with our intuition that there are two clusters.


As a second example, let's consider data that is still somewhat globular but perhaps not enough to be considered normally distributed - this has a major effect on barycentric clustering. This becomes apparent by comparing the means against the medians (since the mean and median are identical in an idealized normal distribution). The figure below shows compares the cost elbow and silhouette for different values of `k`.

![EX2: Comparison of Optimal K](https://github.com/mikeysflix/ClusterAnalysis/blob/master/Figures/smiley_face/barycentric/by_elbow/Barycentric_K-Means_Cost_%26_Silhouette_ByK_Random_Centroids_horizontal.png?raw=true)

The elbow method finds that there are 3 clusters; this result is shown below.

![EX2: Full Result - Barycentric](https://github.com/mikeysflix/ClusterAnalysis/blob/master/Figures/smiley_face/barycentric/by_elbow/Barycentric_K-Means_FullOptimizationResult_k3_dims-0_1_Random_Centroids.png
?raw=true)

There is no objective answer as to which clusters are (in)correct, though I still pefer the result obtained using DBSCAN; this is shown below.

![EX2: Full Result - Density](https://github.com/mikeysflix/ClusterAnalysis/blob/master/Figures/smiley_face/density/DensityDBSCAN_FullOptimizationResult_k3_by_m221_dims-0_1.png?raw=true)

Note that there are two eyes inside the smiley face - these may be hard to distinguish from the dendrogram. But for completeness-of-examples-sake, the dendrogram is shown below.

![EX2: Dendrogram](https://github.com/mikeysflix/ClusterAnalysis/blob/master/Figures/smiley_face/agglomerative/AgglomerativeDendrogram_Single-Linkage_Row0_pos.png?raw=true)

**Things To Add:**

1) Regarding barycentric clustering methods, add k-medoids (PAM) method and add methods to find optimal initial barycenters.

2) Regarding density clustering methods, add mean-shift.

3) Regarding agglomerative (hierarchical) clustering, add more linkage criteria (wards, avg, etc) and add methods to cut the dendrogram.

4) Compare speed and accuracy of these "reinvent the wheel" methods to established methods via scipy and scikit, preferably using standard datasets (like MNIST). 


