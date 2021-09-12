from dataset_configuration import *
from backend_methods import *

## initialize directory to save figures
savedir = "..." # None
save = True # False

if __name__ == '__main__':

    ## set reproducible random state
    np.random.seed(0)

    ## get data
    # data = DataSets().get_annular_dataset(
    #     n=500,
    #     ndim=2,
    #     r_inner=5,
    #     r_outer=10)
    # data = DataSets().get_annular_dataset(
    #     n=100,
    #     ndim=3,
    #     r_inner=5,
    #     r_outer=10)
    data = DataSets().get_smiley_face(
        n=1000,
        is_happy=True)

    ## initialize optimizers
    multi_optimizer = MultiOptimizer(
        data=data,
        data_labels=None,
        distance_metric='euclidean', # 'manhattan'
        distance_matrix=None,
        savedir=savedir)

    ## run barycentric optimization
    multi_optimizer.run_barycentric_optimization(
        optimization_method='k-means', # 'k-medians', ### 'k-medoids'
        barycenter_selection_method='random centroids', # 'random medoids'
        barycenters=None,
        k=None,
        kmin=2,
        kmax=10,
        max_iter=100,
        nruns=10,
        optimal_k_method='elbow', # 'silhouette'
        save=True)

    ## run density optimization
    multi_optimizer.run_density_optimization(
        optimization_method='dbscan', ### 'mean-shift'
        minimum_radius=0,
        maximum_radius=3,
        cluster_threshold=5,
        save=True)

    ## run dendrogram (agglomerative) optimization
    multi_optimizer.run_agglomerative_optimization(
        optimization_method='single-linkage', # 'complete-linkage'
        save=True)



##
