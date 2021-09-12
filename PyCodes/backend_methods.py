from barycentric_optimizer_configuration import *
from density_optimizer_configuration import *
from agglomerative_optimizer_configuration import *

class MultiOptimizer():

    def __init__(self, data, data_labels=None, distance_metric='euclidean', distance_matrix=None, savedir=None):
        super().__init__()
        self.barycentric_optimizer = BarycentricClustering(
            data=data,
            data_labels=data_labels,
            distance_metric=distance_metric,
            savedir=savedir)
        self.density_optimizer = DensityClustering(
            data=data,
            data_labels=data_labels,
            distance_metric=distance_metric,
            distance_matrix=distance_matrix,
            savedir=savedir)
        self.dendrogram_optimizer = Dendrogram(
            data=data,
            data_labels=data_labels,
            distance_metric=distance_metric,
            distance_matrix=distance_matrix,
            savedir=savedir)

    def run_barycentric_optimization(self, optimization_method='k-means', barycenter_selection_method=None, barycenters=None, k=None, kmin=2, kmax=10, max_iter=100, nruns=10, optimal_k_method='elbow', save=False):

        ## view unclassified data
        self.barycentric_optimizer.view_unclassified_data(
            # dims=(0, 1),
            figsize=(12, 7),
            save=save)

        ## initialize barycenters
        if k is None:
            self.barycentric_optimizer.update_meta_optimizations(
                optimization_method=optimization_method,
                kmin=kmin,
                kmax=kmax,
                nruns=nruns,
                barycenter_selection_method=barycenter_selection_method,
                max_iter=max_iter)
            ## view meta-optimizations
            self.barycentric_optimizer.view_cost_by_k(
                figsize=(12, 7),
                save=save)
            self.barycentric_optimizer.view_silhouette_by_k(
                figsize=(12, 7),
                save=save)
            self.barycentric_optimizer.view_cost_and_silhouette_by_k(
                # layout=None,
                figsize=(12, 7),
                save=save)
            ## initialize barycenters
            barycenters = None
            if optimal_k_method == 'elbow':
                k = self.barycentric_optimizer.select_optimal_k_by_cost_elbow(
                    statistic='mean')
            elif optimal_k_method == 'silhouette':
                k = self.barycentric_optimizer.select_optimal_k_by_silhouette(
                    statistic='mean')
            else:
                raise ValueError("invalid optimal_k_method: {}".format(optimal_k_method))
        self.barycentric_optimizer.initialize_barycenters(
            barycenters=barycenters,
            k=k,
            barycenter_selection_method=barycenter_selection_method)

        ## run baryentric optimization && evaluate clustering
        self.barycentric_optimizer.fit(
            optimization_method=optimization_method,
            barycenters=barycenters,
            k=k,
            barycenter_selection_method=barycenter_selection_method,
            max_iter=max_iter)
        self.barycentric_optimizer.evaluate_silhouette()

        ## view results of optimization
        self.barycentric_optimizer.view_data_classification(
            # dims=(0, 1),
            with_clusters=True,
            with_barycenters=True,
            cmap='viridis',
            # facecolors=('darkorange', 'steelblue', 'darkgreen', 'red', 'rebeccapurple'),
            figsize=(12, 7),
            save=save)
        self.barycentric_optimizer.view_cost_history(
            figsize=(12, 7),
            save=save)
        self.barycentric_optimizer.view_silhouette(
            show_coefficient=True,
            cmap='viridis',
            # facecolors=('darkorange', 'steelblue', 'darkgreen', 'red', 'rebeccapurple'),
            figsize=(12, 7),
            save=save)
        self.barycentric_optimizer.view_full_optimization_result(
            # facecolors=('darkorange', 'steelblue', 'darkgreen', 'red', 'rebeccapurple'),
            cmap='viridis',
            # dims=None,
            with_clusters=True,
            with_barycenters=True,
            show_coefficient=True,
            figsize=(12, 7),
            save=save)

        ## save results
        print(self.barycentric_optimizer)
        self.barycentric_optimizer.save_optimization_results()

    def run_density_optimization(self, optimization_method='dbscan', minimum_radius=0, maximum_radius=3, cluster_threshold=5, save=False):

        ## view unclassified data
        self.density_optimizer.view_unclassified_data(
            # dims=(0, 1),
            figsize=(12, 7),
            save=save)

        ## run density optimization && evaluate clustering
        self.density_optimizer.fit(
            optimization_method=optimization_method,
            cluster_threshold=cluster_threshold,
            maximum_radius=maximum_radius,
            minimum_radius=minimum_radius)

        ## view results of optimization
        for differentiate_noise in (True, False):
            self.density_optimizer.view_data_classification(
                # dims=(0, 1),
                with_clusters=True,
                with_noise=True,
                cmap='RdBu',
                differentiate_noise=differentiate_noise,
                # facecolors=('darkorange', 'steelblue', 'darkgreen', 'red', 'rebeccapurple'),
                figsize=(12, 7),
                save=save)
            self.density_optimizer.view_membership_matrix(
                cmap='RdBu',
                with_colorbar=True,
                # with_text_labels=True,
                differentiate_noise=differentiate_noise,
                show_upper_triangle=True,
                show_lower_triangle=True,
                save=save,
                figsize=(12, 7))
        self.density_optimizer.view_distance_matrix(
            cmap='RdBu',
            with_divergent_colors=True,
            with_colorbar=True,
            show_upper_triangle=True,
            show_lower_triangle=True,
            save=save,
            figsize=(12, 7))
        self.density_optimizer.view_full_optimization_result(
            cmap='RdBu',
            # dims=None,
            with_clusters=True,
            with_noise=True,
            # differentiate_noise=True,
            show_upper_triangle=True,
            show_lower_triangle=True,
            save=save,
            figsize=(12, 7))

        ## save results
        print(self.density_optimizer)
        self.density_optimizer.save_optimization_results()

    def run_agglomerative_optimization(self, optimization_method='single-linkage', save=False):

        ## view unclassified data
        self.dendrogram_optimizer.view_unclassified_data(
            # dims=(0, 1),
            figsize=(12, 7),
            save=save)

        ## run dendrogram optimization && evaluate clustering
        self.dendrogram_optimizer.fit(
            optimization_method=optimization_method)

        ## view dendrogram
        if self.dendrogram_optimizer.data_labels is None:
            with_text_labels = False
        else:
            if all(dl is None for dl in self.dendrogram_optimizer.data_labels):
                with_text_labels = False
            else:
                with_text_labels = True
        self.dendrogram_optimizer.view_dendrogram(
            # vector_id='column',
            with_text_labels=with_text_labels,
            # negative_direction=True,
            figsize=(12, 7),
            save=save)

        ## view dendrogram alongside distance matrix
        dendrogram_locations = [
            dict(
                show_top=True,
                show_left=True),
            dict(
                show_bottom=True,
                show_right=True),
            dict(
                show_top=True,
                show_left=True,
                show_bottom=True,
                show_right=True)]
        for show_kws in dendrogram_locations:
            self.dendrogram_optimizer.view_dendrogram_and_distance_matrix(
                # facecolors=None,
                cmap='viridis',
                with_text_labels=with_text_labels,
                sep='/',
                save=save,
                figsize=(12, 7),
                **show_kws)

        ## save results
        print(self.dendrogram_optimizer)
        self.dendrogram_optimizer.save_optimization_results()


##
