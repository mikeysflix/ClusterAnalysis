from coordinates_configuration import *

class BarycentricClusteringVisualizer(ClusterEvaluation):

    def __init__(self, data, data_labels=None, distance_metric='euclidean', savedir=None):
        super().__init__(data, data_labels, distance_metric, savedir)
        self.rows_per_distance = np.arange(self.n, dtype=int)
        self._k = None
        self._barycenter_selection_method = None
        self._distances = None
        self._cost = None
        self._cost_history = []
        self._niter = 0
        self._success = False
        self._flags = None
        self._duration = None
        self._meta_optimization = dict()

    @property
    def k(self):
        return self._k

    @property
    def barycenter_selection_method(self):
        return self._barycenter_selection_method

    @property
    def distances(self):
        return self._distances

    @property
    def cost(self):
        return self._cost

    @property
    def cost_history(self):
        return self._cost_history

    @property
    def niter(self):
        return self._niter

    @property
    def success(self):
        return self._success

    @property
    def flags(self):
        return self._flags

    @property
    def duration(self):
        return self._duration

    @property
    def meta_optimization(self):
        return self._meta_optimization

    def select_facecolors_by_clusters(self, facecolors=None, cmap=None):
        if ((facecolors is None) and (cmap is None)) or ((facecolors is not None) and (cmap is not None)):
            raise ValueError("input either facecolors or cmap")
        if cmap is not None:
            facecolors = self.select_facecolors(
                counts=np.arange(self.k, dtype=int),
                cmap=cmap,
                default_color=None)
        else:
            if isinstance(facecolors, str):
                facecolors = list(facecolors)
            if isinstance(facecolors, (tuple, list, np.ndarray)):
                nc = len(facecolors)
                if nc < self.k:
                    raise ValueError("{} facecolors for {} clusters".format(nc, self.k))
            else:
                raise ValueError("invalid type(facecolors): {}".format(type(facecolors)))
        return facecolors

    def subview_data_classification(self, fig, ax, dims, facecolors, with_clusters=False, with_barycenters=False, with_legend=False):
        if not self.success:
            raise ValueError("data is not classified")
        if not any([with_clusters, with_barycenters]):
            raise ValueError("with_clusters and/or with_barycenters should be set to True")
        barycenter_id = 'Centroid' if self.optimization_method in ('k-means', 'k-medians') else 'Medoid'
        # ax.set_facecolor('lightgray')
        for (i, cluster), barycenter, facecolor in zip(self.clusters.items(), self.barycenters, facecolors):
            cluster_args = [cluster[:, dim] for dim in dims]
            barycenter_args = [barycenter[dim] for dim in dims]
            if dims.size == 1:
                cluster_args.append(
                    np.zeros(cluster_args[0].size, dtype=int))
                barycenter_args.append(
                    np.zeros(barycenter_args[0].size, dtype=int))
            if with_clusters:
                ax.scatter(
                    *cluster_args,
                    label='Cluster ${}$'.format(i+1) if not with_barycenters else None,
                    marker='.',
                    s=2,
                    facecolor=facecolor,
                    alpha=0.8)
            if with_barycenters:
                ax.scatter(
                    *barycenter_args,
                    label='{} ${}$'.format(barycenter_id, i+1),
                    marker='*',
                    s=2,
                    facecolor=facecolor,
                    alpha=0.8)
        xlabel = self.get_vector_dimension_label(dims[0])
        ax.set_xlabel(
            xlabel,
            fontsize=self.labelsize)
        ax.xaxis.set_minor_locator(
            ticker.AutoMinorLocator())
        if dims.size > 1:
            ylabel = self.get_vector_dimension_label(dims[1])
            ax.set_ylabel(
                ylabel,
                fontsize=self.labelsize)
            ax.yaxis.set_minor_locator(
                ticker.AutoMinorLocator())
        if dims.size == 3:
            zlabel = self.get_vector_dimension_label(dims[2])
            ax.set_zlabel(
                zlabel,
                fontsize=self.labelsize)
            ax.zaxis.set_minor_locator(
                ticker.AutoMinorLocator())
            for axis in ('x', 'y', 'z'):
                ax.tick_params(
                    axis=axis,
                    labelsize=self.ticksize)
        else:
            if dims.size == 1:
                ax.tick_params(
                    axis='x',
                    labelsize=self.ticksize)
                ax.set_yticks([])
                ax.grid(
                    axis='x',
                    color='k',
                    linestyle=':',
                    alpha=0.3)
            else:
                ax.tick_params(
                    axis='both',
                    labelsize=self.ticksize)
                ax.grid(
                    color='k',
                    linestyle=':',
                    alpha=0.3)
        ax.set_title(
            '$k={}$ Clusters via {}'.format(
                self.k,
                self.optimization_method.title()),
            fontsize=self.titlesize)
        if with_legend:
            handles, labels = ax.get_legend_handles_labels()
            self.subview_legend(
                fig=fig,
                ax=ax,
                handles=handles,
                labels=labels,
                title='Barycentric Method ({} Distance)'.format(self.distance_metric.title()))
        return ax

    def view_data_classification(self, facecolors=None, cmap=None, dims=None, with_clusters=False, with_barycenters=False, save=False, **kwargs):
        if not self.success:
            raise ValueError("data is not classified")
        facecolors = self.select_facecolors_by_clusters(
            facecolors=facecolors,
            cmap=cmap)
        dims = self.select_viewing_dimensions(dims)
        if dims.size == 3:
            fig = plt.figure(**kwargs)
            ax = fig.add_subplot(1, 1, 1, projection='3d')
        else:
            fig, ax = plt.subplots(**kwargs)
        self.subview_data_classification(
            fig=fig,
            ax=ax,
            dims=dims,
            facecolors=facecolors,
            with_clusters=with_clusters,
            with_barycenters=with_barycenters,
            with_legend=True)
        ## save or show figure
        if save:
            savename = 'Barycentric_{}_Classification_k{}_dims-{}'.format(
                self.optimization_method.title(),
                self.k,
                '_'.join(dims.astype(str)))
            if self.barycenter_selection_method is not None:
                savename += '_{}'.format(self.barycenter_selection_method.title())
            savename = savename.replace(' ', '_')
        else:
            savename = None
        self.display_image(fig, savename)

    def subview_cost_history(self, fig, ax, with_legend=False):
        if not self.success:
            raise ValueError("data is not classified")
        x = np.arange(self.cost_history.size, dtype=int)
        ax.set_xlabel(
            'Number of Iterations',
            fontsize=self.labelsize)
        ax.set_ylabel(
            'Cost',
            fontsize=self.labelsize)
        ax.plot(
            x,
            self.cost_history,
            color='darkorange',
            marker='.',
            markersize=10,
            label='Cost')
        ax.xaxis.set_minor_locator(
            ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(
            ticker.AutoMinorLocator())
        ax.tick_params(
            axis='both',
            labelsize=self.ticksize)
        ax.grid(
            color='k',
            linestyle=':',
            alpha=0.3)
        ax.set_title(
            'Cost History for $k={}$ Clusters via {}'.format(
                self.k,
                self.optimization_method.title()),
            fontsize=self.titlesize)
        if with_legend:
            handles, labels = ax.get_legend_handles_labels()
            self.subview_legend(
                fig=fig,
                ax=ax,
                handles=handles,
                labels=labels,
                title='Barycentric Method ({} Distance)'.format(self.distance_metric.title()))
        return ax

    def view_cost_history(self, save=False, **kwargs):
        if not self.success:
            raise ValueError("data is not classified")
        fig, ax = plt.subplots(**kwargs)
        self.subview_cost_history(
            fig=fig,
            ax=ax,
            with_legend=True)
        ## save or show figure
        if save:
            savename = 'Barycentric_{}_CostHistory_k{}'.format(
                self.optimization_method.title(),
                self.k)
            if self.barycenter_selection_method is not None:
                savename += '_{}'.format(self.barycenter_selection_method.title())
            savename = savename.replace(' ', '_')
        else:
            savename = None
        self.display_image(fig, savename)

    def view_silhouette(self, facecolors=None, cmap=None, show_coefficient=False, save=False, **kwargs):
        if not self.success:
            raise ValueError("data is not classified")
        facecolors = self.select_facecolors_by_clusters(
            facecolors=facecolors,
            cmap=cmap)
        fig, ax = plt.subplots(**kwargs)
        self.subview_silhouette(
            fig=fig,
            ax=ax,
            facecolors=facecolors,
            with_legend=True,
            dy=1,
            show_coefficient=show_coefficient)
        if save:
            savename = 'Barycentric_{}_Silhouette_k{}'.format(
                self.optimization_method.title(),
                self.k)
            if self.barycenter_selection_method is not None:
                savename += "_{}".format(
                    self.barycenter_selection_method.title())
            savename = savename.replace(' ', '_')
        else:
            savename = None
        self.display_image(fig, savename)

    def subview_silhouette_by_k(self, fig, ax, with_legend=False):
        x = self.meta_optimization['info']['k']
        ystd = self.meta_optimization['analysis']['silhouette']['standard deviation']
        z = None # np.zeros(ystds.size)
        for key, facecolor, linestyle in zip(('mean', 'median'), ('red', 'blue'), (':', '--')):
            y = self.meta_optimization['analysis']['silhouette'][key]
            ax.errorbar(
                x,
                y,
                ystd,
                z,
                fmt='none',
                ecolor=facecolor,
                alpha=0.5,
                capsize=5,
                label='{} $±$ standard deviation'.format(key)
                )
            ax.plot(
                x,
                y,
                color=facecolor,
                alpha=0.5,
                marker='.',
                markersize=10,
                linestyle=linestyle)
        ax.set_xlabel(
            'k',
            fontsize=self.labelsize)
        ax.set_ylabel(
            'Silhouette Coefficient',
            fontsize=self.labelsize)
        ax.set_xticks(x)
        ax.yaxis.set_minor_locator(
            ticker.AutoMinorLocator())
        ax.tick_params(
            axis='both',
            labelsize=self.ticksize)
        ax.grid(
            color='k',
            linestyle=':',
            alpha=0.3)
        ax.set_title(
            '{} Silhouette Coefficient via {:,} runs per k'.format(
                self.meta_optimization['info']['optimization_method'].title(),
                self.meta_optimization['info']['nruns']),
            fontsize=self.titlesize)
        if with_legend:
            handles, labels = ax.get_legend_handles_labels()
            self.subview_legend(
                fig=fig,
                ax=ax,
                handles=handles,
                labels=labels,
                title='Barycentric Method ({} Distance)'.format(self.distance_metric.title()))
        return ax

    def view_silhouette_by_k(self, save=False, **kwargs):
        fig, ax = plt.subplots(**kwargs)
        self.subview_silhouette_by_k(
            fig=fig,
            ax=ax,
            with_legend=True)
        if save:
            savename = 'Barycentric_{}_SilhouetteByK_{}'.format(
                self.meta_optimization['info']['optimization_method'].title(),
                self.meta_optimization['info']['barycenter_selection_method'].title())
            savename = savename.replace(' ', '_')
        else:
            savename = None
        self.display_image(fig, savename)

    def subview_cost_by_k(self, fig, ax, with_legend=False):
        x = self.meta_optimization['info']['k']
        for key, facecolor, linestyle in zip(('mean', 'median'), ('red', 'blue'), (':', '--')):
            y = self.meta_optimization['analysis']['cost'][key]
            dev = self.meta_optimization['analysis']['cost']['standard deviation']
            ax.errorbar(
                x,
                y,
                dev,
                None, # np.zeros(dev.size),
                fmt='none',
                ecolor=facecolor,
                alpha=0.5,
                capsize=5,
                label='{} $±$ standard deviation'.format(key)
                )
            ax.plot(
                x,
                y,
                color=facecolor,
                alpha=0.5,
                marker='.',
                markersize=10,
                linestyle=linestyle)
        ax.set_xlabel(
            'k',
            fontsize=self.labelsize)
        ax.set_ylabel(
            'Cost',
            fontsize=self.labelsize)
        ax.set_xticks(x)
        ax.yaxis.set_minor_locator(
            ticker.AutoMinorLocator())
        ax.tick_params(
            axis='both',
            labelsize=self.ticksize)
        ax.grid(
            color='k',
            linestyle=':',
            alpha=0.3)
        ax.set_title(
            '{} Objective Cost via {:,} runs per k'.format(
                self.meta_optimization['info']['optimization_method'].title(),
                self.meta_optimization['info']['nruns']),
            fontsize=self.titlesize)
        if with_legend:
            handles, labels = ax.get_legend_handles_labels()
            self.subview_legend(
                fig=fig,
                ax=ax,
                handles=handles,
                labels=labels,
                title='Barycentric Method ({} Distance)'.format(self.distance_metric.title()))
        return ax

    def view_cost_by_k(self, save=False, **kwargs):
        fig, ax = plt.subplots(**kwargs)
        self.subview_cost_by_k(
            fig=fig,
            ax=ax,
            with_legend=True)
        if save:
            savename = 'Barycentric_{}_CostByK_{}'.format(
                self.meta_optimization['info']['optimization_method'].title(),
                self.meta_optimization['info']['barycenter_selection_method'].title())
            savename = savename.replace(' ', '_')
        else:
            savename = None
        self.display_image(fig, savename)

    def view_full_optimization_result(self, facecolors=None, cmap=None, dims=None, with_clusters=False, with_barycenters=False, show_coefficient=False, save=False, **kwargs):
        if not self.success:
            raise ValueError("data is not classified")
        facecolors = self.select_facecolors_by_clusters(
            facecolors=facecolors,
            cmap=cmap)
        dims = self.select_viewing_dimensions(dims)
        nrows = ncols = 2
        axes = []
        p = 0
        fig = plt.figure(**kwargs)
        for i in range(nrows):
            for j in range(ncols):
                if (dims.size == 3) and (i == 0):
                    ax = fig.add_subplot(
                        nrows,
                        ncols,
                        p + 1,
                        projection='3d')
                else:
                    ax = fig.add_subplot(
                        nrows,
                        ncols,
                        p + 1)
                axes.append(ax)
                p += 1
        (ax_top_left, ax_top_right, ax_btm_left, ax_btm_right) = axes
        self.subview_unclassified_data(
            fig=fig,
            ax=ax_top_left,
            dims=dims,
            with_legend=False)
        self.subview_silhouette(
            fig=fig,
            ax=ax_btm_left,
            facecolors=facecolors,
            with_legend=False,
            dy=1,
            show_coefficient=False)
        self.subview_cost_history(
            fig=fig,
            ax=ax_btm_right,
            with_legend=False)
        self.subview_data_classification(
            fig=fig,
            ax=ax_top_right,
            dims=dims,
            facecolors=facecolors,
            with_clusters=with_clusters,
            with_barycenters=with_barycenters,
            with_legend=False)
        handles, labels = [], []
        if show_coefficient:
            ax_btm_left = self.subview_silhouette_coefficient_line(
                ax=ax_btm_left)
            _handles, _labels = ax_btm_left.get_legend_handles_labels()
            for handle, label in zip(_handles, _labels):
                if 'Coefficient' in label:
                    handles.extend(_handles)
                    labels.extend(_labels)
        for ax in (ax_top_left, ax_top_right, ax_btm_right):
            _handles, _labels = ax.get_legend_handles_labels()
            handles.extend(_handles)
            labels.extend(_labels)
        fig.subplots_adjust(hspace=0.35, wspace=0.3)
        self.subview_legend(
            fig=fig,
            ax=ax_top_left,
            handles=handles,
            labels=labels,
            title='Barycentric Method ({} Distance)'.format(self.distance_metric.title()))
        ## save or show figure
        if save:
            savename = 'Barycentric_{}_FullOptimizationResult_k{}_dims-{}'.format(
                self.optimization_method.title(),
                self.k,
                '_'.join(dims.astype(str)))
            if self.barycenter_selection_method is not None:
                savename += '_{}'.format(self.barycenter_selection_method.title())
            savename = savename.replace(' ', '_')
        else:
            savename = None
        self.display_image(fig, savename)

    def view_cost_and_silhouette_by_k(self, layout=None, save=False, **kwargs):
        if layout is None:
            for _layout in ('horizontal', 'vertical'):
                self.view_cost_and_silhouette_by_k(
                    layout=_layout,
                    save=save,
                    **kwargs)
        else:
            if layout == 'horizontal':
                nrows, ncols = 1, 2
            elif layout == 'vertical':
                nrows, ncols = 2, 1
            else:
                raise ValueError("invalid layout: {}".format(layout))
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                sharex=True,
                **kwargs)
            (ax1, ax2) = axes
            self.subview_cost_by_k(
                fig=fig,
                ax=ax1,
                with_legend=False)
            self.subview_silhouette_by_k(
                fig=fig,
                ax=ax2,
                with_legend=True)
            ax1.set_xlabel('', fontsize=self.labelsize)
            if save:
                savename = 'Barycentric_{}_Cost_&_Silhouette_ByK_{}'.format(
                    self.meta_optimization['info']['optimization_method'].title(),
                    self.meta_optimization['info']['barycenter_selection_method'].title())
                savename += '_{}'.format(layout)
                savename = savename.replace(' ', '_')
            else:
                savename = None
            self.display_image(fig, savename)

class BarycentricClustering(BarycentricClusteringVisualizer):

    def __init__(self, data, data_labels=None, distance_metric='euclidean', savedir=None):
        super().__init__(data, data_labels, distance_metric, savedir)
        self._original_barycenter_selection_inputs = dict()
        self.barycenter_selection_fmap = {
            'random centroids' : self.get_random_centroids,
            'random medoids' : self.get_random_medoids,
            'optimal centroids' : self.get_maximally_distant_centroids,
            'optimal medoids' : self.get_maximally_distant_medoids}

    def __repr__(self):
        return 'BarycentricClustering(%r, %r, %r, %r)' % (self.data, self.data_labels, self.distance_metric, self.savedir)

    def __str__(self):
        s = "\nBarycentric Clustering Optimizer\n"
        s += "\n .. optimization method:\n\t{}\n".format(
            self.optimization_method)
        s += "\n .. barycenter selection method:\n\t{}\n".format(
            self.barycenter_selection_method)
        s += "\n .. number of barycenters (or clusters):\n\tk={}\n".format(
            self.k)
        s += "\n .. data:\n\t{} coordinates in {} dimensions\n".format(
            self.n,
            self.ndim)
        if self.k <= 10:
            s += "\n .. barycenters:\n{}\n".format(
                self.barycenters)
        s += "\n .. success:\n\t{}\n".format(
            self.success)
        if 'coefficient' in list(self.silhouette.keys()):
            s += "\n .. silhouette coefficient:\n\t{}\n".format(
                self.silhouette['coefficient'])
        s += "\n .. number of iterations:\n\t{}\n".format(
            self.niter)
        s += "\n .. time elapsed:\n\t{}\n".format(
            self.duration)
        s += "\n .. cost:\n\t{}\n".format(
            self.cost)
        if self.flags is None:
            s += "\n .. flags:\n\t{}\n".format(
                self.flags)
        else:
            s += "\n .. flags:"
            for _, warning_message in self.flags.items():
                s += "\n\t{}".format(warning_message)
            s += "\n"
        return s

    @property
    def original_barycenter_selection_inputs(self):
        return self._original_barycenter_selection_inputs

    @staticmethod
    def get_max_iter(max_iter):
        return int(max(1, max_iter))

    def get_meta_optimization_cost_by_k(self):
        elbow = dict()
        for key in ('mean', 'median', 'standard deviation'):
            elbow[key] = []
        for k in self.meta_optimization['info']['k']:
            cost = self.meta_optimization['cost'][k]
            success = self.meta_optimization['success'][k]
            loc = (success == True)
            if np.any(loc):
                avg_cost, med_cost, std_cost = np.nanmean(cost[loc]), np.nanmedian(cost[loc]), np.nanstd(cost[loc])
            else:
                avg_cost, med_cost, std_cost = np.nan, np.nan, np.nan
            elbow['mean'].append(avg_cost)
            elbow['median'].append(med_cost)
            elbow['standard deviation'].append(std_cost)
        for key, value in elbow.items():
            elbow[key] = np.array(value)
        return elbow

    def get_meta_optimization_silhouette_by_k(self):
        meta_silhouette = dict()
        for key in ('mean', 'median', 'standard deviation'):
            meta_silhouette[key] = []
        for k in self.meta_optimization['info']['k']:
            arr = self.meta_optimization['silhouette'][k]
            success = self.meta_optimization['success'][k]
            loc = (success == True)
            if np.any(loc):
                avg_sil, med_sil, std_sil = np.nanmean(arr[loc]), np.nanmedian(arr[loc]), np.nanstd(arr[loc])
            else:
                avg_sil, med_sil, std_sil = np.nan, np.nan, np.nan
            meta_silhouette['mean'].append(avg_sil)
            meta_silhouette['median'].append(med_sil)
            meta_silhouette['standard deviation'].append(std_sil)
        for key, value in meta_silhouette.items():
            meta_silhouette[key] = np.array(value)
        return meta_silhouette

    def update_meta_optimizations(self, optimization_method, kmin, kmax, nruns=10, barycenter_selection_method=None, max_iter=1e3):
        for k, s in zip((kmin, kmax, nruns), ('kmin', 'kmax', 'nruns')):
            if not isinstance(k, int):
                raise ValueError("invalid type({}): {}".format(type(s), type(k)))
            if k < 1:
                raise ValueError("invalid {}: {}".format(s, k))
        if not kmax > kmin:
            raise ValueError("kmax must be greater than kmin")
        ks = np.arange(kmin, kmax + 1, dtype=int)
        optimizers, costs, success, silhouettes = dict(), dict(), dict(), dict()
        for k in ks:
            _optimizers, _costs, _success, _silhouettes = [], [], [], []
            for i in range(nruns):
                optimizer = BarycentricClustering(
                    data=self.data,
                    data_labels=self.data_labels,
                    distance_metric=self.distance_metric,
                    savedir=self.savedir)
                optimizer.fit(
                    optimization_method=optimization_method,
                    barycenters=None,
                    k=k,
                    barycenter_selection_method=barycenter_selection_method,
                    max_iter=max_iter)
                optimizer.evaluate_silhouette()
                _optimizers.append(optimizer)
                _costs.append(optimizer.cost)
                _success.append(optimizer.success)
                _silhouettes.append(optimizer.silhouette['coefficient'])
            optimizers[k] = _optimizers
            costs[k] = np.array(_costs)
            success[k] = np.array(_success)
            silhouettes[k] = np.array(_silhouettes)
        self._meta_optimization['optimizer'] = optimizers
        self._meta_optimization['cost'] = costs
        self._meta_optimization['success'] = success
        self._meta_optimization['silhouette'] = silhouettes
        self._meta_optimization['info'] = {
            'optimization_method' : optimization_method,
            'nruns' : nruns,
            'k' : ks,
            'barycenter_selection_method' : barycenter_selection_method,
            'max_iter' : max_iter}
        meta_cost = self.get_meta_optimization_cost_by_k()
        meta_silhouette = self.get_meta_optimization_silhouette_by_k()
        self._meta_optimization['analysis'] = {
            'cost' : meta_cost,
            'silhouette' : meta_silhouette}

    def select_optimal_k_by_cost_elbow(self, statistic='mean'):
        if list(self.meta_optimization.keys()) == 0:
            raise ValueError("meta_optimization is not initialized")
        if statistic not in ('mean', 'median'):
            raise ValueError("invalid statistic: {}".format(statistic))
        dy = np.diff(self.meta_optimization['analysis']['cost'][statistic])
        dx = np.diff(self.meta_optimization['info']['k'])
        slope = dy / dx
        loc = np.argmax(np.abs(slope)) + 1
        return self.meta_optimization['info']['k'][loc]

    def select_optimal_k_by_silhouette(self, statistic='mean'):
        if list(self.meta_optimization.keys()) == 0:
            raise ValueError("meta_optimization is not initialized")
        if statistic not in ('mean', 'median'):
            raise ValueError("invalid statistic: {}".format(statistic))
        loc = np.argmax(self.meta_optimization['analysis']['silhouette'][statistic])
        return self.meta_optimization['info']['k'][loc]

    def initialize_optimization_method(self, optimization_method):
        if optimization_method in ('k-means', 'k-medians', 'k-medoids'):
            self._optimization_method = optimization_method
        else:
            raise ValueError("invalid optimization_method: {}".format(optimization_method))

    def get_random_centroids(self, k):
        ubounds, lbounds = np.max(self.data, axis=0), np.min(self.data, axis=0)
        centroids = []
        for ki in range(k):
            centroid = [np.random.uniform(low=lb, high=ub) for lb, ub in zip(ubounds, lbounds)]
            centroids.append(centroid)
        return np.array(centroids)

    def get_random_medoids(self, k):
        loc = np.random.choice(
            np.arange(self.n, dtype=int),
            size=k,
            replace=False)
        medoids = np.copy(self.data[loc, :])
        return medoids

    def get_maximally_distant_centroids(self, k):
        ubounds, lbounds = np.max(self.data, axis=0), np.min(self.data, axis=0)
        centroids = []
        raise ValueError("not yet implemented")
        # return ...

    def get_maximally_distant_medoids(self, k):
        ubounds, lbounds = np.max(self.data, axis=0), np.min(self.data, axis=0)
        medoids = ... # distance matrix
        raise ValueError("not yet implemented")
        # return ...

    def initialize_barycenters(self, barycenters=None, k=None, barycenter_selection_method=None):
        if (barycenters is None) and (k is None):
            raise ValueError("input either barycenters or k")
        if (barycenters is not None) and (k is not None):
            raise ValueError("input either barycenters or k")
        if len(list(self.original_barycenter_selection_inputs.keys())) == 0:
            self._original_barycenter_selection_inputs.update({
                'barycenters' : barycenters,
                'k' : k,
                'barycenter_selection_method' : barycenter_selection_method})
        if barycenters is not None:
            if isinstance(barycenters, (tuple, list, np.ndarray)):
                if not isinstance(barycenters, np.ndarray):
                    barycenters = np.array(barycenters)
                s = len(barycenters.shape)
                if s != 2:
                    raise ValueError("invalid barycenters.shape: {}".format(barycenters.shape))
                if s != self.ndim:
                    raise ValueError("dimensions of barycenters ({}) does not match dimensions of data ({})".format(s, self.ndim))
                self._barycenters = barycenters
                self._k = barycenters.shape[0]
            else:
                raise ValueError("invalid type(barycenters): {}".format(type(barycenters)))
            if self.optimization_method == 'k-medoids':
                data = self.data.tolist()
                for j, medoid in enumerate(self.barycenters):
                    if medoid.tolist() not in data:
                        raise ValueError("{}-th medoid ({}) is not contained in data".format(j, medoid))
        if k is not None:
            if not isinstance(k, (int, np.int64)):
                raise ValueError("invalid type(k): {}".format(type(k)))
            if k <= 1:
                raise ValueError("invalid number of clusters: k={}".format(k))
            if k > self.n:
                raise ValueError("cannot predict {} clusters from {} coordinates".format(k, self.n))
            if barycenter_selection_method not in list(self.barycenter_selection_fmap.keys()):
                raise ValueError("invalid barycenter_selection_method: {}".format(barycenter_selection_method))
            if (self.optimization_method == 'k-medoids') and ('medoid' not in barycenter_selection_method):
                raise ValueError("cannot initialize non-medoids for k-medoids optimization")
            select_barycenters = self.barycenter_selection_fmap[barycenter_selection_method]
            self._barycenters = select_barycenters(k)
            self._k = k
            self._barycenter_selection_method = barycenter_selection_method

    def get_initial_barycenter_status(self):
        s = set(self.membership)
        if len(s) < self.k:
            return False
        else:
            return True

    def verify_initial_barycenters(self):
        s = set(self.membership)
        if len(s) < self.k:
            raise ValueError("initial barycenters are invalid (membership excludes at least one barycenter)")

    def update_distances(self):
        distances = []
        for barycenter in self.barycenters:
            displacement = self.data - barycenter
            distance = self.f_distance(displacement, axis=1)
            distances.append(distance)
        self._distances = np.array(distances).T # shape=(self.n, self.k)

    def update_membership(self):
        self._membership[:] = np.argmin(self.distances, axis=1)

    def update_cost(self):
        self._cost = np.sum(self.distances[self.rows_per_distance, self.membership])
        self._cost_history.append(self.cost)

    def predict_centroids(self, max_iter):
        fmap = {
            'k-means' : np.mean,
            'k-medians' : np.median}
        f = fmap[self.optimization_method]
        _max_iter = self.get_max_iter(max_iter)
        for i in range(_max_iter):
            self._niter += 1
            prev_cost = self.cost
            prev_centroids = np.copy(self.barycenters)
            centroids = []
            for ki in range(self.k):
                loc = (self.membership == ki)
                centroid = f(self.data[loc, :], axis=0)
                centroids.append(centroid)
            centroids = np.array(centroids)
            self._barycenters = centroids
            self.update_distances()
            self.update_membership()
            self.update_cost()
            if np.allclose(self.barycenters, prev_centroids) and np.isclose(self.cost, prev_cost):
                self._success = True
                break

    def predict_medoids(self, max_iter):
        _max_iter = self.get_max_iter(max_iter)
        for i in range(_max_iter):
            self._niter += 1
            medoids = []
            raise ValueError("not yet implemented")

    def update_clusters(self):
        for ki in range(self.k):
            loc = (self.membership == ki)
            self._clusters[ki] = self.data[loc, :]

    def update_flags(self, nresets):
        flags = dict()
        i = 0
        dcost = np.diff(self.cost_history)
        loc = (dcost > 0)
        if np.any(loc):
            i += 1
            cost_history_flag = "({})\tThere was/were {} consecutively-paired iteration(s) during which the objective cost increased; the maximum increase was by {}".format(
                i,
                np.sum(loc),
                np.nanmax(dcost))
            flags['cost history'] = cost_history_flag
        if nresets > 0:
            i += 1
            reset_history_flag = "({})\tThere was/were {} reset(s) to pick the initial centroids".format(
                i,
                nresets)
            flags['reset history'] = reset_history_flag
        if self.optimization_method in ('k-means', 'k-medians'):
            if self.niter > 25:
                i += 1
                barycenter_selection_flag = "({})\tThere was/were {} iteration(s), which might be excessive for this optimization method".format(
                    i,
                    self.niter)
                flags['barycenter selection'] = barycenter_selection_flag
        if len(list(flags.keys())) > 0:
            self._flags = flags

    def fit(self, optimization_method, barycenters=None, k=None, barycenter_selection_method=None, max_iter=1e3):
        t0 = perf_counter()
        self.initialize_optimization_method(optimization_method)
        self.initialize_barycenters(
            barycenters=barycenters,
            k=k,
            barycenter_selection_method=barycenter_selection_method)
        self.update_distances()
        self.update_membership()
        # self.verify_initial_barycenters()
        nresets = 0
        while True:
            state = self.get_initial_barycenter_status()
            if state:
                break
            else:
                self.initialize_barycenters(**self.original_barycenter_selection_inputs)
                self.update_distances()
                self.update_membership()
                nresets += 1
        self.update_cost()
        if self.optimization_method in ('k-means', 'k-medians'):
            self.predict_centroids(
                max_iter=max_iter)
        elif self.optimization_method == 'k-medoids':
            self.predict_medoids(
                max_iter=max_iter)
        else:
            raise ValueError("invalid optimization_method: {}".format(self.optimization_method))
        self.update_clusters()
        self._cost_history = np.array(self._cost_history)
        self.update_flags(nresets)
        self._duration = perf_counter() - t0

    def save_optimization_results(self):
        if not isinstance(self.savedir, str):
            raise ValueError("savedir is not initialized")
        savename = "{}Barycentric_{}_k{}".format(
            self.savedir,
            self.optimization_method.title(),
            self.k)
        if self.barycenter_selection_method is not None:
            savename += "_{}".format(
                self.barycenter_selection_method.title())
            savename = savename.replace(' ', '_')
        with open("{}.txt".format(savename), "w") as text_file:
            text_file.write("{}".format(str(self)))









##
