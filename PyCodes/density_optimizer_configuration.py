import itertools
from collections import Counter
from coordinates_configuration import *

class DensityClusteringVisualizer(ClusterEvaluation):

    def __init__(self, data, data_labels=None, distance_metric='euclidean', distance_matrix=None, savedir=None):
        super().__init__(data, data_labels, distance_metric, savedir)
        self.initialize_distance_matrix(distance_matrix)
        if self.distance_matrix.shape[0] <= 2:
            raise ValueError("distance_matrix must contain at least two rows/columns")
        self._reducible_matrix = self.mask_diagonal_entries(
            self.distance_matrix,
            k=0)
        for k in range(1, self.distance_matrix.shape[0]):
            self._reducible_matrix = self.mask_diagonal_entries(
                self._reducible_matrix,
                k=-k)
        self._k = None ## number of clusters
        self._m = None ## number of noise
        self._membership *= 0
        self._membership_matrix = np.zeros(self.distance_matrix.shape, dtype=int)
        self._mask_matrix = None
        self._minimum_radius = None
        self._maximum_radius = None
        self._cluster_threshold = None
        self._visited = set()
        self._noise = dict()
        self._success = False
        self._flags = None
        self._duration = None

    @property
    def reducible_matrix(self):
        return self._reducible_matrix

    @property
    def k(self):
        return self._k

    @property
    def m(self):
        return self._m

    @property
    def mask_matrix(self):
        return self._mask_matrix

    @property
    def minimum_radius(self):
        return self._minimum_radius

    @property
    def maximum_radius(self):
        return self._maximum_radius

    @property
    def cluster_threshold(self):
        return self._cluster_threshold

    @property
    def visited(self):
        return self._visited

    @property
    def noise(self):
        return self._noise

    @property
    def success(self):
        return self._success

    @property
    def flags(self):
        return self._flags

    @property
    def duration(self):
        return self._duration

    def get_divergent_colormap_normalization(self, cmap, with_noise=False, differentiate_noise=False):
        vmax = self.k
        if (self.m > 0) and with_noise:
            vcenter = 0
            if differentiate_noise:
                vmin = -1 * self.m
            else:
                vmin = -1
        else:
            # norm = Normalize(
            #     vmin=vmin,
            #     vmax=vmax)
            vmin = -1
            vcenter = -0.5
        norm = TwoSlopeNorm(
            vcenter=vcenter,
            vmin=vmin,
            vmax=vmax)
        return norm

    def get_noise_ticks_and_labels(self, differentiate_noise):
        if self.m > 0:
            if differentiate_noise:
                ticks = np.arange(-self.m, 0, dtype=int)
                ticklabels = ['Noise ${}$'.format(mi + 1) for mi in range(self.m)]
            else:
                ticks = [-1]
                ticklabels = ['Noise']
        else:
            ticks = []
            ticklabels = []
        return ticks, ticklabels

    def get_cluster_ticks_and_labels(self):
        if self.k > 0:
            ticks = np.arange(1, self.k+2, dtype=int)
            ticklabels = ['Cluster ${}$'.format(ki + 1) for ki in range(self.k + 1)]
        else:
            ticks = []
            ticklabels = []
        return ticks, ticklabels

    def get_grouped_ticks_and_labels(self, differentiate_noise):
        noise_ticks, noise_ticklabels = self.get_noise_ticks_and_labels(
            differentiate_noise=differentiate_noise)
        cluster_ticks, cluster_ticklabels = self.get_cluster_ticks_and_labels()
        ticks = np.concatenate((noise_ticks, [0], cluster_ticks))
        ticklabels = np.concatenate((noise_ticklabels, ['Non-reachable'], cluster_ticklabels))
        tick_offsets = ticks - 0.5
        ticks = ticks[:-1]
        ticklabels = ticklabels[:-1]
        return ticks, ticklabels, tick_offsets

    def select_facecolors_by_clusters(self, facecolors=None, cmap=None, with_noise=False, differentiate_noise=False):
        if ((facecolors is None) and (cmap is None)) or ((facecolors is not None) and (cmap is not None)):
            raise ValueError("input either facecolors or cmap")
        if cmap is not None:
            norm = self.get_divergent_colormap_normalization(
                cmap=cmap,
                with_noise=with_noise,
                differentiate_noise=differentiate_noise)
            f_cmap = plt.get_cmap(cmap)
            cluster_counts = np.arange(self.k, dtype=int) + 1
            noise_counts = (np.arange(self.m, dtype=int) + 1) * -1
            counts = np.concatenate((cluster_counts, [0], noise_counts))
            facecolors = f_cmap(norm(counts))
        else:
            if isinstance(facecolors, str):
                facecolors = list(facecolors)
            if isinstance(facecolors, (tuple, list, np.ndarray)):
                nc = len(facecolors)
                if nc < self.k + self.m + 1: ## unvisited
                    raise ValueError("{} facecolors for {} clusters".format(nc, self.k + self.m))
            else:
                raise ValueError("invalid type(facecolors): {}".format(type(facecolors)))
        return facecolors

    def subview_data_classification(self, fig, ax, dims, facecolors, with_clusters=False, with_noise=False, differentiate_noise=False, with_legend=False):
        if not self.success:
            raise ValueError("data is not classified")
        if not any([with_clusters, with_noise]):
            raise ValueError("with_clusters and/or with_noise should be set to True")
        if differentiate_noise and not with_noise:
            raise ValueError("with_noise={} is incompatible with differentiate_noise={}".format(with_noise, differentiate_noise))
        if with_clusters:
            for (i, cluster), facecolor in zip(self.clusters.items(), facecolors[:self.k]):
                cluster_args = [cluster[:, dim] for dim in dims]
                if dims.size == 1:
                    cluster_args.append(
                        np.zeros(cluster_args[0].size, dtype=int))
                ax.scatter(
                    *cluster_args,
                    label='Cluster ${}$'.format(i+1),
                    marker='.',
                    s=2,
                    facecolor=facecolor,
                    alpha=0.8)
        if with_noise:
            if differentiate_noise:
                for (j, noise), facecolor in zip(self.noise.items(), facecolors[-self.m:]):
                    noise_args = [noise[:, dim] for dim in dims]
                    if dims.size == 1:
                        noise_args.append(
                            np.zeros(noise_args[0].size, dtype=int))
                    ax.scatter(
                        *noise_args,
                        label='Noise ${}$'.format(j),
                        marker='.',
                        s=2,
                        facecolor=facecolor, # 'lightgray',
                        alpha=0.8)
            else:
                for j, noise in self.noise.items():
                    noise_args = [noise[:, dim] for dim in dims]
                    if dims.size == 1:
                        noise_args.append(
                            np.zeros(noise_args[0].size, dtype=int))
                    ax.scatter(
                        *noise_args,
                        label='Noise' if j == 0 else None,
                        marker='.',
                        s=2,
                        facecolor=facecolors[-1], # 'lightgray',
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
                self.optimization_method.upper()),
            fontsize=self.titlesize)
        if with_legend:
            handles, labels = ax.get_legend_handles_labels()
            self.subview_legend(
                fig=fig,
                ax=ax,
                handles=handles,
                labels=labels,
                title='Density Method ({} Distance)'.format(self.distance_metric.title()))
        return ax

    def view_data_classification(self, facecolors=None, cmap=None, dims=None, with_clusters=False, with_noise=False, differentiate_noise=False, save=False, **kwargs):
        if not self.success:
            raise ValueError("data is not classified")
        if differentiate_noise and not with_noise:
            raise ValueError("with_noise={} is incompatible with differentiate_noise={}".format(with_noise, differentiate_noise))
        facecolors = self.select_facecolors_by_clusters(
            facecolors=facecolors,
            cmap=cmap,
            with_noise=with_noise,
            differentiate_noise=differentiate_noise)
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
            with_noise=with_noise,
            differentiate_noise=differentiate_noise,
            with_legend=True)
        if save:
            savename = 'Density_{}_Classification_k{}_by_m{}'.format(
                self.optimization_method.title(),
                self.k,
                self.m)
            if with_noise:
                if differentiate_noise:
                    savename += '_withNoiseDf'
                else:
                    savename += '_withNoise'
            savename += '_dims-{}'.format(
                '_'.join(dims.astype(str)))
            savename = savename.replace(' ', '_')
        else:
            savename = None
        self.display_image(fig, savename)

    def subview_distance_matrix(self, fig, ax, cmap, with_divergent_colors=False, with_colorbar=False, with_text_labels=False, show_upper_triangle=False, show_lower_triangle=False, **kwargs):
        if not any([show_upper_triangle, show_lower_triangle]):
            raise ValueError("show_upper_triangle and/or show_lower_triangle must be True")
        mat = self.select_from_matrix(
            mat=self.mask_diagonal_entries(self.distance_matrix),
            show_lower_triangle=show_lower_triangle,
            show_upper_triangle=show_upper_triangle)
        vmin = 0
        vcenter = np.nanmean(mat)
        vmax = np.nanmax(mat)
        if with_divergent_colors:
            norm = TwoSlopeNorm(
                vcenter=vcenter,
                vmin=vmin,
                vmax=vmax)
        else:
            norm = Normalize(
                vmin=vmin,
                vmax=vmax)
        ax, im = self.subview_matrix(
            fig=fig,
            ax=ax,
            mat=mat,
            cmap=cmap,
            norm=norm,
            with_text_labels=with_text_labels)
        ax.xaxis.set_minor_locator(
            ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(
            ticker.AutoMinorLocator())
        ax.tick_params(axis='both', labelsize=self.ticksize)
        ax.grid(
            color='white',
            linestyle=':',
            alpha=0.3)
        xlabel = ylabel = "Data"
        ax.set_xlabel(
            xlabel,
            fontsize=self.labelsize)
        ax.set_ylabel(
            ylabel,
            fontsize=self.labelsize)
        if with_colorbar:
            cbar = fig.colorbar(
                im,
                extend='max',
                **kwargs)
            cbar.ax.yaxis.set_minor_locator(
                ticker.AutoMinorLocator())
            cbar.ax.set_ylim([vmin, vmax])
            cbar.ax.tick_params(axis='y', labelsize=self.ticksize)
            cbar.ax.set_ylabel(
                '{} Distance'.format(
                    self.distance_metric.title()),
                fontsize=self.labelsize)
        ax.set_title(
            'Distance Matrix ({})'.format(self.distance_metric.title()) if not with_colorbar else 'Distance Matrix',
            fontsize=self.titlesize)
        return ax, im

    def view_distance_matrix(self, cmap, with_divergent_colors=False, with_colorbar=False, with_text_labels=False, show_upper_triangle=False, show_lower_triangle=False, save=False, **kwargs):
        if not any([show_upper_triangle, show_lower_triangle]):
            raise ValueError("show_upper_triangle and/or show_lower_triangle must be True")
        fig, ax = plt.subplots(**kwargs)
        self.subview_distance_matrix(
            fig=fig,
            ax=ax,
            cmap=cmap,
            with_divergent_colors=with_divergent_colors,
            with_colorbar=with_colorbar,
            with_text_labels=with_text_labels,
            show_upper_triangle=show_upper_triangle,
            show_lower_triangle=show_lower_triangle)
        if save:
            savename = 'Density_{}_DistanceMatrix'.format(
                self.distance_metric.title())
            savename = savename.replace(' ', '_')
        else:
            savename = None
        self.display_image(fig, savename)

    def subview_membership_matrix(self, fig, ax, cmap, with_colorbar=False, with_text_labels=False, differentiate_noise=False, show_upper_triangle=False, show_lower_triangle=False, **kwargs):
        mat = self.select_from_matrix(
            mat=self.mask_diagonal_entries(self.membership_matrix),
            show_lower_triangle=show_lower_triangle,
            show_upper_triangle=show_upper_triangle)
        norm = self.get_divergent_colormap_normalization(
            cmap=cmap,
            with_noise=True,
            differentiate_noise=differentiate_noise)
        cticks, cticklabels, ctick_offsets = self.get_grouped_ticks_and_labels(
            differentiate_noise=differentiate_noise)
        ax, im = self.subview_matrix(
            fig=fig,
            ax=ax,
            mat=mat,
            cmap=cmap,
            norm=norm,
            with_text_labels=with_text_labels)
        ax.xaxis.set_minor_locator(
            ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(
            ticker.AutoMinorLocator())
        ax.tick_params(axis='both', labelsize=self.ticksize)
        ax.grid(
            color='white',
            linestyle=':',
            alpha=0.3)
        xlabel = ylabel = "Data"
        ax.set_xlabel(
            xlabel,
            fontsize=self.labelsize)
        ax.set_ylabel(
            ylabel,
            fontsize=self.labelsize)
        if with_colorbar:
            cbar = fig.colorbar(
                im,
                ticks=cticks,
                boundaries=ctick_offsets,
                **kwargs)
            cbar.ax.set_yticklabels(
                cticklabels,
                fontsize=self.ticksize)
        ax.set_title(
            'Membership Matrix via {}'.format(
                self.optimization_method.upper()),
            fontsize=self.titlesize)
        return ax, im

    def view_membership_matrix(self, cmap, with_colorbar=False, with_text_labels=False, differentiate_noise=False, show_upper_triangle=False, show_lower_triangle=False, save=False, **kwargs):
        if not self.success:
            raise ValueError("data is not classified")
        if not any([show_upper_triangle, show_lower_triangle]):
            raise ValueError("show_upper_triangle and/or show_lower_triangle must be True")
        fig, ax = plt.subplots(**kwargs)
        self.subview_membership_matrix(
            fig=fig,
            ax=ax,
            cmap=cmap,
            with_colorbar=with_colorbar,
            with_text_labels=with_text_labels,
            differentiate_noise=differentiate_noise,
            show_upper_triangle=show_upper_triangle,
            show_lower_triangle=show_lower_triangle)
        if save:
            savename = 'Density_{}_MembershipMatrix_k{}_by_m{}'.format(
                self.optimization_method.title(),
                self.k,
                self.m)
            if differentiate_noise:
                savename += '_withNoiseDf'
            else:
                savename += '_withNoise'
            savename = savename.replace(' ', '_')
        else:
            savename = None
        self.display_image(fig, savename)

    def view_full_optimization_result(self, cmap, dims=None, with_text_labels=False, with_clusters=False, with_noise=False, differentiate_noise=False, show_upper_triangle=False, show_lower_triangle=False, save=False, **kwargs):
        if not self.success:
            raise ValueError("data is not classified")
        facecolors = self.select_facecolors_by_clusters(
            facecolors=None,
            cmap=cmap,
            with_noise=with_noise,
            differentiate_noise=differentiate_noise)
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
        self.subview_data_classification(
            fig=fig,
            ax=ax_top_right,
            dims=dims,
            facecolors=facecolors,
            with_clusters=with_clusters,
            with_noise=with_noise,
            differentiate_noise=differentiate_noise,
            with_legend=False)
        divided_btm_left = make_axes_locatable(ax_btm_left)
        cax_btm_left = divided_btm_left.append_axes(
            'right',
            size='5%',
            pad=0.05)
        self.subview_distance_matrix(
            fig=fig,
            ax=ax_btm_left,
            cmap=cmap,
            with_divergent_colors=True,
            with_colorbar=True,
            with_text_labels=with_text_labels,
            show_upper_triangle=show_upper_triangle,
            show_lower_triangle=show_lower_triangle,
            cax=cax_btm_left)
        divided_btm_right = make_axes_locatable(ax_btm_right)
        cax_btm_right = divided_btm_right.append_axes(
            'right',
            size='5%',
            pad=0.05)
        self.subview_membership_matrix(
            fig=fig,
            ax=ax_btm_right,
            cmap=cmap,
            with_colorbar=True,
            with_text_labels=with_text_labels,
            differentiate_noise=differentiate_noise,
            show_upper_triangle=show_upper_triangle,
            show_lower_triangle=show_lower_triangle,
            cax=cax_btm_right)
        handles, labels = [], []
        for ax in (ax_top_left, ax_top_right):
            _handles, _labels = ax.get_legend_handles_labels()
            handles.extend(_handles)
            labels.extend(_labels)
        fig.subplots_adjust(hspace=0.35, wspace=0.3)
        self.subview_legend(
            fig=fig,
            ax=ax_top_left,
            handles=handles,
            labels=labels,
            title='Density Method ({} Distance)'.format(self.distance_metric.title()))
        ## save or show figure
        if save:
            savename = 'Density{}_FullOptimizationResult_k{}_by_m{}_dims-{}'.format(
                self.optimization_method.upper(),
                self.k,
                self.m,
                '_'.join(dims.astype(str)))
            savename = savename.replace(' ', '_')
        else:
            savename = None
        self.display_image(fig, savename)

class DensityClustering(DensityClusteringVisualizer):

    def __init__(self, data, data_labels=None, distance_metric='euclidean', distance_matrix=None, savedir=None):
        super().__init__(data, data_labels, distance_metric, distance_matrix, savedir)

    def __repr__(self):
        return 'DensityClustering(%r, %r, %r, %r)' % (self.data, self.data_labels, self.distance_metric, self.savedir)

    def __str__(self):
        s = "\nDensity Clustering Optimizer\n"
        s += "\n .. optimization method:\n\t{}\n".format(
            self.optimization_method)
        s += "\n .. data:\n\t{} coordinates in {} dimensions\n".format(
            self.n,
            self.ndim)
        s += "\n .. cluster threshold (neighborhood eps):\n\t{}\n".format(
            self.cluster_threshold)
        s += "\n .. maximum radius:\n\t{}\n".format(
            self.maximum_radius)
        if self.minimum_radius > 0:
            s += "\n .. maximum radius:\n\t{}\n".format(
                self.maximum_radius)
        s += "\n .. number of clusters:\n\t{}\n".format(
            self.k)
        s += "\n .. number of noise:\n\t{}\n".format(
            self.m)
        s += "\n .. success:\n\t{}\n".format(
            self.success)
        s += "\n .. time elapsed:\n\t{}\n".format(
            self.duration)
        if self.flags is None:
            s += "\n .. flags:\n\t{}\n".format(
                self.flags)
        else:
            s += "\n .. flags:"
            for _, warning_message in self.flags.items():
                s += "\n\t{}".format(warning_message)
            s += "\n"
        return s

    def initialize_optimization_method(self, optimization_method):
        if optimization_method in ('dbscan', ):
            self._optimization_method = optimization_method
        else:
            raise ValueError("invalid optimization_method: {}".format(optimization_method))

    def initialize_dbscan_criteria(self, cluster_threshold, maximum_radius, minimum_radius=0):
        if not isinstance(cluster_threshold, int):
            raise ValueError("invalid type(cluster_threshold): {}".format(type(cluster_threshold)))
        if cluster_threshold < 1:
            raise ValueError("invalid cluster_threshold: {}".format(cluster_threshold))
        for key, arg in zip(('minimum_radius', 'maximum_radius'), (minimum_radius, maximum_radius)):
            if not isinstance(arg, (int, float)):
                raise ValueError("invalid type({}): {}".format(key, type(arg)))
            if arg < 0:
                raise ValueError("invalid {}: {}".format(key, arg))
        if minimum_radius > maximum_radius:
            raise ValueError("minimum_radius cannot exceed maximum_radius")
        if maximum_radius == 0:
            raise ValueError("invalid maximum_radius: {}".format(maximum_radius))
        self._minimum_radius = minimum_radius
        self._maximum_radius = maximum_radius
        self._cluster_threshold = cluster_threshold

    def depth_first_search(self, i, rows, cols, group):
        if i not in self.visited:
            self._visited.add(i)
            group.add(i)
            row_indices = np.where(rows == i)[0]
            col_indices = np.where(cols == i)[0]
            for indices in (row_indices, col_indices):
                rs, cs = rows[indices], cols[indices]
                for j in np.unique(np.concatenate((rs, cs))):
                    group = self.depth_first_search(
                        i=j,
                        rows=rows,
                        cols=cols,
                        group=group)
        return group

    def predict_dbscan(self):
        self._mask_matrix = ((self.minimum_radius <= self.reducible_matrix) & (self.reducible_matrix <= self.maximum_radius))
        rows, cols = np.where(self.mask_matrix)
        number_of_clusters, number_of_noise = 0, 0
        for r, c in zip(rows, cols):
            group = self.depth_first_search(
                i=r,
                rows=rows,
                cols=cols,
                group=set())
            group = self.depth_first_search(
                i=c,
                rows=rows,
                cols=cols,
                group=group)
            group = np.array(list(group))
            if group.size > 0:
                if group.size >= self.cluster_threshold:
                    self._membership[group] = number_of_clusters + 1
                    self._clusters[number_of_clusters] = self.data[group, :]
                    number_of_clusters += 1
                else:
                    self._membership[group] = - (number_of_noise + 1)
                    self._noise[number_of_noise] = self.data[group, :]
                    number_of_noise += 1

        # # non_visited = np.where(self.membership == 0)[0]
        # non_visited = np.array(list(
        #     set(np.arange(self.n, dtype=int)) - self.visited))
        # if non_visited.size > 0:
        #     self._membership[non_visited] = - (number_of_noise + 1)
        #     self._noise[number_of_noise] = self.data[non_visited, :]
        #
        # print("\n .. NON-VISITED ({}):\n{}\n".format(non_visited.shape, non_visited))
        # print(Counter(self.membership))

    def update_membership_matrix(self):
        mat = np.zeros(self.distance_matrix.shape, dtype=int)
        for ki, cluster in self.clusters.items():
            kj = ki + 1
            loc = np.where(self.membership == kj)[0]
            # indices = np.array(list(itertools.combinations_with_replacement(loc, r=2)))
            indices = np.array(list(itertools.combinations(loc, r=2)))
            for i, j in zip(*indices.T):
                self._membership_matrix[i, j] = kj
                self._membership_matrix[j, i] = kj
        for ki, noise in self.noise.items():
            kj = -1 * (ki + 1)
            loc = np.where(self.membership == kj)[0]
            # indices = np.array(list(itertools.combinations_with_replacement(loc, r=2)))
            indices = np.array(list(itertools.combinations(loc, r=2)))
            for i, j in zip(*indices.T):
                self._membership_matrix[i, j] = kj
                self._membership_matrix[j, i] = kj
        # print(Counter(self.membership_matrix.ravel()))

    def update_flags(self, error_msg=None):
        flags = dict()
        i = 0
        if self.n >= 1e3:
            i += 1
            memory_flag = "({})\tOperations involving distance matrices scale as O(n^2) and will consume a lot of memory for large n".format(
                i)
            flags['memory'] = memory_flag
        if self.m > 0:
            p = 0
            for mi in range(self.m):
                p += self.noise[mi].shape[0] * 100 / self.n
            if p >= 10:
                i += 1
                noise_flag = "({})\t{:.3} percent of your data is noise".format(
                    i,
                    p)
                flags['noise'] = noise_flag
        if error_msg is not None:
            i += 1
            error_flag = "({})\t{}".format(
                i,
                error_msg)
            flags['error'] = error_flag
        if self.k == 0:
            i += 1
            cluster_flag = "({})\tZero clusters were found".format(
                i)
            flags['cluster'] = cluster_flag
        if len(list(flags.keys())) > 0:
            self._flags = flags

    def fit(self, optimization_method, cluster_threshold, maximum_radius, minimum_radius=0):
        t0 = perf_counter()
        self.initialize_optimization_method(optimization_method)
        if self.optimization_method == 'dbscan':
            self.initialize_dbscan_criteria(
                cluster_threshold=cluster_threshold,
                maximum_radius=maximum_radius,
                minimum_radius=minimum_radius)
            try:
                error_msg = None
                self.predict_dbscan()
                self._success = True
            except RecursionError as error:
                error_msg = error.args[0]
                ... ## use dendrogram to avoid recursion of distance matrix neighbors
        else:
            raise ValueError("not yet implemented")
        self.update_membership_matrix()
        self._k = len(list(self.clusters.keys()))
        self._m = len(list(self.noise.keys())) ## max of empty (possibility) throws error
        self.update_flags(error_msg)
        self._duration = perf_counter() - t0

    def save_optimization_results(self):
        if not isinstance(self.savedir, str):
            raise ValueError("savedir is not initialized")
        savename = "{}Density_{}_k{}_by_m{}".format(
            self.savedir,
            self.optimization_method.title(),
            self.k,
            self.m)
        savename = savename.replace(' ', '_')
        with open("{}.txt".format(savename), "w") as text_file:
            text_file.write("{}".format(str(self)))















##
