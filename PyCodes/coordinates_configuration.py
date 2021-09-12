import numpy as np
from time import perf_counter
from visual_configuration import *

class DataSetConfiguration(VisualConfiguration):

    def __init__(self, data, data_labels, savedir):
        super().__init__()
        self._data = None
        self.initialize_data(data)
        self.n = self.data.shape[0]
        self.ndim = self.data.shape[1]
        self.data_labels = self.autocorrect_data_labels(data_labels)
        self.update_save_directory(savedir)
        self._membership = np.full(self.data.shape[0], -1, dtype=int)

    @property
    def data(self):
        return self._data

    @property
    def membership(self):
        return self._membership

    @staticmethod
    def get_vector_dimension_label(dim):
        return r'$\vec{X_{%d}}$' % dim

    def initialize_data(self, data):
        if isinstance(data, (tuple, list, np.ndarray)):
            if not isinstance(data, np.ndarray):
                data = np.array(data)
        else:
            raise ValueError("invalid type(data): {}".format(type(data)))
        s = len(data.shape)
        if s not in (1, 2):
            raise ValueError("invalid data.shape: {}".format(data.shape))
        if s == 1:
            x = np.zeros(data.size)
            self._data = np.array([data, x]).T
        else:
            self._data = data

    def autocorrect_data_labels(self, data_labels):
        if data_labels is None:
            data_labels = [None for _ in range(self.n)]
        elif isinstance(data_labels, (tuple, list, np.ndarray)):
            nc = len(data_labels)
            if nc != self.n:
                raise ValueError("{} coordinates with {} labels".format(self.n, nc))
        else:
            raise ValueError("invalid type(data_labels): {}".format(type(data_labels)))
        return data_labels

    def select_viewing_dimensions(self, dims):
        if dims is None:
            dims = np.arange(self.ndim, dtype=int)
        elif isinstance(dims, int):
            dims = np.array([dims], dtype=int)
        elif isinstance(dims, (tuple, list, np.ndarray)):
            if not isinstance(dims, np.ndarray):
                dims = np.array(dims)
            dims = dims.astype(int)
        else:
            raise ValueError("invalid type(dims): {}".format(type(dims)))
        if dims.size > 3: # dim-0, dim-1, dim-2
            raise ValueError("number of dimensions cannot exceed 3")
        if dims.size < 1:
            raise ValueError("at least one dimension is needed")
        if np.max(dims) > self.ndim:
            raise ValueError("invalid dimension: {}".format(np.max(dims)))
        return dims

    def subview_unclassified_data(self, fig, ax, dims, with_legend=False):
        args = [self.data[:, dim] for dim in dims]
        if dims.size == 1:
            args.append(
                np.zeros(self.n, dtype=int))
        ax.scatter(
            *args,
            label='Unclassified Data',
            marker='.',
            s=2,
            facecolor='black',
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
            'Unclassified Data',
            fontsize=self.titlesize)
        if with_legend:
            handles, labels = ax.get_legend_handles_labels()
            self.subview_legend(
                fig=fig,
                ax=ax,
                handles=handles,
                labels=labels,
                title='$N = {:,}$'.format(self.n))
        return ax

    def view_unclassified_data(self, dims=None, save=False, **kwargs):
        dims = self.select_viewing_dimensions(dims)
        if dims.size == 3:
            fig = plt.figure(**kwargs)
            ax = fig.add_subplot(1, 1, 1, projection='3d')
        else:
            fig, ax = plt.subplots(**kwargs)
        self.subview_unclassified_data(
            fig=fig,
            ax=ax,
            dims=dims,
            with_legend=True)
        ## save or show figure
        if save:
            savename = 'Unclassified_Data_dims-{}'.format('_'.join(dims.astype(str)))
            # savename = savename.replace(' ', '_')
        else:
            savename = None
        self.display_image(fig, savename)

class DistanceConfiguration(DataSetConfiguration):

    def __init__(self, data, data_labels, distance_metric, savedir):
        super().__init__(data, data_labels, savedir)
        self._f_distance = None
        self._distance_metric = None
        self._distance_matrix = None
        self.distance_map = {
            'manhattan' : self.get_manhattan_distance,
            'squared euclidean' : self.get_squared_euclidean_distance,
            'euclidean' : self.get_euclidean_distance}
        self.initialize_distance_metric(
            distance_metric=distance_metric)

    @property
    def f_distance(self):
        return self._f_distance

    @property
    def distance_metric(self):
        return self._distance_metric

    @property
    def distance_matrix(self):
        return self._distance_matrix

    @staticmethod
    def mask_diagonal_entries(matrix, k=0):
        loc = np.eye(*matrix.shape, k=k)
        return np.ma.masked_where(loc, matrix)

    @staticmethod
    def get_displacement_matrix(data):
        shape = (data.shape[0], 1, data.shape[1])
        displacement = data - data.reshape(shape)
        return displacement

    @staticmethod
    def get_manhattan_distance(displacement, axis):
        return np.sum(np.abs(displacement), axis=axis)

    @staticmethod
    def get_squared_euclidean_distance(displacement, axis):
        return np.sum(np.square(displacement), axis=axis)

    def get_euclidean_distance(self, displacement, axis):
        sq_distance = self.get_squared_euclidean_distance(displacement, axis)
        return np.sqrt(sq_distance)

    def initialize_distance_metric(self, distance_metric):
        if distance_metric is not None:
            if distance_metric not in list(self.distance_map.keys()):
                raise ValueError("invalid distance_metric: {}".format(distance_metric))
            self._f_distance = self.distance_map[distance_metric]
            self._distance_metric = distance_metric

    def get_distance_matrix(self, distance_matrix=None):
        if distance_matrix is None:
            displacement_matrix = self.get_displacement_matrix(
                self.data)
            distance_matrix = self.f_distance(
                displacement_matrix,
                axis=-1)
        else:
            distance_matrix = np.copy(distance_matrix)
        return distance_matrix

    def initialize_distance_matrix(self, distance_matrix=None, override=False):
        if self.distance_matrix is None:
            self._distance_matrix = self.get_distance_matrix(distance_matrix)
        else:
            if override:
                if distance_matrix is None:
                    self._distance_matrix = self.get_distance_matrix(distance_matrix)

    def select_from_matrix(self, mat, show_upper_triangle=False, show_lower_triangle=False):
        if not any([show_upper_triangle, show_lower_triangle]):
            raise ValueError("show_upper_triangle and/or show_lower_triangle must be True")
        if not all([show_upper_triangle, show_lower_triangle]):
            for diagonal in range(1, mat.shape[0]):
                if not show_upper_triangle:
                    mat = self.mask_diagonal_entries(
                        matrix=mat,
                        k=diagonal)
                if not show_lower_triangle:
                    mat = self.mask_diagonal_entries(
                        matrix=mat,
                        k=-1 * diagonal)
        return mat

    def subview_matrix(self, fig, ax, mat, cmap, norm, with_text_labels=False, **kwargs):
        im = ax.imshow(
            mat,
            cmap=cmap,
            norm=norm,
            **kwargs)
        if with_text_labels:
            for r in range(mat.shape[0]):
                for c in range(mat.shape[1]):
                    if r != c:
                        x = mat[r, c]
                        facecolor = 'k' if norm(x) > 0.5 else 'white'
                        ax.text(
                            c,
                            r,
                            x,
                            color=facecolor,
                            ha='center',
                            va='center')
        return ax, im

class ClusterEvaluation(DistanceConfiguration):

    def __init__(self, data, data_labels, distance_metric, savedir):
        super().__init__(data, data_labels, distance_metric, savedir)
        self._optimization_method = None
        self._barycenters = None
        self._clusters = dict()
        self._silhouette = dict()
        self._membership_matrix = None

    @property
    def optimization_method(self):
        return self._optimization_method

    @property
    def barycenters(self):
        return self._barycenters

    @property
    def clusters(self):
        return self._clusters

    @property
    def silhouette(self):
        return self._silhouette

    @property
    def membership_matrix(self):
        return self._membership_matrix

    def get_intra_cluster_difference(self, cluster):
        displacement_matrix = self.get_displacement_matrix(cluster)
        distance_matrix = self.f_distance(displacement_matrix, axis=-1)
        # distance_matrix = self.mask_diagonal_entries(_distance_matrix, k=0)
        avg_distance_per_point = np.nanmean(distance_matrix, axis=1)
        return avg_distance_per_point

    def get_inter_cluster_difference(self, cluster, nearest_cluster):
        avg_distance_per_point = np.full(cluster.shape[0], np.nan)
        for i, point in enumerate(cluster):
            displacement = point - nearest_cluster
            distance = self.f_distance(displacement, axis=1)
            avg_distance_per_point[i] = np.nanmean(distance)
        return avg_distance_per_point

    def evaluate_silhouette(self):
        # self.initialize_distance_matrix(
        #     distance_matrix=None,
        #     override=False)
        displacement_matrix = self.get_displacement_matrix(
            data=self.barycenters)
        distance_matrix = self.f_distance(
            displacement_matrix,
            axis=-1)
        distance_matrix = self.mask_diagonal_entries(distance_matrix, k=0)
        nearest_loc = np.argmin(distance_matrix, axis=1)
        coefficients = []
        for ki, kj in enumerate(nearest_loc):
            cluster, nearest_cluster = self.clusters[ki], self.clusters[kj]
            avg_intra_distance_per_point = self.get_intra_cluster_difference(
                cluster=cluster)
            avg_inter_distance_per_point = self.get_inter_cluster_difference(
                cluster=cluster,
                nearest_cluster=nearest_cluster)
            numerator = (avg_inter_distance_per_point - avg_intra_distance_per_point)
            denominator = np.nanmax(
                np.array([avg_intra_distance_per_point, avg_inter_distance_per_point]),
                axis=0)
            s = numerator / denominator
            self._silhouette[ki] = s
            coefficients.append(
                np.nanmean(s))
        _coef = np.nanmax(coefficients)
        if _coef < -1 or _coef > 1:
            raise ValueError("invalid silhouette coefficient: {}".format(_coef))
        self._silhouette['coefficient'] = _coef

    def subview_silhouette_coefficient_line(self, ax):
        ax.axvline(
            x=self.silhouette['coefficient'],
            color='k',
            label='Silhouette Coefficient = ${:.4f}$'.format(self.silhouette['coefficient']))
        return ax

    def subview_silhouette(self, fig, ax, facecolors, with_legend=False, dy=1, show_coefficient=False):
        k = self.barycenters.shape[0]
        y = 0
        yticks = [y]
        for ki in range(k):
            # s = self.silhouette[ki]
            s = np.sort(
                self.silhouette[ki])
            ax.barh(
                y,
                s[0],
                height=dy,
                facecolor=facecolors[ki],
                align='center',
                label='Cluster ${}$'.format(ki + 1))
            y += dy
            for width in s[1:]:
                ax.barh(
                    y,
                    width,
                    height=dy,
                    facecolor=facecolors[ki],
                    align='center')
                y += dy
            y += dy
            yticks.append(y)
        if show_coefficient:
            ax = self.subview_silhouette_coefficient_line(ax)
        ax.set_xlabel(
            'Silhouette',
            fontsize=self.labelsize)
        ax.set_ylabel(
            'Clusters',
            fontsize=self.labelsize)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.xaxis.set_minor_locator(
            ticker.AutoMinorLocator())
        ax.set_yticks(yticks)
        ax.yaxis.set_minor_locator(
            ticker.AutoMinorLocator())
        ax.tick_params(
            axis='both',
            labelsize=self.ticksize)
        ax.grid(
            color='k',
            linestyle=':',
            alpha=0.3)
        ax.set_xlim([-1, 1])
        ax.set_ylim([yticks[0], yticks[-1]])
        ax.set_title(
            '{} Silhouette ($k={}$)'.format(
                self.optimization_method.title(),
                k),
            fontsize=self.titlesize)
        if with_legend:
            handles, labels = ax.get_legend_handles_labels()
            _s = "Silhouette Coefficient = ${0:.3f}$".format(self.silhouette['coefficient'])
            self.subview_legend(
                fig=fig,
                ax=ax,
                handles=handles,
                labels=labels,
                title='Barycentric Method ({} Distance) {}'.format(
                    self.distance_metric.title(),
                    _s))
        return ax




##
