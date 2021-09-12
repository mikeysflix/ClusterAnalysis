from coordinates_configuration import *

class DendrogramVisualizer(ClusterEvaluation):

    def __init__(self, data, data_labels=None, distance_metric='euclidean', distance_matrix=None, savedir=None):
        super().__init__(data, data_labels, distance_metric, savedir)
        self.initialize_distance_matrix(distance_matrix)
        if self.distance_matrix.shape[0] <= 2:
            raise ValueError("distance_matrix must contain at least two rows/columns")
        self._reducible_matrix = self.mask_diagonal_entries(self.distance_matrix)
        self._levels = []
        self._groups = None
        self._orientation_info = dict()
        self._flags = None
        self._duration = None

    @property
    def reducible_matrix(self):
        return self._reducible_matrix

    @property
    def levels(self):
        return self._levels

    @property
    def groups(self):
        return self._groups

    @property
    def orientation_info(self):
        return self._orientation_info

    @property
    def flags(self):
        return self._flags

    @property
    def duration(self):
        return self._duration

    def get_ordering(self, iv, vector_id):
        if iv > self.distance_matrix.shape[0]:
            raise ValueError("invalid iv: {}".format(iv))
        if vector_id == 'row':
            indices = np.argsort(self.distance_matrix[iv, :])
        elif vector_id == 'column':
            indices = np.argsort(self.distance_matrix[:, iv])
        else:
            raise ValueError("invalid vector_id: {}".format(vector_id))
        return indices

    def subview_dendrogram_by_row(self, ax, factor, unordered_indices, ordered_indices, sep, with_text_labels):
        dmap = dict()
        for i, j, label in zip(unordered_indices, ordered_indices, self.data_labels):
            submap = {
                'x' : j * factor,
                'y' : 0}
            if label is not None:
                submap['label'] = label
            else:
                submap['label'] = None
            dmap[i] = submap
        f_levels = factor * self.levels
        for (i, j), level in zip(self.groups, f_levels):
            if label is not None:
                dmap[i]['label'] = '{}{}{}'.format(
                    dmap[i]['label'],
                    sep,
                    dmap[j]['label'])
            xi, yi = dmap[i]['x'], dmap[i]['y']
            xj, yj = dmap[j]['x'], dmap[j]['y']
            xp, yp = [xi, xi, xj, xj], [yi, level, level, yj]
            xc = np.nanmean([xi, xj])
            ax.plot(
                xp,
                yp,
                # color=...,
                )
            ax.scatter(
                xc,
                level,
                marker='.',
                s=2,
                # color=...,
                )
            if with_text_labels:
                ax.text(
                    xc,
                    level,
                    dmap[i]['label'],
                    # color=dmap[i]['facecolor'],
                    fontsize=self.textsize,
                    ha='center',
                    va='bottom')
            dmap[i]['x'] = xc
            dmap[i]['y'] = level
        ax.set_ylabel(
            'Level',
            fontsize=self.labelsize)
        yticks = np.array(ax.get_yticks())
        yticklabels = np.abs(yticks)
        ax.set_yticks(yticks)
        ax.set_yticklabels(
            yticklabels,
            fontsize=self.ticksize)
        ax.yaxis.set_minor_locator(
            ticker.AutoMinorLocator())
        if factor < 0:
            ax.xaxis.set_label_position('top')
            ax.xaxis.tick_top()
            ax.set_xticks(unordered_indices * -1)
            ax.set_xticklabels(
                ordered_indices,
                fontsize=self.ticksize)
        else:
            ax.set_xticks(unordered_indices)
            ax.set_xticklabels(
                ordered_indices,
                fontsize=self.ticksize)
        ax.set_xlabel(
            'Data',
            fontsize=self.labelsize)
        ax.tick_params(
            axis='both',
            labelsize=self.ticksize)
        ax.grid(
            color='k',
            linestyle=':',
            alpha=0.3)
        ax.set_title(
            'Dendrogram via {}'.format(
                self.optimization_method.title(),
            fontsize=self.titlesize))
        ylims = [0, np.max(self.levels) * factor * 1.125]
        ax.set_ylim([np.min(ylims), np.max(ylims)])
        return ax

    def subview_dendrogram_by_column(self, ax, factor, unordered_indices, ordered_indices, sep, with_text_labels):
        dmap = dict()
        for i, j, label in zip(unordered_indices, ordered_indices, self.data_labels):
            submap = {
                'x' : 0,
                'y' : j * factor}
            if label is not None:
                submap['label'] = label
            else:
                submap['label'] = None
            dmap[i] = submap
        f_levels = factor * self.levels
        for (i, j), level in zip(self.groups, f_levels):
            if label is not None:
                dmap[i]['label'] = '{}{}{}'.format(
                    dmap[i]['label'],
                    sep,
                    dmap[j]['label'])
            xi, yi = dmap[i]['x'], dmap[i]['y']
            xj, yj = dmap[j]['x'], dmap[j]['y']
            xp, yp = [xi, level, level, xj], [yi, yi, yj, yj]
            yc = np.nanmean([yi, yj])
            ax.plot(
                xp,
                yp,
                # color=...,
                )
            ax.scatter(
                level,
                yc,
                marker='.',
                s=2,
                # color=...,
                )
            if with_text_labels:
                ax.text(
                    level,
                    yc,
                    dmap[i]['label'],
                    # color=dmap[i]['facecolor'],
                    fontsize=self.textsize,
                    ha='center',
                    va='bottom')
            dmap[i]['x'] = level
            dmap[i]['y'] = yc
        ax.set_xlabel(
            'Level',
            fontsize=self.labelsize)
        xticks = np.array(ax.get_xticks())
        xticklabels = np.abs(xticks)
        ax.set_xticks(xticks)
        ax.set_xticklabels(
            xticklabels,
            fontsize=self.ticksize)
        ax.xaxis.set_minor_locator(
            ticker.AutoMinorLocator())
        if factor < 0:
            ax.yaxis.set_label_position('right')
            ax.yaxis.tick_right()
            ax.set_yticks(unordered_indices * -1)
            ax.set_yticklabels(
                ordered_indices,
                fontsize=self.ticksize)
        else:
            ax.set_yticks(unordered_indices)
            ax.set_yticklabels(
                ordered_indices,
                fontsize=self.ticksize)
        ax.set_ylabel(
            'Data',
            fontsize=self.labelsize)
        ax.tick_params(
            axis='both',
            labelsize=self.ticksize)
        ax.grid(
            color='k',
            linestyle=':',
            alpha=0.3)
        ax.set_title(
            'Dendrogram via {}'.format(
                self.optimization_method.title(),
            fontsize=self.titlesize))
        xlims = [0, np.max(self.levels) * factor * 1.125]
        ax.set_xlim([np.min(xlims), np.max(xlims)])
        return ax

    def subview_dendrogram(self, fig, ax, iv, vector_id, facecolors, with_text_labels=False, negative_direction=False, sep='/'):
        ordered_indices = self.get_ordering(
            iv=iv,
            vector_id=vector_id)
        factor = 1 if not negative_direction else -1
        unordered_indices = np.arange(self.n, dtype=int)
        self._orientation_info['iv'] = iv
        self._orientation_info['vector_id'] = vector_id
        self._orientation_info['factor'] = factor
        if vector_id == 'row':
            ax = self.subview_dendrogram_by_row(
                ax=ax,
                factor=factor,
                unordered_indices=unordered_indices,
                ordered_indices=ordered_indices,
                sep=sep,
                with_text_labels=with_text_labels,
                # facecolors=facecolors
                )
        else: # elif vector_id == 'column':
            ax = self.subview_dendrogram_by_column(
                ax=ax,
                factor=factor,
                unordered_indices=unordered_indices,
                ordered_indices=ordered_indices,
                sep=sep,
                with_text_labels=with_text_labels,
                # facecolors=facecolors
                )
        return ax

    def view_dendrogram(self, iv=0, vector_id='row', facecolors=None, cmap=None, with_text_labels=False, negative_direction=False, sep='/', save=False, **kwargs):
        fig, ax = plt.subplots(**kwargs)
        facecolors = ...
        if not isinstance(sep, str):
            raise ValueError("invalid type(sep): {}".format(type(sep)))
        self.subview_dendrogram(
            fig=fig,
            ax=ax,
            iv=iv,
            vector_id=vector_id,
            facecolors=facecolors,
            with_text_labels=with_text_labels,
            negative_direction=negative_direction,
            sep=sep)
        if save:
            savename = 'AgglomerativeDendrogram_{}_{}{}'.format(
                self.optimization_method.title(),
                self.orientation_info['vector_id'].title(),
                self.orientation_info['iv'])
            if negative_direction:
                savename += '_neg'
            else:
                savename += '_pos'
            savename = savename.replace(' ', '_')
        else:
            savename = None
        self.display_image(fig, savename)

    def view_dendrogram_and_distance_matrix(self, show_top=False, show_bottom=False, show_left=False, show_right=False, facecolors=None, cmap=None, with_text_labels=False, sep='/', save=False, **kwargs):
        show_args = np.array([show_top, show_bottom, show_left, show_right])
        nshow = np.sum(show_args)
        if nshow < 1:
            raise ValueError("at least one of the following inputs should be True: show_top, show_bottom, show_left, show_right")
        mat = self.select_from_matrix(
            mat=self.mask_diagonal_entries(self.distance_matrix),
            show_lower_triangle=True,
            show_upper_triangle=True)
        fig, axes = plt.subplots(nrows=3, ncols=3, **kwargs)
        (ax_top_left, ax_top_middle, ax_top_right, ax_middle_left, ax_middle_middle, ax_middle_right, ax_btm_left, ax_btm_middle, ax_btm_right) = axes.ravel()
        for ax in (ax_top_left, ax_btm_left, ax_top_right, ax_btm_right):
            ax.axis('off')
            ax.remove()
        norm = Normalize(vmin=0, vmax=np.nanmax(mat))
        if show_top:
            ax_top_middle = self.subview_dendrogram(
                fig=fig,
                ax=ax_top_middle,
                iv=0,
                vector_id='row',
                facecolors=facecolors,
                with_text_labels=with_text_labels,
                negative_direction=False,
                sep='/')
            ax_top_middle.set_xlabel('', fontsize=self.labelsize)
            ax_top_middle.set_title('', fontsize=self.labelsize)
        else:
            ax_top_middle.axis('off')
            ax_top_middle.remove()
        if show_bottom:
            ax_btm_middle = self.subview_dendrogram(
                fig=fig,
                ax=ax_btm_middle,
                iv=-1,
                vector_id='row',
                facecolors=facecolors,
                with_text_labels=with_text_labels,
                negative_direction=True,
                sep='/')
            ax_btm_middle.set_xlabel('', fontsize=self.labelsize)
            ax_btm_middle.set_title('', fontsize=self.labelsize)
        else:
            ax_btm_middle.axis('off')
            ax_btm_middle.remove()
        if show_left:
            ax_middle_left = self.subview_dendrogram(
                fig=fig,
                ax=ax_middle_left,
                iv=0,
                vector_id='column',
                facecolors=facecolors,
                with_text_labels=with_text_labels,
                negative_direction=True,
                sep='/')
            ax_middle_left.set_ylabel('', fontsize=self.labelsize)
            ax_middle_left.set_title('', fontsize=self.labelsize)
        else:
            ax_middle_left.axis('off')
            ax_middle_left.remove()
        if show_right:
            ax_middle_right = self.subview_dendrogram(
                fig=fig,
                ax=ax_middle_right,
                iv=-1,
                vector_id='column',
                facecolors=facecolors,
                with_text_labels=with_text_labels,
                negative_direction=False,
                sep='/')
            ax_middle_right.set_ylabel('', fontsize=self.labelsize)
            ax_middle_right.set_title('', fontsize=self.labelsize)
        else:
            ax_middle_right.axis('off')
            ax_middle_right.remove()
        ax_middle_middle, im = self.subview_matrix(
            fig=fig,
            ax=ax_middle_middle,
            mat=mat,
            cmap=cmap,
            norm=norm,
            with_text_labels=with_text_labels,
            aspect='auto')
        ax_middle_middle.set_title('', fontsize=self.labelsize)
        ax_middle_middle.xaxis.set_minor_locator(
            ticker.AutoMinorLocator())
        ax_middle_middle.yaxis.set_minor_locator(
            ticker.AutoMinorLocator())
        ax_middle_middle.tick_params(axis='both', labelsize=self.ticksize)
        ax_middle_middle.grid(
            color='gray',
            linestyle=':',
            alpha=0.3)
        fig.subplots_adjust(hspace=0.3)
        if save:
            savename = 'AgglomerativeDendrogramDistanceMatrix_{}'.format(
                self.optimization_method.title())
            for key, v in zip(('top', 'bottom', 'left', 'right'), (show_top, show_bottom, show_left, show_right)):
                if v:
                    savename += '_{}'.format(key)
            savename = savename.replace(' ', '_')
        else:
            savename = None
        self.display_image(fig, savename)


class Dendrogram(DendrogramVisualizer):

    def __init__(self, data, data_labels=None, distance_metric='euclidean', distance_matrix=None, savedir=None):
        super().__init__(data, data_labels, distance_metric, distance_matrix, savedir)
        self._f_statistic = None

    def __repr__(self):
        return 'AgglomerativeClustering(%r, %r)' % (self.data, self.savedir)

    def __str__(self):
        s = "\nAgglomerative Clustering (Dendrogram) Optimizer\n"
        s += "\n .. optimization method:\n\t{}\n".format(
            self.optimization_method)
        s += "\n .. data:\n\t{} coordinates in {} dimensions\n".format(
            self.n,
            self.ndim)
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

    @property
    def f_statistic(self):
        return self._f_statistic

    def initialize_optimization_method(self, optimization_method):
        fmap = dict(zip(
            ('single-linkage', 'complete-linkage'),
            (np.nanmin, np.nanmax)))
        if optimization_method in ('single-linkage', 'complete-linkage'):
            self._optimization_method = optimization_method
        else:
            raise ValueError("invalid optimization_method: {}".format(optimization_method))
        if optimization_method in list(fmap.keys()):
            self._optimization_method = optimization_method
            self._f_statistic = fmap[optimization_method]
        else:
            raise ValueError("invalid optimization_method: {}".format(optimization_method))

    def iteratively_reduce_distance_matrix(self):
        niter = self.distance_matrix.shape[0] - 1
        ordering = np.arange(self.reducible_matrix.shape[0], dtype=int)
        groups = []
        for i in range(niter):
            level = self.f_statistic(self.reducible_matrix)
            self._levels.append(level)
            rows, cols = np.where(self.reducible_matrix == level)
            rcs = [rows[0], cols[0]]
            i, j = np.nanmin(rcs), np.nanmax(rcs)
            groups.append(
                (ordering[i], ordering[j]))
            prev_two_vectors = np.array([self.reducible_matrix[i, :], self.reducible_matrix[j, :]])
            curr_vector = self.f_statistic(prev_two_vectors, axis=0)
            self._reducible_matrix[i, :] = curr_vector
            self._reducible_matrix[:, i] = curr_vector
            self._reducible_matrix = np.delete(
                self._reducible_matrix,
                j,
                axis=0)
            self._reducible_matrix = np.delete(
                self._reducible_matrix,
                j,
                axis=1)
            ordering = np.delete(ordering, j)
            self._reducible_matrix = self.mask_diagonal_entries(self.reducible_matrix)
        self._levels = np.array(self._levels)
        self._groups = np.array(groups)

    def fit(self, optimization_method):
        t0 = perf_counter()
        self.initialize_optimization_method(
            optimization_method=optimization_method)
        self.iteratively_reduce_distance_matrix()
        self._duration = perf_counter() - t0

    def save_optimization_results(self):
        if not isinstance(self.savedir, str):
            raise ValueError("savedir is not initialized")
        savename = "{}AgglomerativeDendrogram_{}".format(
            self.savedir,
            self.optimization_method.title())
        savename = savename.replace(' ', '_')
        with open("{}.txt".format(savename), "w") as text_file:
            text_file.write("{}".format(str(self)))





##
