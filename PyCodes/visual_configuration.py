import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize, LogNorm, TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.legend_handler import HandlerRegularPolyCollection

class ScatterHandler(HandlerRegularPolyCollection):

    """
    https://stackoverflow.com/questions/49297599/why-doesnt-the-color-of-the-points-in-a-scatter-plot-match-the-color-of-the-poi
    """

    def update_prop(self, legend_handle, orig_handle, legend):
        legend._set_artist_props(legend_handle)
        legend_handle.set_clip_box(None)
        legend_handle.set_clip_path(None)

    def create_collection(self, orig_handle, sizes, offsets, transOffset):
        p = type(orig_handle)([orig_handle.get_paths()[0]],
                              sizes=sizes, offsets=offsets,
                              transOffset=transOffset,
                              cmap=orig_handle.get_cmap(),
                              norm=orig_handle.norm )

        a = orig_handle.get_array()
        if type(a) != type(None):
            p.set_array(np.linspace(a.min(),a.max(),len(offsets)))
        else:
            self._update_prop(p, orig_handle)
        return p

class VisualConfiguration():

    def __init__(self, ticksize=7, labelsize=8, textsize=5, titlesize=9, headersize=10, bias='left'):
        """

        """
        super().__init__()
        self._savedir = None
        self.ticksize = ticksize
        self.labelsize = labelsize
        self.textsize = textsize
        self.titlesize = titlesize
        self.headersize = headersize
        self.empty_label = '  '

    @property
    def savedir(self):
        return self._savedir

    @staticmethod
    def select_facecolors(counts, cmap=None, default_color='darkorange'):
        """

        """
        if cmap is None:
            return [default_color]*counts.size
        elif isinstance(cmap, (tuple, list, np.ndarray)):
            nc = len(cmap)
            if nc != counts.size:
                raise ValueError("{} colors for {} bins".format(nc, counts.size))
            return list(cmap)
        else:
            norm = Normalize(vmin=np.min(counts), vmax=np.max(counts))
            f = plt.get_cmap(cmap)
            return f(norm(counts))

    @staticmethod
    def get_number_of_legend_columns(labels):
        """

        """
        if isinstance(labels, int):
            n = labels
        else:
            n = len(labels)
        if n > 2:
            if n % 3 == 0:
                ncol = 3
            else:
                ncol = n // 2
        else:
            ncol = n
        return ncol

    @staticmethod
    def get_empty_handle(ax):
        """

        """
        empty_handle = ax.scatter([np.nan], [np.nan], color='none', alpha=0)
        return empty_handle

    def update_save_directory(self, savedir):
        self._savedir = savedir

    def update_legend_design(self, leg, title=None, textcolor=None, facecolor=None, edgecolor=None, borderaxespad=None):
        """

        """
        if title:
            leg.set_title(
                title,
                prop={
                    'size': self.labelsize,
                    # 'weight' : 'semibold'
                    })
            if textcolor:
                leg.get_title().set_color(textcolor)
            # leg.get_title().set_ha("center")
        leg._legend_box.align = "center"
        frame = leg.get_frame()
        if facecolor:
            frame.set_facecolor(facecolor)
        if edgecolor:
            frame.set_edgecolor(edgecolor)
        if textcolor:
            for text in leg.get_texts():
                text.set_color(textcolor)
        return leg

    def subview_legend(self, fig, ax, handles, labels, title=''):
        """

        """
        if len(labels) == 1:
            ncol = 3
            empty_handle = self.get_empty_handle(ax)
            handles = [empty_handle, handles[0], empty_handle]
            labels = [self.empty_label, labels[0], self.empty_label]
        else:
            ncol = self.get_number_of_legend_columns(labels)
        fig.subplots_adjust(bottom=0.2)
        leg = fig.legend(handles=handles, labels=labels, ncol=ncol, loc='lower center', mode='expand', borderaxespad=0.1, fontsize=self.labelsize)
        leg = self.update_legend_design(leg, title=title, textcolor='darkorange', facecolor='k', edgecolor='steelblue')

    def display_image(self, fig, savename=None, dpi=800, bbox_inches='tight', pad_inches=0.1, extension='.png', **kwargs):
        """

        """
        if savename is None:
            plt.show()
        elif isinstance(savename, str):
            if self.savedir is None:
                raise ValueError("cannot save plot; self.savedir is None")
            savepath = '{}{}{}'.format(self.savedir, savename, extension)
            fig.savefig(savepath, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches, **kwargs)
        else:
            raise ValueError("invalid type(savename): {}".format(type(savename)))
        plt.close(fig)


##
