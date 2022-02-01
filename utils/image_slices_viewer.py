import numpy as np
import matplotlib.pyplot as plt


class IndexTracker(object):
    def __init__(self, ax, x, vmin=None, vmax=None):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.x = x
        rows, cols, self.slices = x.shape
        self.ind = 0

        self.vmin = np.min(self.x) if vmin is None else vmin
        self.vmax = np.max(self.x) if vmax is None else vmax

        self.im = ax.imshow(self.x[:, :, self.ind], vmin=self.vmin, vmax=self.vmax)
        self.update()

    def on_scroll(self, event):
        # print("%s %s" % (event.button, int(event.step)))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.x[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


def launch_3d_slice_viewer(data: tuple, vmin: int = None, vmax: int = None):
    """
    Interactive way of viewing slices
    :param data: Data to visualize
    :param vmin: minimum value for display purposes
    :param vmax: maximum value for display purposes
    :return: None
    """
    if len(data) is 1:
        x = data[0]
        fig, ax = plt.subplots(nrows=1, ncols=1, num=0)
        tracker = IndexTracker(ax, x, vmin, vmax)
        fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
        plt.show()
    elif len(data) is 2:
        x1 = data[0]
        fig1, ax1 = plt.subplots(nrows=1, ncols=1, num=0)
        tracker1 = IndexTracker(ax1, x1, vmin, vmax)
        fig1.canvas.mpl_connect('scroll_event', tracker1.on_scroll)

        x2 = data[1]
        fig2, ax2 = plt.subplots(nrows=1, ncols=1, num=1)
        tracker2 = IndexTracker(ax2, x2, vmin, vmax)
        fig2.canvas.mpl_connect('scroll_event', tracker2.on_scroll)
    else:
        raise ValueError('Data length can only be 1 or 2')

    plt.show()
