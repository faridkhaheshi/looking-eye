from matplotlib import pyplot as plt


class DynamicPlot:
    def __init__(self, subplots=(1, 1), axis_on=False, **figargs):
        fig, _ = plt.subplots(subplots[0], subplots[1], **figargs)
        self.subplots = subplots
        self.figure = fig
        self.axes = self.figure.axes
        if not axis_on:
            for ax in self.axes:
                ax.axis('off')
        # plt.ion()
        # plt.show()

    def update_image(self, image, subplot=(1, 1), pause_ms=None, title=None):
        row, col = subplot
        cols = self.subplots[1]
        ax_index = (row - 1) * cols + (col - 1)
        if len(self.axes[ax_index].images) == 0:
            self.axes[ax_index].imshow(image)
        else:
            self.axes[ax_index].images[0].set_data(image)
        if pause_ms is not None:
            plt.pause(pause_ms / 1000.0)
        if title is not None:
            self.axes[ax_index].set_title(title)

    def release(self):
        plt.ioff()
