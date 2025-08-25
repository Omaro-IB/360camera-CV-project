from math import ceil, floor
import numpy as np
import matplotlib.pyplot as plt


class BinTensor:
    def __init__(self, x_range, y_range, z_range, bin_size, labels=("X", "Y", "Z")):
        """
        :param x_range: tuple: (h_min, h_max)
        :param y_range: tuple: (s_min, s_max)
        :param z_range: tuple: (v_min, v_max)
        :param bin_size: int: size of bins
        """
        # Ranges of values to be added to bins
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

        self.bin_size = bin_size  # Size of bins
        self.labels = labels  # labels for visualization

        # initialize tensor of dimensions h_range / bin_size, s_range / bin_size, v_range / bin_size
        self.data = np.zeros((ceil((x_range[1] - x_range[0] + 1) / bin_size),
                              ceil((y_range[1] - y_range[0] + 1) / bin_size),
                              ceil((z_range[1] - z_range[0] + 1) / bin_size)))
        self.maximum = (-1, -1, -1)  # = index of largest point

    def get_bindex(self, x, y, z):
        """
        Get bin indices for point (x,y,z)
        :param x: int: x coordinate
        :param y: int: y coordinate
        :param z: int: z coordinate
        :return: int, int, int: x_bindex, y_bindex, z_bindex
        """
        x_bindex = floor((x - self.x_range[0]) / self.bin_size)
        y_bindex = floor((y - self.y_range[0]) / self.bin_size)
        z_bindex = floor((z - self.z_range[0]) / self.bin_size)

        # Verify x,y,z ranges
        if x_bindex < 0 or x_bindex >= self.data.shape[0] or y_bindex < 0 or y_bindex >= self.data.shape[1] or z_bindex < 0 or z_bindex >= self.data.shape[2]:
            return

        return x_bindex, y_bindex, z_bindex

    def get_approx_point(self, x_bindex, y_bindex, z_bindex):
        """
        Get approximate point (x,y,z) given bin indices (assuming point lies in middle of bin)
        :param x_bindex: int: bin index x
        :param y_bindex: int: bin index y
        :param z_bindex: int: bin index z
        :return: int, int, int: x, y, z
        """
        x = ((x_bindex + 0.5) * self.bin_size) + self.x_range[0]
        y = ((y_bindex + 0.5) * self.bin_size) + self.y_range[0]
        z = ((z_bindex + 0.5) * self.bin_size) + self.z_range[0]
        return x, y, z

    def add_point(self, x, y, z, suppress_warnings=False):
        """
        Add point (x,y,z) to appropriate bin index
        :param x: x coordinate
        :param y: y coordinate
        :param z: z coordinate
        :param suppress_warnings: bool: default=False: suppress warning messages if point is out of range
        """
        # Calculate bindexes
        bindices = self.get_bindex(x, y, z)

        if bindices:
            # add to bin and update maximum point
            self.data[*bindices] += 1
            if self.data[*bindices] > self.data[self.maximum]:
                self.maximum = bindices
        elif not suppress_warnings:
            print(f"WARNING: point {x, y, z} out of range")

    def visualize(self, title='3D Visualization of Tensor Values'):
        """
        Visualize a 3D scatter plot using matplotlib
        :param title: str (optional): title of 3D scatter plot
        """
        # Create coordinate grids for the 3D scatter plot
        x, y, z = np.indices(self.data.shape)

        # Flatten the grids and tensor values
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        values = self.data.flatten()

        # Create the 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        values[values == 0] = None  # don't display 0 values
        # Plot the points with color corresponding to the tensor value
        scatter = ax.scatter(x, z, y, c=values, cmap='viridis', s=50)

        # Add a color bar to indicate the value scale
        fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)

        # Labels and title
        ax.set_xlabel(self.labels[0] + " Bin")
        ax.set_zlabel(self.labels[1] + " Bin")
        ax.set_ylabel(self.labels[2] + " Bin")
        ax.set_title(title)

        plt.show()
