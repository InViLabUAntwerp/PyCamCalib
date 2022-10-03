"""Module that contains all custom widgets for the GUI."""

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtWidgets import QWidget, QVBoxLayout, QMainWindow
import numpy as np
from typing import Tuple
import numpy.typing as npt


class DistortionPlotWidget(QWidget):
    """Widget for the camera distortion plot."""

    def __init__(self, parent) -> None:
        """Class constructor."""
        super(DistortionPlotWidget, self).__init__(parent)
        self.canvas = FigureCanvasQTAgg(Figure())
        toolbar = CustomToolbar(self.canvas, self)
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(toolbar)
        vertical_layout.addWidget(self.canvas)
        self.setLayout(vertical_layout)
        self.axes = self.canvas.figure.add_subplot(111)
        self.axes.set_title('Radial distortion model')
        self.axes.set_xlabel('Horizontal')
        self.axes.set_ylabel('Vertical')
        self.canvas.show()

    def plot_distortion(self, sensor_size: Tuple[int, int], m: npt.NDArray[np.float64],
                        d: npt.NDArray[np.float64]) -> None:
        """Plot camera distortion in the plot widget."""
        width = sensor_size[0]
        height = sensor_size[1]
        n_steps = 20
        [u, v] = np.meshgrid(np.linspace(0, width - 1, n_steps), np.linspace(0, height - 1, n_steps))
        xyz = np.linalg.solve(m, np.vstack((np.ravel(u, order='F'), np.ravel(v, order='F'), np.ones(u.size))))
        xp = xyz[0, :] / xyz[2, :]
        yp = xyz[1, :] / xyz[2, :]
        r2 = xp ** 2 + yp ** 2
        r4 = r2 ** 2
        r6 = r2 ** 3
        coef = 1 + np.dot(d[0], r2) + np.dot(d[1], r4) + np.dot(d[4], r6)
        xpp = xp * coef + 2 * np.dot(d[2], (xp * yp)) + np.dot(d[3], (r2 + 2 * xp ** 2))
        ypp = yp * coef + np.dot(d[2], (r2 + 2 * yp ** 2)) + 2 * np.dot(d[3], (xp * yp))
        u2 = m[0, 0] * xpp + m[0, 2]
        v2 = m[1, 1] * ypp + m[1, 2]
        du = u2 - np.ravel(u, order='F')
        dv = v2 - np.ravel(v, order='F')
        dr = np.reshape(np.hypot(du, dv), u.shape, order='F')

        # plot
        self.axes.cla()
        self.axes.quiver(np.ravel(u, order='F') + 1, np.ravel(v, order='F') + 1, du, dv, color='b')
        self.axes.plot(width / 2, height / 2, 'x', label='Sensor center')
        self.axes.plot(m[0, 2], m[1, 2], 'o', label='Principal point')
        contour_set = self.axes.contour(u[0, :] + 1, v[:, 0] + 1, dr, colors='k')
        self.axes.clabel(contour_set, inline=1, fontsize=10)
        self.axes.set_xlim(1, width)
        self.axes.set_ylim(1, height)
        self.axes.set_aspect('equal')
        self.axes.set_title('Radial distortion model')
        self.axes.set_xlabel('Horizontal')
        self.axes.set_ylabel('Vertical')
        self.canvas.draw()

    def reset_plot(self) -> None:
        """Reset the plot."""
        self.axes.cla()
        self.axes.set_title('Radial distortion model')
        self.axes.set_xlabel('Horizontal')
        self.axes.set_ylabel('Vertical')
        self.canvas.draw()


class ReprojErrPlotWidget(QWidget):
    """Widget for re-projection error plot."""
    def __init__(self, parent: QMainWindow):
        super(ReprojErrPlotWidget, self).__init__(parent)
        self.bars = []
        self.bar_status = []
        self.image_indices = []
        self.canvas = FigureCanvasQTAgg(Figure())
        toolbar = CustomToolbar(self.canvas, self)
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(toolbar)
        vertical_layout.addWidget(self.canvas)
        self.setLayout(vertical_layout)
        self.axes = self.canvas.figure.add_subplot(111)
        self.axes.set_xlabel("Image index")
        self.axes.set_ylabel("Reprojection error")
        self.axes.set_title("Reprojection error for each detected image")
        self.canvas.show()

    def bar_clicked(self, event) -> None:
        """Select or de-select error bar."""
        if event.inaxes == self.axes:
            for idx in range(len(self.bars)):
                bar = self.bars[idx]
                cont, _ = bar.contains(event)
                if cont:
                    if self.bar_status[idx]:
                        bar.set(color='r')
                        self.bar_status[idx] = 0
                    else:
                        bar.set(color='b')
                        self.bar_status[idx] = 1
                    self.canvas.draw()
                    break

    def plot_reproj_error(self, per_view_err: npt.NDArray[np.float64], image_indices: npt.NDArray[np.float64],
                          rms_reproj_error: float) -> None:
        """Plot the re-projection errors."""
        self.axes.cla()
        self.bar_status = np.ones(len(image_indices))
        self.image_indices = image_indices
        self.bars = self.axes.bar(list(map(str, image_indices)), per_view_err, color='b')
        line = self.axes.axhline(rms_reproj_error, color='g', linestyle='--')
        self.axes.set_xlabel("Image index")
        self.axes.set_ylabel("Reprojection error")
        self.axes.set_title("Reprojection error for each detected image")
        self.axes.legend([self.bars, line], ['Image', 'RMS'])
        self.canvas.mpl_connect('button_press_event', self.bar_clicked)
        self.canvas.draw()

    def reset_plot(self) -> None:
        """Reset the plot."""
        self.axes.cla()
        self.axes.set_xlabel("Image index")
        self.axes.set_ylabel("Reprojection error")
        self.axes.set_title("Reprojection error for each detected image")
        self.canvas.draw()

    def get_good_indices(self) -> list:
        """Get the indices of all selected bars."""
        return [self.image_indices[i]-1 for i in range(len(self.image_indices)) if self.bar_status[i]]


class CustomToolbar(NavigationToolbar2QT):
    """"Custom navigation toolbar that only contains 'home', 'pan' and 'zoom' buttons."""

    toolitems = [t for t in NavigationToolbar2QT.toolitems if t[0] in ('Home', 'Pan', 'Zoom', 'Save')]

