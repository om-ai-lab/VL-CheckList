import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import matplotlib
import random
import os
import json

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)

                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def generate_chart(output_folder,corpus_path,task,chart_type,models):
    """
    output_folder : str
        The path of folder that saves all evaluation results.
    corpus_path : str
        The path of json file that records corresponding file names of all metrics
    task : {'itm' | 'itc'}
        Task type of output result.
    chart_type : str
        Type of chart.
    models : list
        List of model name.

    """    
    m = json.load(open(corpus_path))
    arrs = []
    colors = ["cyan", "magenta", "crimson", "orange","blue"]
    for model in models:
        filepath = os.path.join(output_folder,model,task)
        score_list = []

        for item in m.keys():
            data_num = len(m[item].keys())
            data_score = []
            for data in m[item].keys():
                score = 0
                file_num = len(m[item][data])
                for file in m[item][data]:
                    json_name = os.path.join(filepath,f"{file}.json")
                    if not os.path.exists(json_name):
                        print(f"{file} has not been evaluated.")
                        return
                    else:
                        m1 = json.load(open(json_name))
                        score += m1["total_acc"]
                data_score.append(score/file_num)
            score_list.append(sum(data_score)/data_num)
        arrs.append(score_list)

    fig_ = plt.figure(figsize=(25, 8))
    models_name = [models]
    for i in range(len(models_name)):
        data = [['O-Large', 'O-Medium', 'O-Small', 'O-Center', 'O-Mid', 'O-Margin', 'A-Color', 'A-Material', 'A-Size',
                "A-State", "A-Action", "R-action", "R-spatial"],
                ('', arrs)]
        N = len(data[0])
        theta = radar_factory(N, frame='polygon')

        spoke_labels = data.pop(0)
        title, case_data = data[0]
        ax = fig_.add_subplot(f'13{i+1}', projection='radar')

        ax.set_rgrids([30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90])
        lines = []
        labels = models_name[i]
        for i, d in enumerate(case_data):
            c = colors.pop()
            line = ax.plot(theta, d, label=labels[i], color=c)
            lines.append(line)
            ax.fill(theta, d, alpha=0.25, color=c)
        ax.set_varlabels(spoke_labels)
        plt.legend(loc="best", bbox_to_anchor=(1, 1))

    plt.show()

if __name__ == "__main__":

    generate_chart('./output','corpus.json','itc',chart_type='radar',models=['vilt','albef'])