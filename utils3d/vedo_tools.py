# -*- coding: UTF-8 -*-
"""
==================================================
@path   :BasicKnowledge -> vedo_tools
@IDE    :PyCharm
@Author :NaMuu
@Email  :2458543125@qq.com
@Date   :2025/3/27 15:55
@Version: V0.1
@License: (C)Copyright 2021-2023 , UP3D
@Reference: 
@History:
- 2025/3/27 15:55:
==================================================  
"""
__author__ = 'zxx'

import numpy as np
import vedo


#################################
class matrix3d_by_vedo(vedo.Plotter):
    """
    Generate a rendering window with slicing planes for the input Volume.
    """

    def __init__(
            self,
            data,
            cmaps=("gist_ncar_r", "hot_r", "bone", "bone_r", "jet", "Spectral_r"),
            clamp=True,
            show_histo=True,
            show_icon=True,
            draggable=False,
            at=0,
            **kwargs,
    ):
        """
        Generate a rendering window with slicing planes for the input Volume.

        Arguments:
            cmaps : (list)
                list of color maps names to cycle when clicking button
            clamp : (bool)
                clamp scalar range to reduce the effect of tails in color mapping
            use_slider3d : (bool)
                show sliders attached along the axes
            show_histo : (bool)
                show histogram on bottom left
            show_icon : (bool)
                show a small 3D rendering icon of the volume
            draggable : (bool)
                make the 3D icon draggable
            at : (int)
                subwindow number to plot to
            **kwargs : (dict)
                keyword arguments to pass to Plotter.

        Examples:
            - [slicer1.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/slicer1.py)

            <img src="https://vedo.embl.es/images/volumetric/slicer1.jpg" width="500">
        """
        ################################
        super().__init__(**kwargs)
        self.at(at)
        ################################

        cx, cy, cz, ch = "dr", "dg", "db", (0.3, 0.3, 0.3)
        if np.sum(self.renderer.GetBackground()) < 1.5:
            cx, cy, cz = "lr", "lg", "lb"
            ch = (0.8, 0.8, 0.8)
        self.o_data = data
        volume = vedo.Volume(data)
        self.volume = volume

        box = volume.box().alpha(0.2)
        self.add(box)

        volume_axes_inset = vedo.addons.Axes(
            box,
            xtitle=" ",
            ytitle=" ",
            ztitle=" ",
            yzgrid=False,
            xlabel_size=0,
            ylabel_size=0,
            zlabel_size=0,
            tip_size=0.08,
            axes_linewidth=3,
            xline_color="dr",
            yline_color="dg",
            zline_color="db",
        )

        if show_icon:
            self.add_inset(
                volume,
                volume_axes_inset,
                pos=(0.9, 0.9),
                size=0.15,
                c="w",
                draggable=draggable,
            )

        # inits
        la, ld = 0.7, 0.3  # ambient, diffuse
        dims = volume.dimensions()
        data = volume.pointdata[0]
        rmin, rmax = volume.scalar_range()
        if clamp:
            hdata, edg = np.histogram(data, bins=50)
            logdata = np.log(hdata + 1)
            # mean  of the logscale plot
            meanlog = np.sum(np.multiply(edg[:-1], logdata)) / np.sum(logdata)
            rmax = min(rmax, meanlog + (meanlog - rmin) * 0.9)
            rmin = max(rmin, meanlog - (rmax - meanlog) * 0.9)
            # print("scalar range clamped to range: ("
            #       + precision(rmin, 3) + ", " + precision(rmax, 3) + ")")

        self.cmap_slicer = cmaps[0]

        self.current_i = None
        self.current_j = None
        self.current_k = int(dims[2] / 2)

        self.xslice = None
        self.yslice = None
        self.zslice = None

        self.zslice = volume.zslice(self.current_k).lighting("", la, ld, 0)
        self.zslice.name = "ZSlice"
        self.zslice.cmap(self.cmap_slicer, vmin=rmin, vmax=rmax)
        self.add(self.zslice)

        self.histogram = None
        data_reduced = data
        if show_histo:
            dims = self.volume.dimensions()
            if data.ndim == 1:
                self.histogram = vedo.pyplot.histogram(
                    data_reduced,
                    outline=True,
                    c=self.cmap_slicer,
                    bg=ch,
                    alpha=1,
                    axes=dict(text_scale=2),
                ).clone2d(pos=[-0.925, -0.88], size=0.5)
                self.add(self.histogram)

        #################

        def add_text():
            if self.current_i is not None and self.current_j is not None and self.current_k is not None:
                text = vedo.Text2D(
                    f"i={self.current_i} j={self.current_j} k={self.current_k} == {self.o_data[self.current_i, self.current_j, self.current_k]}")
                text.pos([0.5, 0.01], "bottom-left")
                text.name = "text"
                self.remove("text")
                self.add(text)

        def slider_function_x(widget, event):
            i = int(self.xslider.value)
            if i == self.current_i:
                return
            self.current_i = i
            self.xslice = volume.xslice(i).lighting("", la, ld, 0)
            self.xslice.cmap(self.cmap_slicer, vmin=rmin, vmax=rmax)
            self.xslice.name = "XSlice"
            add_text()
            m = vedo.pyplot.matrix(self.o_data[self.current_i, :, :], cmap=self.cmap_slicer, xtitle="X").clone2d(
                pos=[-0.325, -0.88], size=0.3)
            m.name = "mx"
            self.remove("mx")
            self.remove("XSlice")  # removes the old one
            if 0 < i < dims[0]:
                self.add(self.xslice, m)
            self.render()

        def slider_function_y(widget, event):
            j = int(self.yslider.value)
            if j == self.current_j:
                return
            self.current_j = j
            self.yslice = volume.yslice(j).lighting("", la, ld, 0)
            self.yslice.cmap(self.cmap_slicer, vmin=rmin, vmax=rmax)
            self.yslice.name = "YSlice"
            m = vedo.pyplot.matrix(self.o_data[:, self.current_j, :], cmap=self.cmap_slicer, xtitle="Y").clone2d(
                pos=[0, -0.88], size=0.3)
            m.name = "my"
            self.remove("my")
            add_text()
            self.remove("YSlice")
            if 0 < j < dims[1]:
                self.add(self.yslice, m)
            self.render()

        def slider_function_z(widget, event):
            k = int(self.zslider.value)
            if k == self.current_k:
                return
            self.current_k = k
            self.zslice = volume.zslice(k).lighting("", la, ld, 0)
            self.zslice.cmap(self.cmap_slicer, vmin=rmin, vmax=rmax)
            self.zslice.name = "ZSlice"
            add_text()
            m = vedo.pyplot.matrix(self.o_data[:, :, self.current_k], cmap=self.cmap_slicer, xtitle="Z").clone2d(
                pos=[0.325, -0.88], size=0.3)
            m.name = "mz"
            self.remove("mz")
            self.remove("ZSlice")
            if 0 < k < dims[2]:
                self.add(self.zslice, m)
            self.render()

        bs = box.bounds()
        self.xslider = self.add_slider3d(
            slider_function_x,
            pos1=(bs[0], bs[2], bs[4]),
            pos2=(bs[1], bs[2], bs[4]),
            xmin=0,
            xmax=dims[0],
            t=box.diagonal_size() / vedo.mag(box.xbounds()) * 0.6,
            c=cx,
            show_value=False,
            title="X"
        )
        self.yslider = self.add_slider3d(
            slider_function_y,
            pos1=(bs[1], bs[2], bs[4]),
            pos2=(bs[1], bs[3], bs[4]),
            xmin=0,
            xmax=dims[1],
            t=box.diagonal_size() / vedo.mag(box.ybounds()) * 0.6,
            c=cy,
            show_value=False,
            title="Y"
        )
        self.zslider = self.add_slider3d(
            slider_function_z,
            pos1=(bs[0], bs[2], bs[4]),
            pos2=(bs[0], bs[2], bs[5]),
            xmin=0,
            xmax=dims[2],
            value=int(dims[2] / 2),
            t=box.diagonal_size() / vedo.mag(box.zbounds()) * 0.6,
            c=cz,
            show_value=False,
            title="Z"
        )

        def button_func(obj, ename):
            bu.switch()
            self.cmap_slicer = bu.status()
            for m in self.objects:
                if "Slice" in m.name:
                    m.cmap(self.cmap_slicer, vmin=rmin, vmax=rmax)
            self.remove(self.histogram)
            if show_histo:
                self.histogram = vedo.pyplot.histogram(
                    data_reduced,
                    outline=True,
                    c=self.cmap_slicer,
                    bg=ch,
                    alpha=1,
                    axes=dict(text_scale=2),
                ).clone2d(pos=[-0.925, -0.88], size=0.5)
                self.add(self.histogram)
            self.render()

        if len(cmaps) > 1:
            bu = self.add_button(
                button_func,
                states=cmaps,
                c=["k9"] * len(cmaps),
                bc=["k1"] * len(cmaps),  # colors of states
                size=16,
                bold=True,
            )
            if bu:
                bu.pos([0.1, 0.01], "bottom-left")


def show_matrix_by_vedo(data: np.ndarray):
    """
    用vedo渲染矩阵

    Args:
        data (np.ndarray): 输入的2d/3d数组；
    """
    if data.ndim == 3:
        matrix3d_by_vedo(data).show().close()
    elif data.ndim == 2:
        vedo.show(vedo.pyplot.matrix(data)).close()
    else:
        print(f"只支持2维和3维，不支持{data.ndim}")




