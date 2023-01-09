"""
Copyright (c) 2022 Inria & NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


# Standard Library

# Standard Library
from pathlib import Path
from typing import Optional, Tuple

# Third Party
import bokeh
import bokeh.io
import bokeh.models.sources
import bokeh.plotting
import numpy as np
import PIL
from PIL import Image

# MegaPose
from megapose.utils.types import Resolution


def save_image_figure(f: bokeh.plotting.figure, im_path: Path) -> PIL.Image:
    f.toolbar.logo = None
    f.toolbar_location = None
    f.title = None
    f.sizing_mode = "fixed"
    im = bokeh.io.export.get_screenshot_as_png(f)
    w, h = im.size
    im = im.convert("RGB").crop((1, 1, w, h)).resize((w, h))
    im.save(im_path)
    return im


def to_rgba(im: np.ndarray) -> np.ndarray:
    """Converts (h, w, 3) to (h, w, 4) data for bokeh.
    im must have values in (0, 255)
    NOTE: Maybe this could be simplified only using Pillow ?
    """
    out_im = np.empty((im.shape[0], im.shape[1]), dtype=np.uint32)
    view = out_im.view(dtype=np.uint8).reshape((im.shape[0], im.shape[1], 4))
    pil_im = Image.fromarray(im)
    im = np.asarray(pil_im.convert("RGBA"))
    im = np.flipud(im)
    view[:, :, :] = im
    return out_im


def plot_image(
    im: np.ndarray,
    tools: str = "",
    im_size: Optional[Resolution] = None,
    figure: Optional[bokeh.plotting.figure] = None,
) -> Tuple[bokeh.plotting.figure, bokeh.models.sources.ColumnDataSource]:
    if np.asarray(im).ndim == 2:
        gray = True
    else:
        im = to_rgba(im)
        gray = False

    if im_size is None:
        h, w = im.shape[:2]
    else:
        h, w = im_size
    source = bokeh.models.sources.ColumnDataSource(dict(rgba=[im]))
    f = image_figure("rgba", source, im_size=(h, w), gray=gray, figure=figure)
    return f, source


def make_image_figure(
    im_size: Resolution = (240, 320),
) -> bokeh.plotting.figure:
    h, w = im_size
    f = bokeh.plotting.figure(
        x_range=(0, w - 1),
        y_range=(0, h - 1),
        width=w,
        height=h,
        title="",
        tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],
    )
    f.toolbar.logo = None
    f.toolbar_location = None
    f.axis.visible = False
    f.grid.visible = False
    f.min_border = 0
    f.outline_line_width = 0
    f.outline_line_color = None
    f.background_fill_color = None
    f.border_fill_color = None
    return f


def image_figure(
    key: str,
    source: bokeh.models.sources.ColumnDataSource,
    im_size: Resolution = (240, 320),
    gray: bool = False,
    figure: Optional[bokeh.plotting.figure] = None,
) -> bokeh.plotting.figure:
    # NOTE: Remove this function ?
    h, w = im_size
    if figure is None:
        f = make_image_figure(im_size=im_size)
    else:
        f = figure

    if gray:
        f.image(key, x=0, y=0, dw=w, dh=h, source=source)
    else:
        f.image_rgba(key, x=0, y=0, dw=w, dh=h, source=source)
    return f
