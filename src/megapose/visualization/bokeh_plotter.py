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
from hashlib import sha1
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

# Third Party
import bokeh
import numpy as np
import seaborn as sns
import torch
from bokeh.models import ColumnDataSource, LabelSet
from PIL import Image

# MegaPose
from megapose.utils.tensor_collection import PandasTensorCollection
from megapose.visualization.bokeh_utils import make_image_figure, to_rgba
from megapose.visualization.utils import get_mask_from_rgb, image_to_np_uint8


class BokehPlotter:
    def __init__(
        self,
        dump_image_dir: Optional[Path] = None,
        read_image_dir: Optional[Path] = None,
        is_notebook: bool = True,
    ):
        """Used to plot images.

        Contains an internal state `source_map` holding pointers to image data.
        This can be useful for updating images in real-time without re-creating figures.
        """
        self.source_map: Dict[str, bokeh.models.sources.ColumnDataSource] = dict()
        self.dump_image_dir = dump_image_dir
        self.read_image_dir = read_image_dir
        if is_notebook:
            bokeh.io.output_notebook(hide_banner=True)

    @property
    def hex_colors(self) -> Iterator[str]:
        return cycle(sns.color_palette(n_colors=40).as_hex())

    @property
    def colors(self) -> Iterator[Tuple[float, float, float]]:
        return cycle(sns.color_palette(n_colors=40))

    def get_source(self, name: str) -> Tuple[bokeh.models.sources.ColumnDataSource, bool]:
        if name in self.source_map:
            source = self.source_map[name]
            new = False
        else:
            source = ColumnDataSource()
            self.source_map[name] = source
            new = True
        return source, new

    def plot_image(
        self,
        im: Union[torch.Tensor, np.ndarray],
        figure: Optional[bokeh.plotting.figure] = None,
        name: str = "image",
    ) -> bokeh.plotting.figure:

        im_np = image_to_np_uint8(im)

        h, w, _ = im_np.shape

        if figure is None:
            figure = make_image_figure(im_size=(h, w))
        assert figure is not None
        source, new = self.get_source(f"{figure.id}/{name}")

        if self.dump_image_dir is not None:
            if new:
                figure.image_url("url", x=0, y=0, w=w, h=h, source=source, anchor="bottom_left")
            im_np.flags.writeable = False
            im_hash = sha1(im_np.copy().data).hexdigest()
            im_path = str(self.dump_image_dir / f"{im_hash}.jpg")
            Image.fromarray(im_np).save(im_path)
            im_url = str(self.read_image_dir) + str(Path(im_path).name)
            print(im_url)
            source.data = dict(url=[im_url])
        else:
            if new:
                figure.image_rgba("image", x=0, y=0, dw=w, dh=h, source=source)
            source.data = dict(image=[to_rgba(im_np)])
        return figure

    def plot_overlay(
        self,
        rgb_input: np.ndarray,
        rgb_rendered: np.ndarray,
        figure: Optional[bokeh.plotting.figure] = None,
    ) -> bokeh.plotting.figure:
        """Overlays observed and rendered images.

        A mask is computed using the values <15 px of rgb_rendered.
        All images are np.uint8 with values in (0, 255)

        Args:
            rgb_input: (h, w, 3)
            rgb_rendered: (h, w, 3) with values <15 px as background.
            figure: Optional figure in which the data should be plotted.
        """
        assert rgb_input.dtype == np.uint8 and rgb_rendered.dtype == np.uint8
        mask = get_mask_from_rgb(rgb_rendered)

        rgb_overlay = np.zeros_like(rgb_input).astype(np.float32)
        rgb_overlay[~mask] = rgb_input[~mask] * 0.6 + 255 * 0.4
        rgb_overlay[mask] = rgb_rendered[mask] * 0.8 + 255 * 0.2
        rgb_overlay = rgb_overlay.astype(np.uint8)
        f = self.plot_image(rgb_overlay, figure=figure)
        return f

    def plot_detections(
        self,
        f: bokeh.plotting.figure,
        detections: PandasTensorCollection,
        colors: Union[str, List[str]] = "red",
        text: Optional[Union[str, List[str]]] = None,
        text_auto: bool = True,
        text_font_size: str = "8pt",
        line_width: int = 2,
        source_id: str = "",
    ) -> bokeh.plotting.figure:

        boxes = detections.bboxes.cpu().numpy()
        if text_auto:
            if "score" in detections.infos.columns:
                text = [f"{row.label} {row.score:.2f}" for _, row in detections.infos.iterrows()]
            else:
                text = [f"{row.label}" for _, row in detections.infos.iterrows()]

        xs = []
        ys = []
        patch_colors = []

        if text is not None:
            assert len(text) == len(boxes)
            text_x, text_y = [], []
        if isinstance(colors, (list, tuple, np.ndarray)):
            assert len(colors) == len(boxes)
        else:
            colors = [colors for _ in range(len(boxes))]

        # Convert boxes to bokeh coordinate system
        boxes = np.array(boxes)
        boxes[:, [1, 3]] = f.height - boxes[:, [1, 3]]
        for n, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            xs.append([x1, x2, x2, x1])
            ys.append([y1, y1, y2, y2])
            patch_colors.append(colors[n])
            if text is not None:
                text_x.append(x1)
                text_y.append(y1)
        source, new = self.get_source(f"{f.id}/{source_id}/bboxes")

        if new:
            f.patches(
                xs="xs",
                ys="ys",
                source=source,
                line_width=line_width,
                color="colors",
                fill_alpha=0.0,
            )

            if text is not None:
                labelset = LabelSet(
                    x="text_x",
                    y="text_y",
                    text="text",
                    text_align="left",
                    text_baseline="bottom",
                    text_color="white",
                    source=source,
                    background_fill_color="colors",
                    text_font_size=text_font_size,
                )
                f.add_layout(labelset)
        data = dict(xs=xs, ys=ys, colors=patch_colors)
        if text is not None:
            data.update(text_x=text_x, text_y=text_y, text=text)
        source.data = data
        return f
