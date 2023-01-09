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
import datetime
import textwrap
from collections import OrderedDict
from itertools import cycle
from pathlib import Path

# Third Party
import numpy as np
import pandas as pd
import seaborn as sns
from bokeh.io import output_notebook, show
from bokeh.layouts import gridplot
from bokeh.models import HoverTool
from bokeh.plotting import figure
from IPython.display import display
from omegaconf import OmegaConf

# MegaPose
from megapose.training.pose_models_cfg import (
    check_update_config as check_update_config_pose,
)


class Plotter:
    def __init__(
        self,
        log_dir: Path
    ):
        self.fill_config_fn = check_update_config_pose
        self.log_dir = Path(log_dir)
        assert self.log_dir.exists()
        output_notebook(verbose=False, hide_banner=True)

    @property
    def colors_hex(self):
        return cycle(sns.color_palette().as_hex())

    @property
    def colors_uint8(self):
        return cycle(sns.color_palette())

    def load_logs(self, run_ids):
        configs = OrderedDict()
        log_dicts = OrderedDict()
        eval_dicts = OrderedDict()
        colors = OrderedDict()
        for run_id, color in zip(run_ids, self.colors_hex):
            run_dir = self.log_dir / run_id
            assert run_dir.exists(), f"{run_id} does not exists."
            cfg_path = run_dir / "config.yaml"
            try:
                config = OmegaConf.load(cfg_path)
            except:
                config = yaml.load(cfg_path.read_text(), Loader=yaml.UnsafeLoader)
                config = vars(config)
            configs[run_id] = self.fill_config_fn(config)

            log_path = run_dir / "log.txt"
            if log_path.exists():
                log_df = pd.read_json(run_dir / "log.txt", lines=True)
                last_write = datetime.datetime.fromtimestamp((run_dir / "log.txt").stat().st_mtime)
            else:
                log_df = None
                last_write = datetime.datetime.now()
            configs[run_id]["delta_t"] = (
                f"{(datetime.datetime.now() - last_write).seconds / 60:.1f}" + f"({len(log_df)})"
            )
            log_dicts[run_id] = log_df

            ds_eval = dict()
            for f in run_dir.iterdir():
                if "errors_" in f.name:
                    ds = f.with_suffix("").name.split("errors_")[1]
                    ds_eval[ds] = pd.read_json(f, lines=True)
                    ds_eval[ds] = ds_eval[ds].groupby("epoch").last().reset_index()
            eval_dicts[run_id] = ds_eval

            colors[run_id] = color

        self.colors = colors
        self.log_dicts = log_dicts
        self.eval_dicts = eval_dicts
        self.configs = configs
        self.run_ids = run_ids
        self.figures = [[]]

    def _add_figure(self, f, new_row):
        if new_row:
            row = []
            self.figures.append(row)
        else:
            row = self.figures[-1]
        row.append(f)

    def plot_eval_fields(
        self,
        fields,
        dataset="auto",
        new_row=False,
        semilogy=False,
        legend=False,
        title=None,
        y_range=None,
        dash_patterns=("solid", "dashed", "dotted"),
    ):
        y_axis_type = "auto" if not semilogy else "log"
        f = figure(
            y_axis_type=y_axis_type,
            background_fill_color="#EAEAF2",
            background_fill_alpha=0.6,
            y_range=y_range,
        )
        if dataset == "auto":
            datasets = []
            for ds_eval in self.eval_dicts.values():
                datasets += list(ds_eval.keys())
            if len(datasets) == 0:
                dataset = None
            else:
                dataset = datasets[0]

        for field, dash_pattern in zip(fields, dash_patterns):
            for run_id in self.run_ids:
                color = self.colors[run_id]
                if dataset is None or dataset not in self.eval_dicts[run_id]:
                    continue
                eval_df = self.eval_dicts[run_id][dataset]
                if field in eval_df:
                    x = eval_df["epoch"].values
                    y = eval_df[field].values
                    run_num = run_id.split("-")[-1]
                    name = f"{run_num}/{field}"
                    if len(x) == 1:
                        f.circle(x, y, color=color, line_dash=dash_pattern, name=name)
                        x = np.concatenate(([100], x))
                        y = np.concatenate((y, y))
                    f.line(
                        x,
                        y,
                        line_width=1.0,
                        color=color,
                        line_dash=dash_pattern,
                        legend_label=str(run_num),
                        name=f"{run_num}/{field}",
                    )

        if title is not None:
            f.title.text = title

        if legend:
            f.legend.location = "top_right"
            f.legend.click_policy = "hide"
            f.legend.label_text_font_size = "6pt"
        else:
            f.legend.visible = False

        tool = HoverTool(
            tooltips=[("x,y", "@x, @y"), ("name", "$name")],
            line_policy="nearest",
            point_policy="snap_to_data",
        )
        f.add_tools(tool)
        self._add_figure(f, new_row=new_row)
        return f

    def plot_eval_field(
        self,
        field,
        datasets="auto",
        new_row=False,
        semilogy=False,
        legend=False,
        title=None,
        y_range=None,
        dash_patterns=("solid", "dashed", "dotted"),
    ):
        y_axis_type = "auto" if not semilogy else "log"
        f = figure(
            y_axis_type=y_axis_type,
            background_fill_color="#EAEAF2",
            background_fill_alpha=0.6,
            y_range=y_range,
        )
        assert datasets == "auto" or isinstance(datasets, list)
        if datasets == "auto":
            datasets = []
            for ds_eval in self.eval_dicts.values():
                datasets += list(ds_eval.keys())
            datasets = set(datasets)

        for dataset, dash_pattern in zip(datasets, dash_patterns):
            for run_id in self.run_ids:
                color = self.colors[run_id]
                eval_df = self.eval_dicts[run_id]
                if dataset in eval_df:
                    df = eval_df[dataset]
                    if field in eval_df[dataset]:
                        x = df["epoch"].values
                        y = df[field].values
                        run_num = run_id.split("-")[-1]
                        name = f"{run_num}/{dataset}"
                        name = "\n ".join(textwrap.wrap(name, width=20))
                        if len(x) == 1:
                            f.circle(x, y, color=color, line_dash=dash_pattern, name=name)
                            x = np.concatenate(([0], x))
                            y = np.concatenate((y, y))
                        f.line(
                            x,
                            y,
                            line_width=1.0,
                            color=color,
                            line_dash=dash_pattern,
                            legend_label=str(run_num),
                            name=name,
                        )

        if title is not None:
            f.title.text = title

        if legend:
            f.legend.location = "top_right"
            f.legend.click_policy = "hide"
            f.legend.label_text_font_size = "6pt"
        else:
            f.legend.visible = False

        tool = HoverTool(
            tooltips=[("x,y", "@x, @y"), ("name", "$name")],
            line_policy="nearest",
            point_policy="snap_to_data",
        )
        f.add_tools(tool)
        self._add_figure(f, new_row=new_row)
        return f

    def plot_train_fields(
        self,
        fields,
        new_row=False,
        semilogy=False,
        y_range=None,
        legend=False,
        title=None,
        dash_patterns=("solid", "dashed", "dotted", "dotdash"),
    ):
        y_axis_type = "auto" if not semilogy else "log"
        f = figure(
            y_axis_type=y_axis_type,
            background_fill_color="#EAEAF2",
            background_fill_alpha=0.6,
            y_range=y_range,
        )
        for field, dash_pattern in zip(fields, dash_patterns):
            for run_id in self.run_ids:
                color = self.colors[run_id]
                log_df = self.log_dicts[run_id]
                if field in log_df:
                    x = log_df["epoch"]
                    y = log_df[field]
                    m = np.logical_not(np.isnan(y))
                    x, y = x[m], y[m]
                    run_num = run_id.split("-")[-1]
                    f.line(
                        x,
                        y,
                        line_width=1.0,
                        color=color,
                        line_dash=dash_pattern,
                        legend_label=str(run_num),
                        name=f"{run_num}/{field}",
                    )
        if title is not None:
            f.title.text = title

        if legend:
            f.legend.location = "top_right"
            f.legend.click_policy = "hide"
            f.legend.label_text_font_size = "6pt"
        else:
            f.legend.visible = False

        tool = HoverTool(
            tooltips=[("x,y", "@x, @y"), ("name", "$name")],
            line_policy="nearest",
            point_policy="snap_to_data",
        )
        f.add_tools(tool)
        self._add_figure(f, new_row=new_row)
        return f

    def show(self):
        layout = gridplot(self.figures, sizing_mode="scale_width")
        show(layout)

    def show_configs(self, ignore=None, diff=True):
        if ignore is None:
            ignore = (
                "n_workers",
                "save_dir",
                "job_dir",
                "seed",
                "val_ds_names",
                "run_id",
                "label_to_category_id",
                "categories",
                "train_datasets",
                "hardware",
                "run_comment",
            )
        ignore = list(ignore)

        df = {k: dict(v) for k, v in self.configs.items()}
        df = pd.DataFrame(df).T
        config_df = df.copy()
        self.config_df = config_df

        name2color = {k: v for k, v in zip(self.run_ids, self.colors_uint8)}

        def f_row(data):
            rgb = (np.array(name2color[data.name]) * 255).astype(np.uint8)
            return [
                f"background-color: rgba({rgb[0]},{rgb[1]},{rgb[2]},1.0)" for _ in range(len(data))
            ]

        if "possible_roots" in df.columns:
            df["possible_roots"] = df["possible_roots"].apply(str)
        if "train_ds_names" in df.columns:
            df["train_ds_names"] = df["train_ds_names"].apply(str)
        if diff:
            for ignore_n in ignore:
                if ignore_n in df:
                    df.drop(ignore_n, axis=1, inplace=True)
            for k in df.columns:
                if isinstance(df[k][0], list):
                    df[k] = df[k].map(lambda x: tuple(x) if isinstance(x, list) else x)
            diff_df = df.loc[:, df.nunique() > 1]
            display_df = diff_df
        else:
            display_df = config_df

        display_df = display_df.style.apply(f_row, axis=1)
        display(display_df)
        return display_df
