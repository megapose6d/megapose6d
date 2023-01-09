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



# Third Party
import numpy as np


def xr_merge(ds1, ds2, on, how="left", dim1="dim_0", dim2="dim_0", fill_value=np.nan):
    if how != "left":
        raise NotImplementedError

    ds1 = ds1.copy()
    ds1 = ds1.reset_coords().set_coords(on)
    ds2 = ds2.reset_coords().set_coords(on)

    ds2 = ds2.rename({dim2: dim1})
    df1 = ds1.reset_coords()[on].to_dataframe()
    df2 = ds2.reset_coords()[on].to_dataframe()

    df1["idx1"] = np.arange(len(df1))
    df2["idx2"] = np.arange(len(df2))

    merge = df1.merge(df2, on=on, how=how)
    assert len(merge) == ds1.dims[dim1]

    idx1 = merge["idx1"].values
    idx2 = merge["idx2"].values
    mask = np.isfinite(idx2)
    idx1 = idx1[mask]
    idx2 = idx2[mask].astype(int)

    for k, data_var in ds2.data_vars.items():
        array = data_var.values
        if isinstance(fill_value, dict):
            fill = fill_value.get(k, float("nan"))
        else:
            fill = fill_value
        assert data_var.dims[0] == dim1
        shape = list(array.shape)
        shape[0] = len(merge)
        new_array = np.empty(shape, dtype=np.array(fill).dtype)
        new_array[:] = fill
        new_array[idx1] = array[idx2]
        ds1[k] = data_var.dims, new_array
    return ds1
