"""

 @title: Optimized Sentinel-2 Coregistration using Phase Cross Correlation
 @author: Matthew Tralka
 @date: May 2021
 @version: 0.1

"""

from datetime import datetime
from pathlib import Path
import re
from typing import Dict
from typing import Union

import gdal
import gdalconst
import numpy as np

from OPCC.OptimizedPhaseCrossCorrelation import phase_cross_correlation


class PhaseCorrelationControl:
    def __init__(
        self,
        reference_img: Union[Path, str],
        moving_img: Union[Path, str],
        outfile_dir: Union[Path, str] = Path(__file__).parent.absolute(),
        outfile_name: str = f"parallax_{datetime.now().isoformat(timespec='minutes')}",
        upsample: int = 1,
        col_start: int = -1,
        col_end: int = -1,
        row_start: int = -1,
        row_end: int = -1,
        window_size: int = 64,
        window_step: int = 6,
        outfile_driver: str = "GTiff",
        no_data: float = -9999.0,
    ):

        path_inputs: Dict[Union[Path, str], str] = {
            reference_img: "reference_path",
            moving_img: "moving_path",
            outfile_dir: "outfile_dir"
        }

        for path, name in path_inputs.items():

            if isinstance(path, str):
                path = Path(path)

            if not path.exists():
                raise OSError(f"{path} not found")

            if name.endswith("path"):
                if not path.is_file():
                    raise FileNotFoundError(f"{path} file not found")
            elif name.endswith("dir"):
                if not path.is_dir():
                    raise NotADirectoryError(f"{path} is not a directory")

            setattr(self, name, path)

        self.outfile_name: str = self._get_valid_filename(outfile_name)
        self.upsample: int = upsample
        self.col_start: int = col_start
        self.col_end: int = col_end
        self.row_start: int = row_start
        self.row_end: int = row_end
        self.window_size: int = window_size
        self.window_step: int = window_step
        self.outfile_driver: str = outfile_driver
        self.outfile_type = gdalconst.GDT_Int16
        self.no_data: float = float(no_data)
        self.total_shift = None

        self.run()

    def run(self) -> None:

        start = datetime.now()
        self._intake_files()
        self._process_correlation()
        self._save_results()
        print(f"Complete in: {datetime.now() - start}")

    def _intake_files(self):

        def _extract_array(
            file: Path, band: int = 1, d_type: str = "int16"
        ) -> np.ndarray:
            file = str(file)
            file_ds = gdal.Open(file)
            file_arr: np.ndarray = np.array(
                file_ds.GetRasterBand(band).ReadAsArray()
            ).astype(d_type)
            file_ds = None
            return file_arr

        reference_arr = _extract_array(self.reference_path)
        moving_arr = _extract_array(self.moving_path)

        assert (
            reference_arr.shape == moving_arr.shape
        ), "`reference` and `moving` must be the same shape"

        self.reference_arr, self.moving_arr = reference_arr, moving_arr

        self.reference_arr = reference_arr[
            self.y0: self.y1, self.x0: self.x1
        ].astype("intc")

        self.moving_arr = moving_arr[
            self.y0: self.y1, self.x0: self.x1
        ].astype("intc")

    def _process_correlation(self):

        total_shift = phase_cross_correlation(
            self.reference_arr,
            self.moving_arr,
            self.window_size,
            self.window_step,
            self.no_data,
            self.upsample,
        )

        total_shift = np.where(
            total_shift != self.no_data, 1000.0 * total_shift, self.no_data
        )
        total_shift = np.where(total_shift > 32000, 32000, total_shift)

        self.total_shift = total_shift

    def _save_results(self):

        out_driver = gdal.GetDriverByName(self.outfile_driver)
        out_ds = out_driver.Create(
            self.outfile_full_path,
            self.total_shift.shape[1],
            self.total_shift.shape[0],
            1,
            self.outfile_type,
        )

        reference_ds = gdal.Open(str(self.reference_path))
        geo_transform = reference_ds.GetGeoTransform()

        x_offset = geo_transform[0] + geo_transform[1] * self.x0
        y_offset = geo_transform[3] + geo_transform[5] * self.y0

        geo_transform_subset: tuple = tuple(
            [
                x_offset,
                geo_transform[1],
                geo_transform[2],
                y_offset,
                geo_transform[4],
                geo_transform[5],
            ]
        )

        out_ds.SetGeoTransform(geo_transform_subset)
        out_ds.SetProjection(reference_ds.GetProjectionRef())
        out_ds.GetRasterBand(1).WriteArray(self.total_shift.astype("int16"))
        out_ds.GetRasterBand(1).SetNoDataValue(self.no_data)

        reference_ds, out_ds = None, None

    @staticmethod
    def _get_valid_filename(name: str) -> str:
        """
        Returns sanitizied filename. Credit @ Django
        """
        washed = name.strip().replace(' ', '_')
        washed = re.sub(r'(?u)[^-\w.]', '', washed)
        if washed in {'', '.', '..'}:
            raise ValueError(f"Could not sanitize ouput name {name}")
        return washed

    @property
    def x0(self) -> int:
        return self.col_start if self.col_start != -1 else 0

    @property
    def x1(self) -> int:
        return self.col_end if self.col_end != -1 else self.reference_shape_row

    @property
    def y0(self) -> int:
        return self.row_start if self.row_start != -1 else 0

    @property
    def y1(self) -> int:
        return self.row_end if self.row_end != -1 else self.reference_shape_col

    @property
    def moving_shape(self) -> tuple:
        return self.moving_arr.shape

    @property
    def reference_shape(self) -> tuple:
        return self.reference_arr.shape

    @property
    def reference_shape_row(self) -> int:
        return self.reference_arr.shape[1]

    @property
    def reference_shape_col(self) -> int:
        return self.reference_arr.shape[0]

    @property
    def outfile_full_path(self) -> str:

        full_name: str = str(self.outfile_dir / self.outfile_name)

        if not full_name.endswith(".tif"):
            full_name = full_name + ".tif"

        return full_name
