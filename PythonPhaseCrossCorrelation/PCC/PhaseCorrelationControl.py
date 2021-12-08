"""

@title: Optimized Python Phase Cross Correlation
@author: Matthew Tralka
@date: September 2021
@version: 1.0

"""

from datetime import datetime
from enum import Enum
from enum import auto
from pathlib import Path
import re
from typing import Any
from typing import Optional
from typing import Union
import warnings

import numpy as np

from .CPU import phase_cross_correlation as pcc_cpu


class PCCMethods(Enum):
    CPU = auto()
    GPU = auto()


class PhaseCorrelationControl:
    """
    Phase Correlation Control Clasas

    Attributes
    ----------
    `reference_img` : str | Path | np.ndarray
        Path to desired reference image OR np.ndarray
    `moving_img` : str | Path | np.ndarray
        Path to desired moving image OR np.ndarray
    `outfile_dir` : str | Path
        Path to desired output directory. Default parent directory
    `outfile_name` : str
        Name of outfile. Default to iso timestamp
    `upsample` : int
        Upsampling factor for sub-pixel correlation. Default 1
    `col_start` : int
        Upper left starting column - X0. Default `-1` - full
    `col_end` : int
        Lower right ending column - X1. Default `-1` - full
    `row_start` : int
        Upper right starting row - Y0. Default `-1` - full
    `row_end` : int
        Lower right ending row - Y1. Default `-1` - full
    `window_size` : int
        Window size for PCC. Default `64`
    `window_step` : int
        Window step for PCC. Default `6`
    `outfile_driver` str
        GDAL driver type. Default `GTiff`
    `no_data` : float
        NODATA value for GDAl. Default `-9999.0`
    `method` : str or PCCMethods
        Processing type - `CPU` or `GPU`. Default `CPU`
    `reference_band` : int
        Reference band to read from `reference_img`. Default 1
    `moving_band` : int
        Moving band to read from `moving_img`. Default 1
    `auto_save` : bool
        If true, saves results immediately to `outfile_dir`/`outfile_name`. False, does not
    `out_geo_transform` : tuple (valid geotransform)
        Must be given to `save()` results if  `reference_img` and `moving_img` are passed as np.ndarrays
    `out_projection_ref` : valid projection ref
        Must be given to `save()` results if  `reference_img` and `moving_img` are passed as np.ndarrays

    Methods
    -------
    `run()`
        Run phase cross correlation
    `save()`
        Save results from phase cross correlation. `out_geo_transform` and `out_projection_ref` must be
        manually given if `reference_img` and `moving_img` are given as np.ndarrays
    """

    def __init__(
        self,
        reference_img: Union[Path, str, np.ndarray],
        moving_img: Union[Path, str, np.ndarray],
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
        method: Union[str, PCCMethods] = "CPU",
        reference_band: int = 1,
        moving_band: int = 1,
        auto_save: bool = False,
        out_geo_transform: tuple = None,
        out_projection_ref: Any = None,
    ):

        ##
        # if NP arrays --> set
        # elif if given Path/str --> intake
        ##
        if isinstance(reference_img, np.ndarray) and isinstance(moving_img, np.ndarray):
            self.reference_arr = reference_img
            self.moving_arr = moving_img

            self.moving_path = None
            self.reference_path = None

            if not out_geo_transform or not out_projection_ref:
                warnings.warn(
                    "`out_geo_transform` and `out_projection_ref` not given - you are unable to `save()` results",
                    Warning,
                )

        elif isinstance(reference_img, (Path, str)) and isinstance(
            moving_img, (Path, str)
        ):
            self.reference_path: Path = self._valdiate_path(
                reference_img, check_exists=True, check_is_file=True
            )

            self.moving_path: Path = self._valdiate_path(
                moving_img, check_exists=True, check_is_file=True
            )

            self.reference_band = reference_band
            self.moving_band = moving_band

            self.reference_arr = self._read_array(self.reference_path, reference_band)
            self.moving_arr = self._read_array(self.moving_path, reference_band)
        else:
            raise ValueError(
                "`reference_img` and `moving_img` must be given as a Path/str or np.ndaray"
            )

        ##
        # If set, validate outfile_dir
        ##
        if outfile_dir is not None:
            self.outfile_dir = self._valdiate_path(
                outfile_dir, check_exists=True, check_is_file=False, check_is_dir=True
            )

        ##
        # Validate `method` as str or PCCMethods Enum
        ##
        for pcc_method in PCCMethods.__members__.values():
            if method == pcc_method:
                self.method = method
                break
            elif isinstance(method, str) and method.upper().strip() == pcc_method.name:
                self.method = PCCMethods[method.upper().strip()]
                break
            else:
                raise AttributeError(
                    f"{method} not recognized. Select from {','.join([item.name for item in PCCMethods.__members__.values()])}"
                )

        self.outfile_name: str = self._get_valid_filename(outfile_name)
        self.upsample: int = upsample
        self.col_start: int = col_start
        self.col_end: int = col_end
        self.row_start: int = row_start
        self.row_end: int = row_end
        self.window_size: int = window_size
        self.window_step: int = window_step
        self.outfile_driver: str = outfile_driver
        self.no_data: float = float(no_data)
        self.auto_save: bool = auto_save
        self.out_geo_transform: tuple = out_geo_transform
        self.out_projection_ref: Any = out_projection_ref
        self.total_shift = None

        if self.upsample > 1 and self.method == PCCMethods.CPU:
            warnings.warn(
                "CPU upsampling not implemented. Performance will be impacted", Warning
            )

        if self.method.value == PCCMethods.GPU:
            raise NotImplementedError("GPU not implemented. Use CPU")

        self.run()

    @staticmethod
    def _valdiate_path(
        path: Union[str, Path],
        check_exists: bool = False,
        check_is_file: bool = False,
        check_is_dir=False,
    ) -> Path:

        valid_path: Path = Path(path) if isinstance(path, str) else path

        if check_exists:
            if not valid_path.exists():
                raise FileExistsError(f"{path} must exist")

        if check_is_file:
            if not valid_path.is_file():
                raise FileNotFoundError(f"{path} must be a file")

        if check_is_dir:
            if not valid_path.is_dir():
                raise ValueError(f"{path} must be a directory")

        return valid_path

    def run(self, auto_save: Optional[bool] = None) -> None:
        """
        Time and run PhaseCrossCorrelation

        """
        auto_save = self.auto_save if auto_save is None else auto_save
        start = datetime.now()
        self._process_arrays()
        self._process_correlation()

        if auto_save:
            self.save()
        print(f"Complete in: {datetime.now() - start}")

    @staticmethod
    def _read_array(
        file: Union[str, Path], band: int = 1, dtype: str = "int16"
    ) -> np.ndarray:
        """
        Opens and extract array from `file` using `band` and `dtype`. Requires `gdal`

        """
        try:
            import gdal
        except ImportError:
            print("`gdal` must be installed to read arrays")

        filename: Path = file if isinstance(file, Path) else Path(file)

        if not filename.is_file():
            raise FileNotFoundError("`file` not found")

        file_ds = gdal.Open(str(filename))

        if file_ds is None:
            raise FileNotFoundError("GDAL failed to open `file`")

        file_arr: np.ndarray = np.array(
            file_ds.GetRasterBand(band).ReadAsArray()
        ).astype(dtype)

        file_ds = None

        return file_arr

    def _process_arrays(self):
        """
        Extracts target area from `reference_arr` and `moving_arr`

        """

        if self.reference_arr.shape != self.moving_arr.shape:
            raise ValueError("`reference_arr` and `moving_arr` must be the same shape")

        self.reference_arr = self.reference_arr[
            self.y0 : self.y1, self.x0 : self.x1
        ].astype("intc")

        self.moving_arr = self.moving_arr[self.y0 : self.y1, self.x0 : self.x1].astype(
            "intc"
        )

    def _process_correlation(self):
        if self.method == PCCMethods.CPU:
            total_shift = pcc_cpu(
                self.reference_arr,
                self.moving_arr,
                self.window_size,
                self.window_step,
                self.no_data,
                self.upsample,
            )
        elif self.method == PCCMethods.GPU:
            raise NotImplementedError("GPU not implemented")
        else:
            raise AttributeError("`method` must be `CPU` or `GPU`")

        total_shift = np.where(
            total_shift != self.no_data, 1000.0 * total_shift, self.no_data
        )
        total_shift = np.where(total_shift > 32000, 32000, total_shift)

        self.total_shift = total_shift

    def save(self):
        """
        Saves `total_shift` array to disk. Requires `gdal`

        """

        try:
            import gdal
            import gdalconst
        except ImportError:
            print("`gdal` must be installed to read arrays")

        out_driver = gdal.GetDriverByName(self.outfile_driver)
        out_ds = out_driver.Create(
            self.outfile_full_path,
            self.total_shift.shape[1],
            self.total_shift.shape[0],
            1,
            gdalconst.GDT_Int16,
        )

        geo_transform: tuple
        projection_ref: Any
        if self.reference_path is not None or self.moving_path is not None:
            reference_ds = gdal.Open(
                str(
                    self.reference_path
                    if self.reference_path is not None
                    else self.moving_path
                )
            )
            geo_transform = reference_ds.GetGeoTransform()
            projection_ref = reference_ds.GetProjectionRef()
            reference_ds = None
        elif self.out_geo_transform is None or self.out_projection_ref is None:
            raise ValueError(
                "`reference_path` or `moving_path` or (`out_geo_transform` and `out_projection_ref`) must exist to `save()` results"
            )
        else:
            geo_transform = self.out_geo_transform
            projection_ref = self.out_projection_ref

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
        out_ds.SetProjection(projection_ref)
        out_ds.GetRasterBand(1).WriteArray(self.total_shift.astype("int16"))
        out_ds.GetRasterBand(1).SetNoDataValue(self.no_data)

        out_ds = None

        print(np.mean(self.total_shift))

    @staticmethod
    def _get_valid_filename(name: str) -> str:
        """
        Returns sanitizied filename. Credit @ Django

        """
        washed = name.strip().replace(" ", "_")
        washed = re.sub(r"(?u)[^-\w.]", "", washed)
        if washed in {"", ".", ".."}:
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
