"""

 @title: Optimized Sentinel-2 Coregistration using Phase Cross Correlation
 @author: Matthew Tralka
 @date: May 2021
 @version: 0.1

"""

from pathlib import Path
import sys

import gdal
import numpy as np
from typer.testing import CliRunner


sys.path.append("...")
from main import app


runner = CliRunner()

TEST_REFERENCE_FILE: str = "./IMG_DATA/T18SUJ_20180809T154901_B04.jp2"
TEST_MOVING_FILE: str = "./IMG_DATA/T18SUJ_20180809T154901_B02.jp2"
TEST_OUTFILE_DIR: str = "./IMG_DATA/"
TEST_OUTFILE_NAME: str = "TEST_FILE.tif"

TEST_CORRELATION_MEAN: dict = {1: 126.266904}  # upscaling factor : mean


def get_file_mean(path: Path) -> float:

    file: str = str(path)
    file_ds = gdal.Open(file)
    file_arr: np.ndarray = np.array(
        file_ds.GetRasterBand(1).ReadAsArray()
    ).astype("int16")
    file_ds = None

    return np.mean(file_arr)


def test_app():

    for upscaling, mean in TEST_CORRELATION_MEAN.items():

        # TODO add `upscaling` to invokation
        result = runner.invoke(
            app,
            [
                TEST_REFERENCE_FILE,
                TEST_MOVING_FILE,
                "--out-path",
                TEST_OUTFILE_DIR,
                "--out-name",
                TEST_OUTFILE_NAME
            ]
        )
        outfile = Path(TEST_OUTFILE_DIR) / TEST_OUTFILE_NAME

        assert result.exit_code == 0
        assert "Complete" in result.stdout
        assert outfile.exists()
        assert outfile.is_file()
        assert get_file_mean(outfile) == mean