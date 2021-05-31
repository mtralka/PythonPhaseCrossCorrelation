"""

 @title: Optimized Sentinel-2 Coregistration using Phase Cross Correlation
 @author: Matthew Tralka
 @date: May 2021
 @version: 0.1

"""
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from OPCC.PhaseCorrelationControl import PhaseCorrelationControl

app = typer.Typer()


def index_callback(value: int):
    if value < -1:
        raise typer.BadParameter("Index must be positive or -1 (full extent)")
    return value


def window_option_callback(value: int):
    if value < 0:
        raise typer.BadParameter("Window option must be positive")
    return value


@app.command()
def main(
    reference_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
        help="Path to reference image"
    ),
    moving_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
        help="Path to moving image"
    ),
    outfile_dir: Path = typer.Option(
        Path(__file__).parent.absolute(),
        "--out-path",
        "-op",
        exists=False,
        file_okay=True,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
        help="Path to desired output directory"
    ),
    outfile_name: str = typer.Option(
        f"parallax_{datetime.now().isoformat(timespec='minutes').replace(':', '.')}",
        "--out-name", "-on",
        help="Desired output name"
    ),
    col_start: Optional[int] = typer.Option(
        -1,
        "--col-start", "-cs", "-x0",
        show_default=False,
        help="Upper left starting column - X0",
        callback=index_callback
    ),
    col_end: Optional[int] = typer.Option(
        -1,
        "--col-end", "-ce", "-x1",
        show_default=False,
        help="Lower right ending column - X1",
        callback=index_callback
    ),
    row_start: Optional[int] = typer.Option(
        -1,
        "--row-start", "-rs", "-y0",
        show_default=False,
        help="Upper right starting row - Y0",
        callback=index_callback
    ),
    row_end: Optional[int] = typer.Option(
        -1,
        "--row-end", "-re", "-y1",
        show_default=False,
        help="Lower right ending row - Y1",
        callback=index_callback
    ),
    window_size: Optional[int] = typer.Option(
        64,
        "--window-size", "-wsize",
        help="Correlation window size",
        callback=window_option_callback,
    ),
    window_step: Optional[int] = typer.Option(
        6,
        "--window-step", "-wstep",
        help="Correlation window step size",
        callback=window_option_callback
    ),
):
    PhaseCorrelationControl(
        reference_path,
        moving_path,
        outfile_dir=outfile_dir,
        outfile_name=outfile_name,
        col_start=col_start,
        col_end=col_end,
        row_start=row_start,
        row_end=row_end,
        window_size=window_size,
        window_step=window_step,
    )  


if __name__ == "__main__":
    app()
