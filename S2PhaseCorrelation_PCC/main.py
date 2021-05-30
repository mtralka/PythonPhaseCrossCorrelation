"""

 @title: Optimized Sentinel-2 Coregistration using Phase Cross Correlation
 @author: Matthew Tralka
 @date: May 2021
 @version: .1

"""
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from PhaseCorrelationControl import PhaseCorrelationControl


def main(
    reference_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
    moving_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
    outfile_path: Path = typer.Option(
        Path(__file__).parent.absolute(),
        "--out-path",
        "-op",
        exists=False,
        file_okay=True,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
    outfile_name: str = typer.Option(
        f"parallax_{datetime.now().isoformat(timespec='minutes').replace(':', '.')}",
        "--out-name", "-on"
    ),
    col_start: Optional[int] = typer.Option(-1, "--col-start", "-cs"),
    col_end: Optional[int] = typer.Option(-1, "--col-end", "-ce"),
    row_start: Optional[int] = typer.Option(-1, "--row-start", "-rs"),
    row_end: Optional[int] = typer.Option(-1, "--row-end", "-re"),
    window_size: Optional[int] = typer.Option(64),
    window_step: Optional[int] = typer.Option(6),
):
    start = datetime.now()
    PhaseCorrelationControl(
        reference_path,
        moving_path,
        outfile_path=outfile_path,
        outfile_name=outfile_name,
        col_start=col_start,
        col_end=col_end,
        row_start=row_start,
        row_end=row_end,
        window_size=window_size,
        window_step=window_step,
    )
    print(datetime.now() - start)


if __name__ == "__main__":
    typer.run(main)
