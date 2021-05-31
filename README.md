# Sentinel2PhaseCrossCorrelation

## CLI

    python main.py --help

 explicitly

    python main.py [REFERNCE IMAGE PATH] [MOVING IMAGE PATH] **OPTIONS

## By Import

    from OPCC import PhaseCorrelationControl
    
    reference_image_path: Union[str, Path] = "path/to/reference/image"
    moving_image_path: Union[str, Path] = "path/to/moving/image"
    
    PhaseCorrelationControl(
        reference_image_path,
        moving_image_path
    )

unless specified, `outfile_dir` is the absolute dir of `main.py` and `outfile_name` is `parallax_ISOTIMESTAMP`
