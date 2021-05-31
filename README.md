# Sentinel2PhaseCrossCorrelation

## CLI

    python S2PhaseCrossCorrelation/main.py --help

 explicitly

    python S2PhaseCrossCorrelation/main.py [REFERENCE IMAGE PATH] [MOVING IMAGE PATH] **OPTIONS

## By Import

    from OPCC import PhaseCorrelationControl
    
    reference_image_path: Union[str, Path] = "path/to/reference/image"
    moving_image_path: Union[str, Path] = "path/to/moving/image"
    
    PhaseCorrelationControl(
        reference_image_path,
        moving_image_path
    )

unless specified, `outfile_dir` is the absolute dir of `main.py` and `outfile_name` is `parallax_ISOTIMESTAMP`

## ReCompile OPCC algorithm

    cd S2PhaseCrossCorrelation/OPCC
    python setup.py build_ext --inplace
