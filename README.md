# Sentinel2PhaseCrossCorrelation

Optimized and pythonic program for phase cross correlation of imagery products

## Comparison

|                             | Time (seconds) | Note |
|-----------------------------|:--------------:|------|
| Scikit-Image - 0 Upscaling |      307      |      |
| **.this - 0 Upscaling**        |       **74**       |      |
| Scikit-Image - 100 Upscaling |      1020      |      |
|     **.this - 100 Upscaling**    |       **780**      | *optimized upscaling not yet implemented, benchmarked using scikit-image dft upscaling* |

*as benchmarked on an i7-4790K @ 4.0 GHz, 16GB ram*

## CLI

    python S2PhaseCrossCorrelation/main.py --help

 explicitly

    python S2PhaseCrossCorrelation/main.py [REFERENCE IMAGE PATH] [MOVING IMAGE PATH] **OPTIONS

## Object

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

## ToDo

- Implement optimized DFT upscaling
- explore Parallelization options
