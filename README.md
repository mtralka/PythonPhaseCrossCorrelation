# Python Phase Cross Correlation

Optimized CPU & GPU implementation of phase cross correlation. Target usage for remote sensing coregistration

## Comparison

|                              | Time (seconds) | Improvement | Note                  |
|------------------------------|:--------------:|-------------|-----------------------|
| Scikit-Image - No Upscaling  |       307        |             |                       |
| .this - No Upscaling         |       74       |     **414%**    |                       |
| Scikit-Image - 100 Upscaling |      1020      |             |                       |
|     .this - 100 Upscaling    |       540      |     **188%**    | *Not fully optimized* |

*as benchmarked on an i7-4790K @ 4.0 GHz, 16GB ram*

## Dependencies

This project uses `conda`. Please create and activate the required conda command using the `environment.yml` found in `PythonPhaseCrossCorrelation/environment.yml`

    conda env create -f environment.yml
    conda activate PPCC

## CLI

    python PythonPhaseCrossCorrelation/main.py --help

 explicitly

    python PythonPhaseCrossCorrelation/main.py [REFERENCE IMAGE PATH] [MOVING IMAGE PATH] **OPTIONS

## Object

    from PCC import PhaseCorrelationControl
    
    reference_image_path: Union[str, Path] = "path/to/reference/image"
    moving_image_path: Union[str, Path] = "path/to/moving/image"
    
    PhaseCorrelationControl(
        reference_image_path,
        moving_image_path
    )

unless specified, `outfile_dir` is the absolute dir of `main.py` and `outfile_name` is `parallax_{ISOTIMESTAMP}`

## ReCompile CPU-based PCC algorithm

    cd PythonPhaseCrossCorrelation/PCC/CPU
    python setup.py build_ext --inplace

## GPU-based PCC algorithm

    not yet implemented

## Test

    cd PythonPhaseCrossCorrelation/tests
    pytest

Validates PCC results against benchmarked data

## ToDo

- Implement optimized DFT upscaling
  - validate results
- finish implementation of PyTorch-based GPU algorithm
- explore Parallelization options
