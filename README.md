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

This project uses `conda`. To use locally, please create and activate the required conda environment using the `environment.yml` found in `PythonPhaseCrossCorrelation/environment.yml`

    conda env create -f environment.yml
    conda activate PPCC

## Docker

For convenience, this project includes a dockerfile and docker-compose for consistent and worry-free development across multiple machine.

With `docker` and `docker-compose` installed:

    docker-compose -f "docker-compose.yml" up -d --build

A Jupyter Notebook instance will then be accessible at `http://localhost:8888/?token=PPCC`

- `/PythonPhaseCrossCorrelation/` is live-mounted to `/PythonPhaseCrossCorrelation/`
- Port `8888` passed through
- Jupyter Notebook token `PPCC`

To access local-files in the docker container mount addition volumes as needed

## CLI

    python PythonPhaseCrossCorrelation/main.py --help

 explicitly

    python PythonPhaseCrossCorrelation/main.py [REFERENCE IMAGE PATH] [MOVING IMAGE PATH] **OPTIONS

![CLI Example](images\PythonPhaseCrossCorrelation-CLI-Example.png)

## Object

    from PCC import PhaseCorrelationControl
    
    reference_image_path: Union[str, Path] = "path/to/reference/image"
    moving_image_path: Union[str, Path] = "path/to/moving/image"
    
    PhaseCorrelationControl(
        reference_image_path,
        moving_image_path
    )

see `ExampleNotebook.ipynb`

unless specified, `outfile_dir` is the absolute dir of `main.py` and `outfile_name` is `parallax_{ISOTIMESTAMP}`

## ReCompile CPU-based PCC algorithm

    cd PythonPhaseCrossCorrelation/PCC/CPU
    python setup.py build_ext --inplace

    ### Windows Compiling

    Users on Windows-based machines will receive the error

        error: Unable to find vcvarsall.bat

    if they do not have the required Build Tools for Visual Studio 2019.

    To solve this dependency issue you can either

    - install [Build Tools for Visual Studio 2019](https://visualstudio.microsoft.com/downloads/). Note you only need the `Build Tools` and not the complete Visual Studio IDE

    - (**preferred**) Use [Chocolatey](https://chocolatey.org), the Window package manager.

        - Chocolatey install instructions - https://chocolatey.org/install
        - Built Tools Install - https://community.chocolatey.org/packages/visualstudio2019-workload-vctools
            `choco install visualstudio2019-workload-vctools`

## GPU-based PCC algorithm

    not yet implemented

## Test

    cd PythonPhaseCrossCorrelation/tests
    pytest

Validates PPCC results against benchmarked truth data

## ToDo

- Implement optimized DFT upscaling
  - validate results
- finish implementation of PyTorch-based GPU algorithm
- explore Parallelization options
