# Building and running with Conda

## 1. Create the environment

From the **python** directory:

```bash
cd /path/to/OSM-S-BKI/python
conda env create -f environment.yml
```

This creates an environment named `osm_bki` with Python, NumPy, pybind11, a C++ compiler, and optional script dependencies.

## 2. Activate and build the extension

```bash
conda activate osm_bki
pip install -e .
```

This builds the `composite_bki_cpp` extension and installs the `osm_bki` package in editable mode.

## 3. Run the scripts

From the **python** directory (so that `configs/` and paths in scripts resolve correctly):

```bash
python scripts/basic_usage.py
```

Or run from anywhere after installation:

```bash
python -c "import composite_bki_cpp; print(composite_bki_cpp.run_pipeline)"
```

## Troubleshooting

- **ModuleNotFoundError: composite_bki_cpp**  
  Ensure you ran `pip install -e .` from the `python` directory with the conda env activated.

- **Compiler errors**  
  The `cxx-compiler` package from conda-forge provides a C++17-capable compiler. If you use a system compiler instead, ensure it supports C++17 and OpenMP (e.g. `g++-9` or newer).

- **OpenMP**  
  The extension uses OpenMP if available. On macOS with conda’s clang, OpenMP may require `conda install libomp` or using the conda compiler stack.
