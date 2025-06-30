# AWE Production Estimation

**AWE Production Estimation** is a tool for estimating production output and performance of **Airborne Wind Energy (AWE)** systems. It provides a framework to evaluate and forecast the energy yield of AWE technologies based on configurable parameters and input data.

## Installation

To get started, first clone the repository:

```bash
git clone https://github.com/yourusername/awe-production-estimation.git
cd awe-production-estimation
```

### Option 1: Using Python `venv`

Ensure you have **Python 3.6 or later** installed.

1. Create a virtual environment inside the project folder:

    ```bash
    python -m venv venv
    ```

2. Activate the virtual environment:

   - On **Linux/macOS**:
     ```bash
     source venv/bin/activate
     ```
   - On **Windows**:
     ```cmd
     venv\Scripts\activate
     ```

3. Install required packages:

    ```bash
    pip install -r requirements.txt
    ```

### Option 2: Using Conda

Ensure you have **conda** installed (Anaconda or Miniconda).

1. Create a conda environment inside the project directory:

    ```bash
    conda create --prefix ./venv python=3.8
    ```

2. Activate the environment:

   - On **Linux/macOS**:
     ```bash
     source activate ./venv
     ```
   - On **Windows**:
     ```cmd
     conda activate .\venv
     ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

> 📝 **Note**: Replace `python=3.8` in the conda command with your desired version (must be >= 3.6) if needed.
