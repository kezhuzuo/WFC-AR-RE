1. # Approximate Reasoning (AR) Implementation

## Overview

"Ablation.py" provides an implementation of an Ablation experiment and an Approximate Reasoning (AR) method
for belief function fusion. The core functionality includes:

- **Ablation.py**: Implements an ablation study to evaluate the impact of Approximate Reasoning (AR) in the fusion
  process.
- **AR.py**: Implements an approximation algorithm to BBA for improved fusion efficiency.

## Features

- **Belief Function Fusion**: Implements an approximation technique for handling BBA.
- **Focal Element Approximation**: Reduces the complexity of BBAs while preserving critical belief mass information.

## Usage

### Running the Approximate Reasoning Algorithm

```python
import numpy as np
import AR

# Example inputs
BoE = np.array([[[0.3, 0.4, 0.3]], [[0.2, 0.5, 0.3]]])
P = ["A", "B", "C"]

BoE_app, P1 = AR.BBA_approximation(BoE, P)
print("Approximated BBA:", BoE_app)
print("Retained Focal Elements:", P1)
```

### Running the Ablation Experiment

```python
python
Ablation.py
```

## Explanation of Key Functions

### `BBA_approximation(BoE, P)`

- **Input**:
    - `BoE`: A NumPy array representing all BBAs
    - `P`: A list of focal elements.
- **Process**:
    - Selects the two highest belief elements per BBA.
    - Computes their union to form a new refined focal element.
    - Updates belief masses accordingly.
- **Output**:
    - `BoE_app`: The approximated BAAs.
    - `P1`: The updated set of focal elements.


2. ## Overview

"Example.py" implements various fusion methods. The primary objective is to evaluate the performance of different fusion
strategies, including:

- **Dempster-Shafer Theory (DST) Fusion**
- **Murphy's Rule**
- **Xiao's BJS Method**
- **Tang's Uncertainty Negation Measure**
- **Proposed WFC-AR-RE Method**

The implementation is designed for experimentation with various fusion methods, enabling a comparative analysis of their
efficiency and accuracy.

## Installation

### Requirements

Ensure you have Python 3 installed, along with the following dependencies:

## Usage

### Running the Fusion Experiments

Execute the main script to run the fusion methods and compare their results:

```bash
python Ablation.py
```

This will output the fusion results for different methods, including the proposed **WFC-AR-RE** method.

### Understanding the Code

#### **1. Evidence Fusion Implementation**

The project includes various evidence fusion techniques:

- **Dempster-Shafer Fusion (DST):** Implements classical Dempster's rule for fusing multiple sources.
- **Murphy’s Rule:** A averaging method for fusing evidence.
- **Xiao’s BJS Divergence Method:** A statistical method incorporating divergence measures.
- **Tang’s Uncertainty Negation:** Evaluates and modifies uncertainty in belief functions.
- **Proposed WFC-AR-RE:** Combines Approximate Reasoning (AR) and Reliability Evaluation (RE) to enhance fusion.

### Expected Output

The script will print the fusion results:

## Code Structure

```
|-- evidence-fusion/
    |-- Ablation.py          # Main script for fusion experiments
    |-- AR.py                # Approximate reasoning methods
    |-- fusionRules.py       # Evidence fusion implementations
    |-- SOTA.py              # State-of-the-art comparison methods
    |-- similarityMeasure.py # Similarity measures between beliefs
```

3. # Installation

### Prerequisites

Ensure you have Python installed (recommended version: 3.8+). Install required dependencies using:

```bash
pip install -r requirements.txt
```

## Contributing

Feel free to submit issues or pull requests to improve the project.

## Contact

For any questions, contact [kezhuzuo@seu.edu.cn].

## License

This project is licensed under the License.


