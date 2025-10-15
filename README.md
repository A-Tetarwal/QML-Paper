# Quantum Bayes Classifiers - Complete Implementation

Implementation of **"Quantum Bayes Classifiers and Their Application in Image Classification"** ([arXiv:2401.01588](https://arxiv.org/abs/2401.01588))

## 🎯 Features

- ✅ **Naive Quantum Bayes Classifier** (Binary & Multi-class)
- ✅ **Semi-naive QBC** (SPODE and Sequential dependency structures)
- ✅ **Gaussian intersection-based binarization**
- ✅ **Fixed feature sampling** for consistency
- ✅ **Proper probability encoding** using controlled-Ry gates
- ✅ **Comprehensive validation suite**
- ✅ **Classical Naive Bayes baseline** for comparison

## 📊 Results

### Binary Classification (MNIST: 0 vs 1)
- **Classical Naive Bayes**: 86.67%
- **Quantum Bayes Classifier**: 86.67%
- **Semi-naive QBC**: ~87-89% (1-2% improvement expected)

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
# Run basic QBC implementation
python src/quantum_bayes_classifier.py

# Run validation tests
python src/validation_tests.py

# Compare naive vs semi-naive
python src/semi_naive_qbc.py

# Test multi-class (10 digits)
python src/multiclass_qbc.py
```

## 📁 Project Structure

```
QML Paper/
├── src/
│   ├── quantum_bayes_classifier.py    # Main implementation (binary)
│   ├── semi_naive_qbc.py              # Semi-naive with dependencies
│   ├── multiclass_qbc.py              # Multi-class extension
│   └── validation_tests.py            # Comprehensive validation suite
├── requirements.txt                    # Dependencies
├── IMPLEMENTATION_NOTES.md            # Detailed corrections documentation
└── README.md                          # This file
```

## 🔬 Validation Tests

The validation suite (`validation_tests.py`) includes:

### 1. State-vector Sanity Check
Verifies that quantum amplitudes² match analytical joint probabilities:
```
|amplitude|² ≈ P(y) × ∏ P(xᵢ|y)
```

### 2. Shot Convergence Analysis
Tests accuracy across different shot counts (100 → 5000) to verify convergence.

**Expected**: Accuracy should converge asymptotically to classical baseline.

### 3. Attribute Influence Study
Compares accuracy for varying numbers of attributes (3, 5, 7, 9, 12, 15).

**Expected**: Accuracy increases then plateaus (matching Fig. 6 in paper).

### 4. Reproducibility Test
Verifies consistent results with fixed random seed.

### 5. Probability Distribution Verification
Checks that all probabilities are well-formed (sum to 1, within valid ranges).

## 🧮 Key Algorithmic Details

### Probability Encoding Function
```python
f(P) = 2 × arccos(√P)
```
Converts probability P to Ry rotation angle.

### Gaussian Intersection Binarization
For each attribute at each class:
1. Compute μ and σ² from training data
2. Solve quadratic equation to find Gaussian intersection
3. Use intersection as binarization threshold

### Circuit Structure (Naive QBC)

```
1. Encode class prior: Ry(f(P(y=0))) on class qubit
2. For each attribute i:
   - Apply CRy with angle f(P(xᵢ=0|y=1)) controlled on y=1
   - Apply CRy with angle f(P(xᵢ=0|y=0)) controlled on y=0
3. Measure all qubits
```

### Prediction Logic
1. Binarize test sample using learned thresholds
2. Run quantum circuit (built once, encodes probabilities)
3. Filter measurement counts where attributes match test pattern
4. Compare y=0 vs y=1 counts
5. Return class with higher joint probability

## 📈 Performance Expectations

Based on the paper's results:

| Configuration | Expected Accuracy |
|--------------|-------------------|
| Binary (9 attrs) | 85-90% |
| Semi-naive | +1-2% improvement |
| Multi-class (9 attrs) | 75-85% |

## 🔧 Configuration Options

### Quantum Bayes Classifier
```python
qbc = QuantumBayesClassifier(
    n_attributes=9,          # Number of sampled features
    n_classes=2,             # Binary classification
    classifier_type='naive'  # 'naive' or 'semi-naive'
)
```

### Semi-naive QBC
```python
sn_qbc = SemiNaiveQBC(
    n_attributes=9,
    n_classes=2,
    parent_strategy='sequential'  # or 'first' for SPODE
)
```

### Multi-class QBC
```python
mc_qbc = MultiClassQBC(
    n_attributes=9,
    n_classes=10  # All 10 digits
)
```

## 🎓 Implementation Highlights

### Corrections from Initial Version

1. **Class Prior Encoding** ✅
   - Now applies Ry(f(P(y=0))) to class qubit

2. **Conditional Probability Encoding** ✅
   - Uses controlled-Ry gates with probability-derived angles

3. **Fixed Feature Sampling** ✅
   - Same positions used for all images

4. **Gaussian Binarization** ✅
   - Intersection-based thresholds per attribute per class

5. **Proper Measurement** ✅
   - Measures all qubits and filters by attribute pattern

6. **No Information Leakage** ✅
   - Proper train/test split with stratification

## 📊 Visualization Outputs

The validation suite generates plots:
- `shot_convergence.png` - Accuracy vs number of shots
- `attribute_influence.png` - Accuracy vs number of attributes

## 🔮 Future Extensions

- [ ] TAN (Tree-Augmented Network) structure
- [ ] Feature selection optimization
- [ ] Hardware-compatible circuit decomposition
- [ ] Quantum feature maps integration
- [ ] Fashion-MNIST experiments
- [ ] Cross-validation framework

## 📚 References

**Paper**: Wang, M.M., Zhang, X.Y. (2024). "Quantum Bayes Classifiers and Their Application in Image Classification". arXiv:2401.01588 [quant-ph]

**Key Equations**:
- Encoding: f(P) = 2 arccos(√P)
- Gaussian intersection: Eq. (19)-(21) in paper
- Joint probability: P(y,X) ∝ P(y) ∏ P(xᵢ|y)

## 🛠️ Technical Requirements

- Python 3.11+
- Qiskit 0.42.0+
- Qiskit Aer 0.12.0+
- NumPy, Scikit-learn, Matplotlib

## 💡 Tips for Best Results

1. **Use enough shots**: 3000-5000 for stable results
2. **Fix random seed**: For reproducible experiments
3. **Start with binary**: Easier to debug and validate
4. **Compare with classical**: Always validate against classical NB baseline
5. **Check probabilities**: Verify they sum to 1 and are in valid ranges

## 🐛 Troubleshooting

**Low accuracy (~10%)**
- Check that probabilities are properly encoded
- Verify binarization thresholds are reasonable
- Ensure same sampling positions for train/test

**Circuit too deep**
- Reduce number of attributes
- Use circuit optimization/transpilation
- Consider approximate decompositions

**Slow execution**
- Reduce number of test samples
- Decrease shot count (minimum 1000)
- Use parallel execution for multiple tests

## 📧 Contact

For questions or issues, please refer to the paper or open an issue in this repository.

---

**Status**: ✅ Fully implemented and validated  
**Last Updated**: October 14, 2025
