# Quantum Bayes Classifier - Correct Implementation

## Summary of Corrections

This implementation now correctly follows the paper "Quantum Bayes Classifiers and Their Application in Image Classification" (arXiv:2401.01588).

### Major Fixes Applied

#### 1. **Class Prior Encoding** ✅
- **Before**: Class qubit was created but never rotated
- **After**: Applied `Ry(f(P(y=0)))` on class qubit where `f(P) = 2 * arccos(sqrt(P))`
- **Impact**: Circuit now correctly encodes class probability distribution

#### 2. **Conditional Probability Encoding** ✅
- **Before**: Probabilities computed but never used in circuit
- **After**: Use controlled Ry gates with angles `f(P(x_i=0|y))` for each attribute
- **Impact**: Circuit amplitudes now proportional to `sqrt(P(y)) * prod(sqrt(P(x_i|y)))`

#### 3. **Feature Encoding Method** ✅
- **Before**: Encoded feature values directly as `Ry(value * π)`
- **After**: Circuit encodes probabilities only; predictions compare measurement counts for matching attribute patterns
- **Impact**: Correct quantum amplitude structure matching paper's formulation

#### 4. **Fixed Feature Sampling** ✅
- **Before**: Random positions sampled independently per image
- **After**: Sample positions once and reuse for all training and test images
- **Impact**: Consistent attribute-to-label statistics across dataset

#### 5. **Gaussian-Based Binarization** ✅
- **Before**: Simple threshold at 0.5 after StandardScaler
- **After**: Gaussian MLE with intersection-based thresholds per attribute per class
- **Impact**: Proper P(x_i|y) estimation following paper's methodology

#### 6. **Train/Test Split** ✅
- **Before**: Fitted scaler on entire dataset, brittle slicing
- **After**: Proper `train_test_split` with stratification
- **Impact**: No information leakage, proper evaluation

#### 7. **Quantum Circuit Gates** ✅
- **Before**: Used CX gates for entanglement
- **After**: Use controlled Ry gates (`mcry`) with probability-derived angles
- **Impact**: Correct encoding of conditional probabilities

#### 8. **Prediction Logic** ✅
- **Before**: Measured only class qubit, returned most frequent bit
- **After**: Measure all qubits, filter counts where attributes match test pattern, compare y=0 vs y=1 counts
- **Impact**: Correctly computes P(y, X=X*) as per paper

## Key Implementation Details

### Probability Encoding Function
```python
def f_of_P(P):
    return 2 * np.arccos(np.sqrt(np.clip(P, 1e-12, 1.0)))
```

### Gaussian Intersection Binarization
- Computes μ and σ² for each class at each sampled position
- Solves quadratic equation to find Gaussian intersection
- Creates optimal binary threshold for classification

### Circuit Structure
1. **Class qubit**: Encodes P(y=0) via Ry rotation
2. **Attribute qubits**: Each encodes P(x_i=0|y) via controlled Ry
3. **Measurement**: All qubits measured to classical register

### Prediction Process
1. Binarize test sample using learned thresholds
2. Run quantum circuit (built once, encodes probabilities)
3. Filter measurement counts matching test pattern
4. Compare y=0 vs y=1 counts
5. Return class with higher joint probability

## Classical Naive Bayes Comparison
Implemented classical NB on same binarized features to validate:
- Data preprocessing correctness
- Binarization quality
- Expected baseline performance

## Current Configuration
- **Binary classification**: Classes 0 vs 1 (easily extendable)
- **Attributes**: 9 (as per paper)
- **Training samples**: 5000
- **Test samples**: 500
- **Quantum shots**: 3000

## Expected Performance
Based on the paper:
- Classical NB: ~0.85-0.90 accuracy on MNIST binary tasks
- Quantum BC: Should match or slightly exceed classical NB
- Current implementation shows both achieving similar high accuracy

## How to Run
```bash
python src/quantum_bayes_classifier.py
```

## Future Enhancements
1. Multi-class extension (10 digits)
2. Semi-naive QBC (SPODE/TAN structures)
3. Fashion-MNIST dataset
4. Visualization of sampled features
5. Circuit depth optimization

## References
- Paper: arXiv:2401.01588
- MindQuantum → Qiskit adaptation
- Binary classification focus (paper's primary evaluation)
