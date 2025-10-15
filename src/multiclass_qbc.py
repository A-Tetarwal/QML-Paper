"""
Multi-class Quantum Bayes Classifier
Extends QBC to handle K>2 classes using multiple class qubits
"""

import numpy as np
from sklearn.model_selection import train_test_split
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from tqdm import tqdm
import math
from quantum_bayes_classifier import (
    f_of_P,
    gaussian_intersection_threshold,
    load_and_preprocess_data,
    evaluate_classifier
)


class MultiClassQBC:
    """
    Multi-class Quantum Bayes Classifier
    Uses ceil(log2(K)) qubits to encode K classes
    """
    
    def __init__(self, n_attributes, n_classes=10):
        """
        Initialize multi-class QBC
        
        Args:
            n_attributes: Number of feature attributes
            n_classes: Number of classes (K)
        """
        self.n_attributes = n_attributes
        self.n_classes = n_classes
        self.n_class_qubits = math.ceil(math.log2(n_classes))
        self.class_probs = None
        self.cond_probs = None
        self.thresholds = None
        self.sampling_positions = None
        self.class_to_bitstring = {}
        self.bitstring_to_class = {}
        
        # Create binary encoding for each class
        for c in range(n_classes):
            bitstring = format(c, f'0{self.n_class_qubits}b')
            self.class_to_bitstring[c] = bitstring
            self.bitstring_to_class[bitstring] = c
    
    def _encode_class_prior(self, qc, qr_y):
        """
        Encode class prior probabilities into class qubits
        Uses cascade of controlled rotations
        """
        # Encode full probability distribution over K classes
        # Strategy: use Ry rotations to create superposition with desired amplitudes
        
        # For simplicity, we'll use a direct state preparation approach
        # This is a simplified version; more sophisticated methods exist
        
        # First qubit: encode P(most significant bit = 0)
        # Calculate probability that MSB is 0
        p_msb_0 = sum(self.class_probs[c] for c in range(self.n_classes) 
                      if self.class_to_bitstring[c][0] == '0')
        
        if p_msb_0 > 1e-12 and p_msb_0 < 1 - 1e-12:
            theta = f_of_P(p_msb_0)
            qc.ry(theta, qr_y[0])
        
        # Subsequent qubits: conditional on previous qubits
        for qubit_idx in range(1, self.n_class_qubits):
            # For each possible state of previous qubits, compute conditional probability
            for prev_bits in range(2**qubit_idx):
                prev_bitstring = format(prev_bits, f'0{qubit_idx}b')
                
                # Count classes matching this prefix with current bit = 0
                total_prob = 0
                prob_current_0 = 0
                
                for c in range(self.n_classes):
                    class_bits = self.class_to_bitstring[c]
                    if class_bits[:qubit_idx] == prev_bitstring:
                        total_prob += self.class_probs[c]
                        if class_bits[qubit_idx] == '0':
                            prob_current_0 += self.class_probs[c]
                
                if total_prob > 1e-12:
                    conditional_prob = prob_current_0 / total_prob
                    theta = f_of_P(conditional_prob)
                    
                    # Apply controlled rotation based on previous qubits
                    controls = []
                    for i, bit in enumerate(prev_bitstring):
                        if bit == '0':
                            qc.x(qr_y[i])
                    
                    if qubit_idx > 0:
                        qc.mcry(theta, list(qr_y[:qubit_idx]), qr_y[qubit_idx], None, mode='noancilla')
                    
                    for i, bit in enumerate(prev_bitstring):
                        if bit == '0':
                            qc.x(qr_y[i])
    
    def _build_quantum_circuit(self):
        """Build quantum circuit for multi-class classification"""
        qr_y = QuantumRegister(self.n_class_qubits, 'y')
        qr_x = QuantumRegister(self.n_attributes, 'x')
        cr = ClassicalRegister(self.n_class_qubits + self.n_attributes, 'c')
        qc = QuantumCircuit(qr_y, qr_x, cr)
        
        # Encode class prior distribution
        self._encode_class_prior(qc, qr_y)
        
        # For each attribute, encode conditional probabilities
        # Controlled on each possible class pattern
        for attr_idx in range(self.n_attributes):
            for c in range(self.n_classes):
                class_bits = self.class_to_bitstring[c]
                theta = f_of_P(self.cond_probs[c][attr_idx][0])
                
                # Setup controls for this class pattern
                for i, bit in enumerate(class_bits):
                    if bit == '0':
                        qc.x(qr_y[i])
                
                # Apply controlled rotation
                qc.mcry(theta, list(qr_y), qr_x[attr_idx], None, mode='noancilla')
                
                # Restore
                for i, bit in enumerate(class_bits):
                    if bit == '0':
                        qc.x(qr_y[i])
        
        # Measure all qubits
        for i in range(self.n_class_qubits):
            qc.measure(qr_y[i], cr[i])
        for i in range(self.n_attributes):
            qc.measure(qr_x[i], cr[self.n_class_qubits + i])
        
        return qc
    
    def fit(self, X_raw, y, sampling_positions=None):
        """Train the multi-class QBC"""
        n_samples = len(X_raw)
        
        # Sample fixed positions
        if sampling_positions is None:
            self.sampling_positions = np.random.choice(
                X_raw.shape[1], size=self.n_attributes, replace=False
            )
        else:
            self.sampling_positions = sampling_positions
        
        X_sampled = X_raw[:, self.sampling_positions]
        
        # Gaussian-based binarization (pairwise for each attribute)
        self.thresholds = np.zeros(self.n_attributes)
        X_binary = np.zeros_like(X_sampled, dtype=int)
        
        for i in range(self.n_attributes):
            # Use first two classes for threshold determination
            # (could be improved to use all classes)
            class0_data = X_sampled[y == 0, i]
            class1_data = X_sampled[y == 1, i]
            
            if len(class0_data) > 0 and len(class1_data) > 0:
                mu0, sigma0 = np.mean(class0_data), np.std(class0_data) + 1e-6
                mu1, sigma1 = np.mean(class1_data), np.std(class1_data) + 1e-6
                self.thresholds[i] = gaussian_intersection_threshold(mu0, sigma0, mu1, sigma1)
            else:
                self.thresholds[i] = np.median(X_sampled[:, i])
            
            X_binary[:, i] = (X_sampled[:, i] > self.thresholds[i]).astype(int)
        
        # Calculate class priors
        self.class_probs = np.zeros(self.n_classes)
        for c in range(self.n_classes):
            count = np.sum(y == c)
            self.class_probs[c] = max(count / n_samples, 1e-6)
        
        # Normalize
        self.class_probs /= np.sum(self.class_probs)
        
        # Calculate conditional probabilities P(x_i | y) for each class
        self.cond_probs = {}
        for c in range(self.n_classes):
            self.cond_probs[c] = np.zeros((self.n_attributes, 2))
            class_samples = X_binary[y == c]
            
            if len(class_samples) > 0:
                for i in range(self.n_attributes):
                    epsilon = 1e-6
                    p0 = np.mean(class_samples[:, i] == 0)
                    self.cond_probs[c][i][0] = np.clip(p0, epsilon, 1-epsilon)
                    self.cond_probs[c][i][1] = 1 - self.cond_probs[c][i][0]
            else:
                # Uniform if no samples
                for i in range(self.n_attributes):
                    self.cond_probs[c][i][0] = 0.5
                    self.cond_probs[c][i][1] = 0.5
        
        # Build circuit
        print(f"Building multi-class circuit with {self.n_class_qubits} class qubits for {self.n_classes} classes...")
        self.circuit = self._build_quantum_circuit()
        
        print(f"Multi-class QBC trained")
        print(f"Class priors: {self.class_probs[:5]}... (showing first 5)")
    
    def _binarize_sample(self, x_raw):
        """Binarize sample using learned thresholds"""
        x_sampled = x_raw[self.sampling_positions]
        x_binary = (x_sampled > self.thresholds).astype(int)
        return x_binary
    
    def predict(self, X_raw, shots=5000):
        """Predict classes for input features"""
        backend = Aer.get_backend('qasm_simulator')
        transpiled_qc = transpile(self.circuit, backend)
        
        job = backend.run(transpiled_qc, shots=shots)
        counts = job.result().get_counts()
        
        predictions = []
        
        for x_raw in tqdm(X_raw, desc="Predicting (Multi-class)"):
            x_pattern = self._binarize_sample(x_raw)
            
            # Count occurrences for each class
            class_counts = {c: 0 for c in range(self.n_classes)}
            
            for bitstring, count in counts.items():
                bits = bitstring[::-1]
                
                # Extract class bits and attribute bits
                class_bits = ''.join(bits[:self.n_class_qubits])
                attr_bits = [int(bits[self.n_class_qubits + i]) for i in range(self.n_attributes)]
                
                # Check if attributes match
                if np.array_equal(attr_bits, x_pattern):
                    # Decode class from bitstring
                    if class_bits in self.bitstring_to_class:
                        c = self.bitstring_to_class[class_bits]
                        class_counts[c] += count
            
            # Predict class with highest count
            total_counts = sum(class_counts.values())
            if total_counts == 0:
                # Fallback to prior
                pred = np.argmax(self.class_probs)
            else:
                pred = max(class_counts.items(), key=lambda x: x[1])[0]
            
            predictions.append(pred)
        
        return np.array(predictions)


def test_multiclass_qbc():
    """Test multi-class QBC on MNIST 10-class problem"""
    print("="*80)
    print("MULTI-CLASS QBC TEST (10 digits)")
    print("="*80)
    
    # Load data
    np.random.seed(42)
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        'mnist', n_train=10000, n_test=1000
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Classes: {np.unique(y_train)}")
    
    # Train multi-class QBC
    print("\nTraining Multi-class QBC...")
    mc_qbc = MultiClassQBC(n_attributes=9, n_classes=10)
    mc_qbc.fit(X_train, y_train)
    
    # Predict
    print("\nPredicting...")
    predictions = mc_qbc.predict(X_test[:100], shots=5000)  # Test on subset first
    
    accuracy = evaluate_classifier(y_test[:100], predictions)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Multi-class QBC Accuracy: {accuracy:.4f}")
    
    # Confusion analysis
    from collections import Counter
    pred_dist = Counter(predictions)
    true_dist = Counter(y_test[:100])
    
    print("\nPrediction distribution:")
    for digit in range(10):
        print(f"  Digit {digit}: predicted {pred_dist.get(digit, 0)}, actual {true_dist.get(digit, 0)}")
    
    return accuracy


if __name__ == "__main__":
    test_multiclass_qbc()
