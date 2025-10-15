"""
Semi-naive Quantum Bayes Classifier (SPODE variant)
Implements dependency structures for improved accuracy
"""

import numpy as np
from sklearn.model_selection import train_test_split
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from tqdm import tqdm
from quantum_bayes_classifier import (
    f_of_P, 
    gaussian_intersection_threshold,
    load_and_preprocess_data,
    filter_binary_classes,
    evaluate_classifier
)


class SemiNaiveQBC:
    """
    Semi-naive Quantum Bayes Classifier implementing SPODE structure
    Adds pairwise dependencies between attributes
    """
    
    def __init__(self, n_attributes, n_classes=2, parent_strategy='sequential'):
        """
        Initialize Semi-naive QBC
        
        Args:
            n_attributes: Number of feature attributes
            n_classes: Number of classes (binary for now)
            parent_strategy: How to assign parent attributes
                - 'sequential': Each attribute's parent is the previous one
                - 'first': All attributes depend on the first one (SPODE)
        """
        self.n_attributes = n_attributes
        self.n_classes = n_classes
        self.parent_strategy = parent_strategy
        self.class_probs = None
        self.cond_probs = None  # P(x_i | y)
        self.joint_cond_probs = None  # P(x_j | y, x_i) for dependencies
        self.thresholds = None
        self.sampling_positions = None
        self.parent_map = None
        
    def _build_parent_structure(self):
        """Build parent-child dependency structure"""
        self.parent_map = {}
        
        if self.parent_strategy == 'sequential':
            # Each attribute depends on previous: x₁→x₂→x₃...
            for i in range(1, self.n_attributes):
                self.parent_map[i] = i - 1
        elif self.parent_strategy == 'first':
            # All depend on first attribute (SPODE): x₀→x₁, x₀→x₂, ...
            for i in range(1, self.n_attributes):
                self.parent_map[i] = 0
        else:
            raise ValueError(f"Unknown parent strategy: {self.parent_strategy}")
    
    def _build_quantum_circuit(self):
        """
        Build quantum circuit with pairwise dependencies
        Uses controlled-controlled-Ry for P(x_j | y, x_i)
        """
        if self.n_classes != 2:
            raise NotImplementedError("Currently only binary classification supported")
        
        qr_y = QuantumRegister(1, 'y')
        qr_x = QuantumRegister(self.n_attributes, 'x')
        cr = ClassicalRegister(1 + self.n_attributes, 'c')
        qc = QuantumCircuit(qr_y, qr_x, cr)
        
        # Encode class prior P(y=0)
        theta_y = f_of_P(self.class_probs[0])
        qc.ry(theta_y, qr_y[0])
        
        # First attribute: depends only on class (no parent)
        theta_y1 = f_of_P(self.cond_probs[1][0][0])
        qc.mcry(theta_y1, [qr_y[0]], qr_x[0], None, mode='noancilla')
        
        theta_y0 = f_of_P(self.cond_probs[0][0][0])
        qc.x(qr_y[0])
        qc.mcry(theta_y0, [qr_y[0]], qr_x[0], None, mode='noancilla')
        qc.x(qr_y[0])
        
        # Remaining attributes: depend on class AND parent attribute
        for i in range(1, self.n_attributes):
            parent_idx = self.parent_map[i]
            
            # Four cases: (y=0,x_parent=0), (y=0,x_parent=1), (y=1,x_parent=0), (y=1,x_parent=1)
            for y_val in [0, 1]:
                for parent_val in [0, 1]:
                    # Get conditional probability P(x_i=0 | y=y_val, x_parent=parent_val)
                    prob = self.joint_cond_probs[y_val][i][parent_val][0]
                    theta = f_of_P(prob)
                    
                    # Apply controlled-controlled Ry
                    # Control on y_val and parent_val
                    controls = []
                    
                    # Setup control bits
                    if y_val == 0:
                        qc.x(qr_y[0])
                    if parent_val == 0:
                        qc.x(qr_x[parent_idx])
                    
                    # Apply controlled rotation
                    qc.mcry(theta, [qr_y[0], qr_x[parent_idx]], qr_x[i], None, mode='noancilla')
                    
                    # Restore control bits
                    if parent_val == 0:
                        qc.x(qr_x[parent_idx])
                    if y_val == 0:
                        qc.x(qr_y[0])
        
        # Measure all qubits
        qc.measure(qr_y[0], cr[0])
        for i in range(self.n_attributes):
            qc.measure(qr_x[i], cr[i+1])
        
        return qc
    
    def fit(self, X_raw, y, sampling_positions=None):
        """Train the semi-naive QBC"""
        n_samples = len(X_raw)
        
        # Build parent structure
        self._build_parent_structure()
        
        # Sample fixed positions
        if sampling_positions is None:
            self.sampling_positions = np.random.choice(
                X_raw.shape[1], size=self.n_attributes, replace=False
            )
        else:
            self.sampling_positions = sampling_positions
        
        X_sampled = X_raw[:, self.sampling_positions]
        
        # Gaussian-based binarization
        self.thresholds = np.zeros(self.n_attributes)
        X_binary = np.zeros_like(X_sampled, dtype=int)
        
        for i in range(self.n_attributes):
            class_stats = []
            for c in range(self.n_classes):
                class_data = X_sampled[y == c, i]
                mu = np.mean(class_data)
                sigma = np.std(class_data) + 1e-6
                class_stats.append((mu, sigma))
            
            mu0, sigma0 = class_stats[0]
            mu1, sigma1 = class_stats[1]
            self.thresholds[i] = gaussian_intersection_threshold(mu0, sigma0, mu1, sigma1)
            X_binary[:, i] = (X_sampled[:, i] > self.thresholds[i]).astype(int)
        
        # Calculate class priors
        self.class_probs = np.zeros(self.n_classes)
        for c in range(self.n_classes):
            self.class_probs[c] = np.sum(y == c) / n_samples
        
        # Calculate marginal conditional probabilities P(x_i | y) for first attribute
        self.cond_probs = {}
        for c in range(self.n_classes):
            self.cond_probs[c] = np.zeros((self.n_attributes, 2))
            class_samples = X_binary[y == c]
            
            for i in range(self.n_attributes):
                epsilon = 1e-6
                self.cond_probs[c][i][0] = np.clip(np.mean(class_samples[:, i] == 0), epsilon, 1-epsilon)
                self.cond_probs[c][i][1] = 1 - self.cond_probs[c][i][0]
        
        # Calculate joint conditional probabilities P(x_j | y, x_i) for dependencies
        self.joint_cond_probs = {}
        for c in range(self.n_classes):
            self.joint_cond_probs[c] = {}
            class_samples = X_binary[y == c]
            
            for child_idx in range(1, self.n_attributes):
                parent_idx = self.parent_map[child_idx]
                self.joint_cond_probs[c][child_idx] = np.zeros((2, 2))  # [parent_val][child_val]
                
                for parent_val in [0, 1]:
                    # Samples where parent has value parent_val
                    parent_mask = class_samples[:, parent_idx] == parent_val
                    conditional_samples = class_samples[parent_mask]
                    
                    if len(conditional_samples) > 0:
                        epsilon = 1e-6
                        # P(x_child=0 | y=c, x_parent=parent_val)
                        p_0 = np.mean(conditional_samples[:, child_idx] == 0)
                        self.joint_cond_probs[c][child_idx][parent_val][0] = np.clip(p_0, epsilon, 1-epsilon)
                        self.joint_cond_probs[c][child_idx][parent_val][1] = 1 - self.joint_cond_probs[c][child_idx][parent_val][0]
                    else:
                        # Fallback to marginal if no samples
                        self.joint_cond_probs[c][child_idx][parent_val][0] = self.cond_probs[c][child_idx][0]
                        self.joint_cond_probs[c][child_idx][parent_val][1] = self.cond_probs[c][child_idx][1]
        
        # Build circuit
        self.circuit = self._build_quantum_circuit()
        
        print(f"Semi-naive QBC trained with {self.parent_strategy} parent structure")
        print(f"Class priors: {self.class_probs}")
    
    def _binarize_sample(self, x_raw):
        """Binarize sample using learned thresholds"""
        x_sampled = x_raw[self.sampling_positions]
        x_binary = (x_sampled > self.thresholds).astype(int)
        return x_binary
    
    def predict(self, X_raw, shots=3000):
        """Predict classes for input features"""
        backend = Aer.get_backend('qasm_simulator')
        transpiled_qc = transpile(self.circuit, backend)
        
        job = backend.run(transpiled_qc, shots=shots)
        counts = job.result().get_counts()
        
        predictions = []
        
        for x_raw in tqdm(X_raw, desc="Predicting (Semi-naive)"):
            x_pattern = self._binarize_sample(x_raw)
            
            counts_y_given_pattern = {0: 0, 1: 0}
            
            for bitstring, count in counts.items():
                bits = bitstring[::-1]
                y_bit = int(bits[0])
                attr_bits = [int(bits[i+1]) for i in range(self.n_attributes)]
                
                if np.array_equal(attr_bits, x_pattern):
                    counts_y_given_pattern[y_bit] += count
            
            if counts_y_given_pattern[0] + counts_y_given_pattern[1] == 0:
                pred = 0 if self.class_probs[0] > self.class_probs[1] else 1
            else:
                pred = max(counts_y_given_pattern.items(), key=lambda x: x[1])[0]
            
            predictions.append(pred)
        
        return np.array(predictions)


def compare_naive_vs_seminaive():
    """Compare naive and semi-naive QBC performance"""
    print("="*80)
    print("NAIVE vs SEMI-NAIVE QBC COMPARISON")
    print("="*80)
    
    # Load data
    np.random.seed(42)
    X_train_full, X_test_full, y_train_full, y_test_full = load_and_preprocess_data(
        'mnist', n_train=5000, n_test=500
    )
    
    X_train, X_test, y_train, y_test = filter_binary_classes(
        X_train_full, X_test_full, y_train_full, y_test_full, class0=0, class1=1
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    results = {}
    
    # Test naive QBC
    print("\n" + "-"*80)
    print("Training Naive QBC...")
    print("-"*80)
    from quantum_bayes_classifier import QuantumBayesClassifier
    
    naive_qbc = QuantumBayesClassifier(n_attributes=9, n_classes=2)
    naive_qbc.fit(X_train, y_train)
    naive_preds = naive_qbc.predict(X_test, shots=3000)
    naive_acc = evaluate_classifier(y_test, naive_preds)
    results['Naive QBC'] = naive_acc
    print(f"Naive QBC Accuracy: {naive_acc:.4f}")
    
    # Test semi-naive QBC with sequential structure
    print("\n" + "-"*80)
    print("Training Semi-naive QBC (Sequential)...")
    print("-"*80)
    seminaive_seq = SemiNaiveQBC(n_attributes=9, n_classes=2, parent_strategy='sequential')
    seminaive_seq.fit(X_train, y_train)
    seq_preds = seminaive_seq.predict(X_test, shots=3000)
    seq_acc = evaluate_classifier(y_test, seq_preds)
    results['Semi-naive (Sequential)'] = seq_acc
    print(f"Semi-naive (Sequential) Accuracy: {seq_acc:.4f}")
    
    # Test semi-naive QBC with SPODE structure
    print("\n" + "-"*80)
    print("Training Semi-naive QBC (SPODE)...")
    print("-"*80)
    seminaive_spode = SemiNaiveQBC(n_attributes=9, n_classes=2, parent_strategy='first')
    seminaive_spode.fit(X_train, y_train)
    spode_preds = seminaive_spode.predict(X_test, shots=3000)
    spode_acc = evaluate_classifier(y_test, spode_preds)
    results['Semi-naive (SPODE)'] = spode_acc
    print(f"Semi-naive (SPODE) Accuracy: {spode_acc:.4f}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for name, acc in results.items():
        print(f"{name:<30} {acc:.4f}")
    
    improvement = max(seq_acc, spode_acc) - naive_acc
    print(f"\nImprovement from dependencies: {improvement:+.4f}")
    
    return results


if __name__ == "__main__":
    compare_naive_vs_seminaive()
