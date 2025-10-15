import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def f_of_P(P):
    """
    Encoding function from probability to rotation angle.
    f(P) = 2 * arccos(sqrt(P))
    """
    return 2 * np.arccos(np.sqrt(np.clip(P, 1e-12, 1.0)))

def gaussian_intersection_threshold(mu0, sigma0, mu1, sigma1):
    """
    Find the intersection point of two Gaussian distributions.
    This is used for binarization as described in the paper.
    """
    if abs(sigma0 - sigma1) < 1e-6:
        # Equal variances: threshold is midpoint
        return (mu0 + mu1) / 2
    
    # Solve quadratic equation for intersection
    a = sigma1**2 - sigma0**2
    b = 2 * (mu0 * sigma1**2 - mu1 * sigma0**2)
    c = mu1**2 * sigma0**2 - mu0**2 * sigma1**2 + 2 * sigma0**2 * sigma1**2 * np.log(sigma1 / sigma0)
    
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return (mu0 + mu1) / 2
    
    x1 = (-b + np.sqrt(discriminant)) / (2*a)
    x2 = (-b - np.sqrt(discriminant)) / (2*a)
    
    # Choose threshold between means
    if min(mu0, mu1) <= x1 <= max(mu0, mu1):
        return x1
    elif min(mu0, mu1) <= x2 <= max(mu0, mu1):
        return x2
    else:
        return (mu0 + mu1) / 2

class QuantumBayesClassifier:
    def __init__(self, n_attributes, n_classes=2, classifier_type='naive'):
        """
        Initialize Quantum Bayes Classifier following the paper's methodology
        
        Args:
            n_attributes (int): Number of feature attributes
            n_classes (int): Number of classes (start with 2 for binary classification)
            classifier_type (str): Type of classifier ('naive' or 'semi-naive')
        """
        self.n_attributes = n_attributes
        self.n_classes = n_classes
        self.classifier_type = classifier_type
        self.class_probs = None
        self.cond_probs = None  # P(x_i | y)
        self.thresholds = None
        self.sampling_positions = None
        
    def _build_quantum_circuit(self):
        """
        Build quantum circuit encoding probabilities (not input features!)
        Circuit is built once after fit() and reused for all predictions.
        """
        if self.n_classes != 2:
            raise NotImplementedError("Currently only binary classification is supported")
        
        qr_y = QuantumRegister(1, 'y')
        qr_x = QuantumRegister(self.n_attributes, 'x')
        cr = ClassicalRegister(1 + self.n_attributes, 'c')
        qc = QuantumCircuit(qr_y, qr_x, cr)
        
        # Encode class prior P(y=0) on class qubit
        theta_y = f_of_P(self.class_probs[0])
        qc.ry(theta_y, qr_y[0])
        
        # For each attribute, encode conditional probabilities P(x_i|y)
        for i in range(self.n_attributes):
            # Angle for P(x_i=0|y=1) - controlled on y=1
            theta_y1 = f_of_P(self.cond_probs[1][i][0])
            qc.mcry(theta_y1, [qr_y[0]], qr_x[i], None, mode='noancilla')
            
            # Angle for P(x_i=0|y=0) - controlled on y=0 (flip, apply, flip back)
            theta_y0 = f_of_P(self.cond_probs[0][i][0])
            qc.x(qr_y[0])
            qc.mcry(theta_y0, [qr_y[0]], qr_x[i], None, mode='noancilla')
            qc.x(qr_y[0])
        
        # Measure all qubits
        qc.measure(qr_y[0], cr[0])
        for i in range(self.n_attributes):
            qc.measure(qr_x[i], cr[i+1])
        
        return qc
        
    def fit(self, X_raw, y, sampling_positions=None):
        """
        Train the quantum Bayes classifier
        
        Args:
            X_raw (np.array): Raw training features (images flattened)
            y (np.array): Training labels
            sampling_positions (np.array): Fixed positions to sample (if None, generate once)
        """
        n_samples = len(X_raw)
        
        # Step 1: Sample fixed positions (same for all images)
        if sampling_positions is None:
            self.sampling_positions = np.random.choice(
                X_raw.shape[1], 
                size=self.n_attributes, 
                replace=False
            )
        else:
            self.sampling_positions = sampling_positions
        
        # Extract features at sampled positions
        X_sampled = X_raw[:, self.sampling_positions]
        
        # Step 2: Gaussian-based binarization per attribute per class
        self.thresholds = np.zeros(self.n_attributes)
        X_binary = np.zeros_like(X_sampled, dtype=int)
        
        for i in range(self.n_attributes):
            # Compute Gaussian parameters for each class
            class_stats = []
            for c in range(self.n_classes):
                class_data = X_sampled[y == c, i]
                mu = np.mean(class_data)
                sigma = np.std(class_data) + 1e-6  # Avoid zero variance
                class_stats.append((mu, sigma))
            
            # Find intersection threshold between class 0 and class 1
            mu0, sigma0 = class_stats[0]
            mu1, sigma1 = class_stats[1]
            self.thresholds[i] = gaussian_intersection_threshold(mu0, sigma0, mu1, sigma1)
            
            # Binarize: 0 if below threshold, 1 if above
            X_binary[:, i] = (X_sampled[:, i] > self.thresholds[i]).astype(int)
        
        # Step 3: Calculate class priors P(y)
        self.class_probs = np.zeros(self.n_classes)
        for c in range(self.n_classes):
            self.class_probs[c] = np.sum(y == c) / n_samples
        
        # Step 4: Calculate conditional probabilities P(x_i = value | y)
        self.cond_probs = {}
        for c in range(self.n_classes):
            self.cond_probs[c] = np.zeros((self.n_attributes, 2))
            class_samples = X_binary[y == c]
            
            for i in range(self.n_attributes):
                # P(x_i = 0 | y = c)
                self.cond_probs[c][i][0] = np.mean(class_samples[:, i] == 0)
                # P(x_i = 1 | y = c)
                self.cond_probs[c][i][1] = np.mean(class_samples[:, i] == 1)
                
                # Add smoothing to avoid zero probabilities
                epsilon = 1e-6
                self.cond_probs[c][i][0] = np.clip(self.cond_probs[c][i][0], epsilon, 1 - epsilon)
                self.cond_probs[c][i][1] = 1 - self.cond_probs[c][i][0]
        
        # Step 5: Build quantum circuit (once, encoding probabilities)
        self.circuit = self._build_quantum_circuit()
        
        print(f"Class priors: {self.class_probs}")
        print(f"Sampling positions: {self.sampling_positions[:5]}... (showing first 5)")
    
    def _binarize_sample(self, x_raw):
        """Binarize a single sample using learned thresholds"""
        x_sampled = x_raw[self.sampling_positions]
        x_binary = (x_sampled > self.thresholds).astype(int)
        return x_binary
    
    def predict(self, X_raw, shots=3000):
        """
        Predict classes for input features
        
        Args:
            X_raw (np.array): Raw input features
            shots (int): Number of quantum measurements
            
        Returns:
            np.array: Predicted class labels
        """
        backend = Aer.get_backend('qasm_simulator')
        transpiled_qc = transpile(self.circuit, backend)
        
        # Run circuit once to get probability distribution
        job = backend.run(transpiled_qc, shots=shots)
        counts = job.result().get_counts()
        
        predictions = []
        
        for x_raw in tqdm(X_raw, desc="Predicting"):
            # Binarize the test sample
            x_pattern = self._binarize_sample(x_raw)
            
            # Count occurrences where attribute bits match x_pattern
            counts_y_given_pattern = {0: 0, 1: 0}
            
            for bitstring, count in counts.items():
                # Parse bitstring: Qiskit uses reverse order
                # bitstring format: 'x_{n-1}...x_1 x_0 y'
                bits = bitstring[::-1]  # Reverse to get correct order
                y_bit = int(bits[0])
                attr_bits = [int(bits[i+1]) for i in range(self.n_attributes)]
                
                # Check if attribute bits match the pattern
                if np.array_equal(attr_bits, x_pattern):
                    counts_y_given_pattern[y_bit] += count
            
            # Predict class with higher count
            if counts_y_given_pattern[0] + counts_y_given_pattern[1] == 0:
                # No exact match found, fallback to prior
                pred = 0 if self.class_probs[0] > self.class_probs[1] else 1
            else:
                pred = max(counts_y_given_pattern.items(), key=lambda x: x[1])[0]
            
            predictions.append(pred)
        
        return np.array(predictions)


def load_and_preprocess_data(dataset_name='mnist', n_train=5000, n_test=20000):
    """
    Load and preprocess MNIST or Fashion-MNIST dataset with proper train/test split
    
    Args:
        dataset_name (str): Name of the dataset ('mnist' or 'fashion-mnist')
        n_train (int): Number of training samples
        n_test (int): Number of test samples
        
    Returns:
        tuple: Training and test data (X_train, X_test, y_train, y_test)
    """
    print(f"Loading {dataset_name} dataset...")
    
    if dataset_name == 'mnist':
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    else:
        X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False, parser='auto')
    
    # Convert labels to integers
    y = y.astype(int)
    
    # Normalize pixel values to [0, 1]
    X = X.astype(np.float32) / 255.0
    
    # Use proper train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=n_train, test_size=n_test, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def filter_binary_classes(X_train, X_test, y_train, y_test, class0=0, class1=1):
    """
    Filter dataset to only include two classes for binary classification
    
    Args:
        X_train, X_test, y_train, y_test: Full dataset
        class0, class1: Two classes to keep
        
    Returns:
        Filtered dataset with only two classes
    """
    # Filter training data
    train_mask = (y_train == class0) | (y_train == class1)
    X_train_binary = X_train[train_mask]
    y_train_binary = y_train[train_mask]
    y_train_binary = (y_train_binary == class1).astype(int)  # Convert to 0/1
    
    # Filter test data
    test_mask = (y_test == class0) | (y_test == class1)
    X_test_binary = X_test[test_mask]
    y_test_binary = y_test[test_mask]
    y_test_binary = (y_test_binary == class1).astype(int)  # Convert to 0/1
    
    return X_train_binary, X_test_binary, y_train_binary, y_test_binary

def evaluate_classifier(y_true, y_pred):
    """
    Evaluate classifier performance
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
        
    Returns:
        float: Classification accuracy
    """
    accuracy = np.mean(y_true == y_pred)
    return accuracy

def test_classical_naive_bayes(X_train, X_test, y_train, y_test, qbc):
    """
    Test classical Naive Bayes on same binarized features for comparison
    """
    # Binarize training data
    X_train_bin = np.array([qbc._binarize_sample(x) for x in X_train])
    X_test_bin = np.array([qbc._binarize_sample(x) for x in X_test])
    
    # Compute class priors and conditional probs (already in qbc)
    predictions = []
    
    for x_test in X_test_bin:
        # Compute log probability for each class
        log_probs = []
        for c in range(2):
            log_p = np.log(qbc.class_probs[c])
            for i in range(qbc.n_attributes):
                log_p += np.log(qbc.cond_probs[c][i][x_test[i]])
            log_probs.append(log_p)
        
        predictions.append(np.argmax(log_probs))
    
    return np.array(predictions)

def main():
    """
    Main function implementing the paper's methodology
    """
    print("=" * 80)
    print("Quantum Bayes Classifier - Paper Implementation")
    print("=" * 80)
    
    # Parameters (following the paper)
    n_attributes = 9  # Paper uses 9 attributes
    n_train = 5000    # Training samples
    n_test = 500      # Test samples
    class0, class1 = 0, 1  # Binary classification (digits 0 vs 1)
    
    # Load and preprocess data
    X_train_full, X_test_full, y_train_full, y_test_full = load_and_preprocess_data(
        'mnist', n_train=n_train, n_test=n_test
    )
    
    # Filter to binary classification
    print(f"\nFiltering to binary classification: class {class0} vs class {class1}")
    X_train, X_test, y_train, y_test = filter_binary_classes(
        X_train_full, X_test_full, y_train_full, y_test_full, class0, class1
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Initialize and train Quantum Bayes Classifier
    print(f"\nTraining Quantum Bayes Classifier with {n_attributes} attributes...")
    qbc = QuantumBayesClassifier(n_attributes=n_attributes, n_classes=2, classifier_type='naive')
    qbc.fit(X_train, y_train)
    
    # Test classical Naive Bayes for comparison
    print("\nTesting Classical Naive Bayes (on same binarized features)...")
    classical_predictions = test_classical_naive_bayes(X_train, X_test, y_train, y_test, qbc)
    classical_accuracy = evaluate_classifier(y_test, classical_predictions)
    print(f"Classical Naive Bayes Accuracy: {classical_accuracy:.4f}")
    
    # Predict using Quantum Bayes Classifier
    print("\nPredicting with Quantum Bayes Classifier...")
    quantum_predictions = qbc.predict(X_test, shots=3000)
    quantum_accuracy = evaluate_classifier(y_test, quantum_predictions)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Classical Naive Bayes Accuracy: {classical_accuracy:.4f}")
    print(f"Quantum Bayes Classifier Accuracy: {quantum_accuracy:.4f}")
    print(f"Difference: {abs(quantum_accuracy - classical_accuracy):.4f}")
    print("=" * 80)
    
    # Print some conditional probabilities for verification
    print("\nSample Conditional Probabilities P(x_i=0|y):")
    for i in range(min(3, n_attributes)):
        print(f"  Attribute {i}: P(x_{i}=0|y=0)={qbc.cond_probs[0][i][0]:.3f}, "
              f"P(x_{i}=0|y=1)={qbc.cond_probs[1][i][0]:.3f}")

if __name__ == "__main__":
    main()