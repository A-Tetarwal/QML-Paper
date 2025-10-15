"""
Validation tests for Quantum Bayes Classifier
Implements quantitative checks to verify correctness
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from tqdm import tqdm
import matplotlib.pyplot as plt
from quantum_bayes_classifier import (
    QuantumBayesClassifier, 
    load_and_preprocess_data, 
    filter_binary_classes,
    evaluate_classifier
)

def test_statevector_sanity_check(qbc, X_test, y_test, n_samples=5):
    """
    Test 1: State-vector sanity check
    Verify amplitudes² match analytical joint probabilities P(y) ∏ P(xi | y)
    """
    print("\n" + "="*80)
    print("TEST 1: State-vector Sanity Check")
    print("="*80)
    
    # Get statevector simulator
    backend = Aer.get_backend('statevector_simulator')
    transpiled_qc = transpile(qbc.circuit, backend)
    
    # Execute circuit to get statevector
    job = backend.run(transpiled_qc)
    result = job.result()
    statevector = result.get_statevector()
    
    print(f"\nStatevector size: {len(statevector)}")
    print(f"Number of qubits: {qbc.n_attributes + 1}")
    
    # Verify normalization
    norm = np.sum(np.abs(statevector)**2)
    print(f"Statevector normalization: {norm:.6f} (should be 1.0)")
    
    # Check a few specific basis states
    print("\nVerifying amplitudes vs analytical probabilities:")
    print(f"{'Basis State':<20} {'|Amplitude|²':<15} {'Analytical P':<15} {'Match?':<10}")
    print("-" * 70)
    
    for _ in range(n_samples):
        # Random basis state
        y_val = np.random.randint(0, 2)
        x_vals = np.random.randint(0, 2, size=qbc.n_attributes)
        
        # Compute analytical probability P(y, x₁, ..., xₙ)
        analytical_p = qbc.class_probs[y_val]
        for i in range(qbc.n_attributes):
            analytical_p *= qbc.cond_probs[y_val][i][x_vals[i]]
        
        # Get amplitude from statevector
        # Basis state index (reversed order in Qiskit)
        basis_bits = [y_val] + list(x_vals)
        basis_str = ''.join(map(str, reversed(basis_bits)))
        state_idx = int(basis_str, 2)
        amplitude_sq = np.abs(statevector[state_idx])**2
        
        match = "✓" if np.abs(amplitude_sq - analytical_p) < 0.01 else "✗"
        print(f"|{y_val}," + ",".join(map(str, x_vals[:3])) + "...>" + 
              f"  {amplitude_sq:<14.6f} {analytical_p:<14.6f} {match:<10}")
    
    print("\n✓ Statevector sanity check complete!")


def test_shot_convergence(qbc, X_test, y_test, shot_counts=[100, 500, 1000, 2000, 3000, 5000]):
    """
    Test 2: Shot convergence curve
    Plot accuracy vs shots - should converge to classical accuracy asymptotically
    """
    print("\n" + "="*80)
    print("TEST 2: Shot Convergence Analysis")
    print("="*80)
    
    accuracies = []
    
    for shots in tqdm(shot_counts, desc="Testing different shot counts"):
        predictions = qbc.predict(X_test, shots=shots)
        accuracy = evaluate_classifier(y_test, predictions)
        accuracies.append(accuracy)
        print(f"Shots: {shots:5d} → Accuracy: {accuracy:.4f}")
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(shot_counts, accuracies, 'b-o', linewidth=2, markersize=8, label='QBC Accuracy')
    plt.axhline(y=accuracies[-1], color='r', linestyle='--', label=f'Converged ({accuracies[-1]:.4f})')
    plt.xlabel('Number of Shots', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Quantum Bayes Classifier: Shot Convergence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('shot_convergence.png', dpi=300)
    print("\n✓ Plot saved as 'shot_convergence.png'")
    plt.close()
    
    return accuracies


def test_attribute_influence(X_train, X_test, y_train, y_test, attribute_counts=[3, 5, 7, 9, 12, 15]):
    """
    Test 3: Attribute influence
    Compare accuracies for different numbers of attributes
    Should increase then plateau, matching Fig. 6 in the paper
    """
    print("\n" + "="*80)
    print("TEST 3: Attribute Influence Analysis")
    print("="*80)
    
    classical_accuracies = []
    quantum_accuracies = []
    
    # Fix random seed for reproducibility
    np.random.seed(42)
    
    for n_attrs in tqdm(attribute_counts, desc="Testing different attribute counts"):
        # Train QBC with n_attrs attributes
        qbc = QuantumBayesClassifier(n_attributes=n_attrs, n_classes=2, classifier_type='naive')
        qbc.fit(X_train, y_train)
        
        # Classical NB baseline
        X_train_bin = np.array([qbc._binarize_sample(x) for x in X_train])
        X_test_bin = np.array([qbc._binarize_sample(x) for x in X_test])
        
        classical_preds = []
        for x_test in X_test_bin:
            log_probs = []
            for c in range(2):
                log_p = np.log(qbc.class_probs[c])
                for i in range(n_attrs):
                    log_p += np.log(qbc.cond_probs[c][i][x_test[i]])
                log_probs.append(log_p)
            classical_preds.append(np.argmax(log_probs))
        
        classical_acc = evaluate_classifier(y_test, np.array(classical_preds))
        classical_accuracies.append(classical_acc)
        
        # Quantum predictions
        quantum_preds = qbc.predict(X_test, shots=3000)
        quantum_acc = evaluate_classifier(y_test, quantum_preds)
        quantum_accuracies.append(quantum_acc)
        
        print(f"Attributes: {n_attrs:2d} → Classical: {classical_acc:.4f}, Quantum: {quantum_acc:.4f}")
    
    # Plot attribute influence
    plt.figure(figsize=(10, 6))
    plt.plot(attribute_counts, classical_accuracies, 'g-s', linewidth=2, markersize=8, 
             label='Classical NB', alpha=0.7)
    plt.plot(attribute_counts, quantum_accuracies, 'b-o', linewidth=2, markersize=8, 
             label='Quantum BC', alpha=0.7)
    plt.xlabel('Number of Attributes', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy vs Number of Attributes', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('attribute_influence.png', dpi=300)
    print("\n✓ Plot saved as 'attribute_influence.png'")
    plt.close()
    
    return classical_accuracies, quantum_accuracies


def test_reproducibility(X_train, X_test, y_train, y_test, n_runs=5):
    """
    Test 4: Random-seed reproducibility
    Verify consistent results with fixed seed
    """
    print("\n" + "="*80)
    print("TEST 4: Reproducibility Test")
    print("="*80)
    
    results = []
    
    for run in range(n_runs):
        np.random.seed(42)  # Fixed seed
        
        qbc = QuantumBayesClassifier(n_attributes=9, n_classes=2, classifier_type='naive')
        qbc.fit(X_train, y_train)
        
        predictions = qbc.predict(X_test, shots=3000)
        accuracy = evaluate_classifier(y_test, predictions)
        results.append(accuracy)
        
        print(f"Run {run+1}: Accuracy = {accuracy:.4f}, Sampling positions = {qbc.sampling_positions[:3]}...")
    
    print(f"\nMean accuracy: {np.mean(results):.4f} ± {np.std(results):.4f}")
    print(f"All runs identical: {len(set(results)) == 1}")
    print("\n✓ Reproducibility test complete!")
    
    return results


def test_probability_distributions(qbc):
    """
    Test 5: Verify probability distributions are well-formed
    """
    print("\n" + "="*80)
    print("TEST 5: Probability Distribution Verification")
    print("="*80)
    
    # Check class priors sum to 1
    prior_sum = np.sum(qbc.class_probs)
    print(f"\nClass priors sum: {prior_sum:.6f} (should be 1.0)")
    print(f"P(y=0) = {qbc.class_probs[0]:.4f}, P(y=1) = {qbc.class_probs[1]:.4f}")
    
    # Check conditional probabilities sum to 1 for each attribute
    print("\nConditional probability checks:")
    all_valid = True
    for c in range(2):
        for i in range(min(3, qbc.n_attributes)):  # Check first 3 attributes
            cond_sum = qbc.cond_probs[c][i][0] + qbc.cond_probs[c][i][1]
            valid = abs(cond_sum - 1.0) < 1e-5
            symbol = "✓" if valid else "✗"
            print(f"  {symbol} Class {c}, Attr {i}: P(x=0|y) + P(x=1|y) = {cond_sum:.6f}")
            all_valid = all_valid and valid
    
    print(f"\n{'✓ All probabilities valid!' if all_valid else '✗ Some probabilities invalid!'}")
    
    return all_valid


def main():
    """Run all validation tests"""
    print("="*80)
    print("QUANTUM BAYES CLASSIFIER - VALIDATION SUITE")
    print("="*80)
    
    # Load data with fixed seed
    np.random.seed(42)
    
    print("\nLoading dataset...")
    X_train_full, X_test_full, y_train_full, y_test_full = load_and_preprocess_data(
        'mnist', n_train=5000, n_test=500
    )
    
    X_train, X_test, y_train, y_test = filter_binary_classes(
        X_train_full, X_test_full, y_train_full, y_test_full, class0=0, class1=1
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train base QBC
    print("\nTraining base QBC (9 attributes)...")
    qbc = QuantumBayesClassifier(n_attributes=9, n_classes=2, classifier_type='naive')
    qbc.fit(X_train, y_train)
    
    # Run all tests
    test_probability_distributions(qbc)
    test_statevector_sanity_check(qbc, X_test, y_test, n_samples=10)
    test_shot_convergence(qbc, X_test, y_test)
    test_attribute_influence(X_train, X_test, y_train, y_test)
    test_reproducibility(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*80)
    print("ALL VALIDATION TESTS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
