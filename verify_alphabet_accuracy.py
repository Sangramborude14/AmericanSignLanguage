import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

def main():
    print("=== ASL Model Alphabet Accuracy Verification ===")
    
    # Paths
    existing_dataset = 'model/keypoint_classifier/keypoint.csv'
    new_dataset = 'model/keypoint_classifier/keypoint_new.csv'
    npz_path = 'model/keypoint_classifier/keypoint_classifier_weights.npz'

    if not os.path.exists(npz_path):
        print(f"NPZ weights file {npz_path} not found.")
        return

    # 1. Load data
    X, y = [], []
    for path in [existing_dataset, new_dataset]:
        if os.path.exists(path):
            print(f"Reading dataset: {path}")
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 43:
                        try:
                            lbl = int(parts[0])
                            if 0 <= lbl < 26:
                                X.append([float(x) for x in parts[1:]])
                                y.append(lbl)
                        except ValueError:
                            continue

    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int32')
    print(f"Total samples loaded: {len(X)}")

    # Train-test split (using seed 42 to match the training process validation set)
    _, X_test, _, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    print(f"Test samples for validation: {len(X_test)}")

    # 2. Build model and load weights from npz
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((21 * 2, )),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='mish'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='mish'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation='mish'),
        tf.keras.layers.Dense(26, activation='softmax')
    ])
    
    # Force layer building
    model.predict(np.zeros((1, 42), dtype='float32'), verbose=0)
    
    # Load weights manually from npz
    weights = np.load(npz_path)
    model.layers[0].set_weights([weights['bn_gamma'], weights['bn_beta'], weights['bn_mean'], weights['bn_var']])
    model.layers[1].set_weights([weights['w0'], weights['b0']])
    model.layers[3].set_weights([weights['w1'], weights['b1']])
    model.layers[5].set_weights([weights['w2'], weights['b2']])
    model.layers[6].set_weights([weights['w3'], weights['b3']])

    # 3. Predict on test set
    predictions = model.predict(X_test, verbose=0)
    pred_indices = np.argmax(predictions, axis=1)

    # 4. Compute accuracy per class
    print("\n--- Accuracy Breakdown Per Alphabet (A-Z) ---")
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    class_correct = {char: 0 for char in alphabet}
    class_total = {char: 0 for char in alphabet}

    for true_label, pred_label in zip(y_test, pred_indices):
        char_true = alphabet[true_label]
        class_total[char_true] += 1
        if true_label == pred_label:
            class_correct[char_true] += 1

    overall_correct = 0
    overall_total = 0
    
    print(f"{'Letter':<10} | {'Correct':<10} | {'Total':<10} | {'Accuracy':<10}")
    print("-" * 50)
    for char in alphabet:
        correct = class_correct[char]
        total = class_total[char]
        accuracy = (correct / total * 100) if total > 0 else 0
        overall_correct += correct
        overall_total += total
        print(f"{char:<10} | {correct:<10} | {total:<10} | {accuracy:.2f}%")

    overall_accuracy = (overall_correct / overall_total * 100) if overall_total > 0 else 0
    print("-" * 50)
    print(f"{'OVERALL':<10} | {overall_correct:<10} | {overall_total:<10} | {overall_accuracy:.2f}%")

if __name__ == '__main__':
    main()
