import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import json
import os

# Specify paths
existing_dataset = 'model/keypoint_classifier/keypoint.csv'
new_dataset = 'model/keypoint_classifier/keypoint_new.csv'
model_save_path = 'model/keypoint_classifier/keypoint_classifier.keras'
tflite_save_path = 'model/keypoint_classifier/keypoint_classifier.tflite'
npz_file = 'model/keypoint_classifier/keypoint_classifier_weights.npz'
NUM_CLASSES = 26


def augment_landmarks(landmarks_batch):
    """
    Augments 2D hand landmark coordinates with realistic perturbations:
    1. Small random 2D rotation (+-12 degrees)
    2. Subtle scaling / stretch (0.92 to 1.08)
    3. Gaussian coordinate jitter to simulate sensor noise
    4. Max-abs re-normalization
    """
    N = len(landmarks_batch)
    points = landmarks_batch.reshape((N, 21, 2)).copy()

    # 1. Random 2D Rotation
    angles = np.random.uniform(-np.radians(12), np.radians(12), size=N)
    cos_a = np.cos(angles)[:, None, None]
    sin_a = np.sin(angles)[:, None, None]

    x = points[:, :, 0:1]
    y = points[:, :, 1:2]

    rot_x = x * cos_a - y * sin_a
    rot_y = x * sin_a + y * cos_a
    points = np.concatenate([rot_x, rot_y], axis=2)

    # 2. Random Scale
    scale = np.random.uniform(0.92, 1.08, size=(N, 1, 2))
    points = points * scale

    # 3. Random Gaussian jitter
    jitter = np.random.normal(0, 0.012, size=points.shape)
    points = points + jitter

    # 4. Max-abs re-normalization
    flattened = points.reshape((N, 42))
    max_vals = np.max(np.abs(flattened), axis=1, keepdims=True)
    max_vals[max_vals == 0] = 1.0
    return (flattened / max_vals).astype(np.float32)


def main():
    print("=== ASL High-Accuracy Model Training Pipeline ===")

    # 1. Load and merge datasets
    X, y = [], []

    datasets_to_load = []
    if os.path.exists(existing_dataset):
        datasets_to_load.append(existing_dataset)
    if os.path.exists(new_dataset):
        datasets_to_load.append(new_dataset)

    for path in datasets_to_load:
        print(f"Loading dataset: {path}")
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 43:  # Label + 42 features
                    try:
                        lbl = int(parts[0])
                        if 0 <= lbl < NUM_CLASSES:
                            feats = [float(x) for x in parts[1:]]
                            X.append(feats)
                            y.append(lbl)
                    except ValueError:
                        continue

    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int32')

    print(f"Loaded total of {len(X)} raw samples.")

    # 2. Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42, stratify=y
    )

    # 3. Apply Data Augmentation on Training Set
    print("Applying landmark data augmentation (rotation, scale, jitter)...")
    X_aug = augment_landmarks(X_train)
    X_train_combined = np.vstack([X_train, X_aug])
    y_train_combined = np.concatenate([y_train, y_train])

    # 4. Balanced Class Weighting
    class_weights = compute_class_weight(
        'balanced', classes=np.unique(y_train_combined), y=y_train_combined
    )
    class_weight_dict = dict(enumerate(class_weights))

    print(f"Training samples: {len(X_train_combined)}, Validation samples: {len(X_test)}")

    # 5. Build Deep High-Capacity MLP Model
    inputs = tf.keras.layers.Input(shape=(42,))
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Dense(256, activation='mish')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(256, activation='mish')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation='mish')(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    x = tf.keras.layers.Dense(64, activation='mish')(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.5, patience=5, min_lr=1e-5, verbose=1
    )
    es_callback = tf.keras.callbacks.EarlyStopping(
        patience=15, verbose=1, restore_best_weights=True
    )

    # 6. Fit Model
    print("Training neural network...")
    model.fit(
        X_train_combined,
        y_train_combined,
        epochs=70,
        batch_size=128,
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
        callbacks=[lr_reducer, es_callback],
        verbose=1
    )

    # Evaluate
    val_loss, val_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n>>> Final Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}% <<<")

    # 7. Save model
    model.save(model_save_path)
    print(f"Saved model to: {model_save_path}")

    # 8. Extract weights directly from Keras model
    print("Extracting weights directly from trained layers...")

    bn_gamma, bn_beta, bn_mean, bn_var = model.layers[1].get_weights()
    w0, b0 = model.layers[2].get_weights()
    w1, b1 = model.layers[4].get_weights()
    w2, b2 = model.layers[6].get_weights()
    w3, b3 = model.layers[8].get_weights()
    w4, b4 = model.layers[9].get_weights()

    # Save npz file
    np.savez(
        npz_file,
        bn_gamma=bn_gamma,
        bn_beta=bn_beta,
        bn_mean=bn_mean,
        bn_var=bn_var,
        w0=w0, b0=b0,
        w1=w1, b1=b1,
        w2=w2, b2=b2,
        w3=w3, b3=b3,
        w4=w4, b4=b4
    )
    print("Successfully saved weights to npz:", npz_file)

    # 9. Convert weights to JSON & Javascript bundles
    weights = np.load(npz_file)
    data = {k: weights[k].tolist() for k in weights.files}

    # Save model_weights.json
    with open('model/keypoint_classifier/model_weights.json', 'w') as f:
        json.dump(data, f)

    # Write weights.js for Web app & Chrome Extension
    js_json = json.dumps(data)
    content = f"""// Auto-generated ASL Neural Network weights
(function (root, factory) {{
    var w = factory();
    if (typeof module === 'object' && module.exports) {{
        module.exports = w;
    }}
    if (typeof window !== 'undefined') window.ASL_MODEL_WEIGHTS = w;
    if (typeof globalThis !== 'undefined') globalThis.ASL_MODEL_WEIGHTS = w;
    if (typeof global !== 'undefined') global.ASL_MODEL_WEIGHTS = w;
}})(typeof self !== 'undefined' ? self : this, function () {{
    return {js_json};
}});
"""

    with open('web/js/weights.js', 'w', encoding='utf-8') as f:
        f.write(content)

    with open('chrome_extension/weights.js', 'w', encoding='utf-8') as f:
        f.write(content)

    print("Successfully updated web/js/weights.js and chrome_extension/weights.js with newly trained model weights!")


if __name__ == "__main__":
    main()

