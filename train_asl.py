import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import json
import os

# Specify paths
existing_dataset = 'model/keypoint_classifier/keypoint.csv'
new_dataset = 'model/keypoint_classifier/keypoint_new.csv'
model_save_path = 'model/keypoint_classifier/keypoint_classifier.keras'
tflite_save_path = 'model/keypoint_classifier/keypoint_classifier.tflite'
npz_file = 'model/keypoint_classifier/keypoint_classifier_weights.npz'
NUM_CLASSES = 26

def main():
    print("=== ASL Model Training Pipeline ===")
    
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
                if len(parts) == 43: # Label + 42 features
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
    
    print(f"Loaded total of {len(X)} samples.")
    
    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    
    # 3. Build Model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((21 * 2, )),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='mish', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='mish', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation='mish', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    es_callback = tf.keras.callbacks.EarlyStopping(patience=15, verbose=1, restore_best_weights=True)
    
    # 4. Fit Model
    print("Training neural network...")
    model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=128,
        validation_data=(X_test, y_test),
        callbacks=[es_callback]
    )
    
    # Evaluate
    val_loss, val_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%")
    
    # 5. Save model
    model.save(model_save_path)
    print(f"Saved model to: {model_save_path}")
    
    # 6. Extract weights directly from Keras model
    print("Extracting weights directly from trained layers...")
    
    # Get layers weights
    bn_gamma, bn_beta, bn_mean, bn_var = model.layers[0].get_weights()
    w0, b0 = model.layers[1].get_weights()
    w1, b1 = model.layers[3].get_weights()
    w2, b2 = model.layers[5].get_weights()
    w3, b3 = model.layers[6].get_weights()
    
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
        w3=w3, b3=b3
    )
    print("Successfully saved weights to npz:", npz_file)
    
    # 7. Convert weights to JSON & Javascript bundles
    weights = np.load(npz_file)
    data = {k: weights[k].tolist() for k in weights.files}
    
    # Save model_weights.json
    with open('model/keypoint_classifier/model_weights.json', 'w') as f:
        json.dump(data, f)
        
    # Write weights.js for Web app & Chrome Extension
    content = f"// Auto-generated ASL Neural Network weights\nwindow.ASL_MODEL_WEIGHTS = {json.dumps(data)};\n"
    
    with open('web/js/weights.js', 'w', encoding='utf-8') as f:
        f.write(content)
        
    with open('chrome_extension/weights.js', 'w', encoding='utf-8') as f:
        f.write(content)
        
    print("Successfully updated web/js/weights.js and chrome_extension/weights.js with newly trained model weights!")

if __name__ == "__main__":
    main()
