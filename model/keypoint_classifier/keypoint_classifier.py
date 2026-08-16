import os
import numpy as np

class KeyPointClassifier(object):
    def __init__(
        self,
        model_path="model/keypoint_classifier/keypoint_classifier.tflite",
        num_threads=1,
    ):
        # Allow loading weights from npz file instead of tflite
        if model_path.endswith(".tflite"):
            model_path = model_path.replace(".tflite", "_weights.npz")
        
        # Load weights
        weights = np.load(model_path)
        self.bn_gamma = weights["bn_gamma"]
        self.bn_beta = weights["bn_beta"]
        self.bn_mean = weights["bn_mean"]
        self.bn_var = weights["bn_var"]
        
        self.w0 = weights["w0"]
        self.b0 = weights["b0"]
        self.w1 = weights["w1"]
        self.b1 = weights["b1"]
        self.w2 = weights["w2"]
        self.b2 = weights["b2"]
        self.w3 = weights["w3"]
        self.b3 = weights["b3"]

    def _softplus(self, x):
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    def _mish(self, x):
        return x * np.tanh(self._softplus(x))

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def __call__(
        self,
        landmark_list,
    ):
        x = np.array([landmark_list], dtype=np.float32)
        
        # 1. Batch Normalization (epsilon=0.001)
        x = (x - self.bn_mean) / np.sqrt(self.bn_var + 0.001) * self.bn_gamma + self.bn_beta
        
        # 2. Dense layer 0 + Mish
        x = x @ self.w0 + self.b0
        x = self._mish(x)
        
        # 3. Dense layer 1 + Mish
        x = x @ self.w1 + self.b1
        x = self._mish(x)
        
        # 4. Dense layer 2 + Mish
        x = x @ self.w2 + self.b2
        x = self._mish(x)
        
        # 5. Dense layer 3 + Softmax
        x = x @ self.w3 + self.b3
        result = self._softmax(x)
        
        result_index = np.argmax(np.squeeze(result))
        return result_index
