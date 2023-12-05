import cv2
import numpy as np

class CLAHE:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def apply_clahe_to_batch(self, batch):
        clahe_output = np.empty_like(batch)
        for i in range(batch.shape[0]):
            for channel in range(batch.shape[3]):
                clahe_output[i, :, :, channel] = self.clahe.apply(batch[i, :, :, channel])
        return clahe_output

    def __call__(self, x):
        return self.apply_clahe_to_batch(x)



    
