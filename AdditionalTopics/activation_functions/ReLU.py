import numpy as np

class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, x):
        """
        Berechnet die ReLU-Funktion: f(x) = max(0, x).
        Speichert den Input für den Backward-Pass.
        """
        self.input = x
        return np.maximum(0, x)

    def backward(self, output_gradient):
        """
        Berechnet die Ableitung der ReLU-Funktion.
        Wenn x > 0, ist die Ableitung 1.
        Wenn x <= 0, ist die Ableitung 0.
        """
        if self.input is None:
            raise RuntimeError("Forward muss vor Backward aufgerufen werden.")
        
        # Kopiere den Gradienten von der nächsten Schicht
        input_gradient = output_gradient.copy()
        
        # Setze den Gradienten auf 0, wo der Input <= 0 war
        input_gradient[self.input <= 0] = 0
        
        return input_gradient
