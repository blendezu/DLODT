import numpy as np
from activation_functions.ReLU import ReLU

def test_relu():
    relu = ReLU()

    # Test Forward
    x = np.array([-2, -1, 0, 1, 2])
    print(f"Input: {x}")
    out = relu.forward(x)
    print(f"Forward Output: {out}")
    
    expected_out = np.array([0, 0, 0, 1, 2])
    assert np.array_equal(out, expected_out), "Forward pass incorrect"

    # Test Backward
    # Use a dummy gradient coming from the next layer (e.g., all 1s)
    grad_output = np.array([1, 1, 1, 1, 1])
    grad_input = relu.backward(grad_output)
    print(f"Backward Output (Gradient): {grad_input}")

    expected_grad = np.array([0, 0, 0, 1, 1]) # Derivative is 0 for x<=0, 1 for x>0
    
    # Note: derivative at 0 is technically undefined, usually handled as 0 or 1. Our implementation makes it 0.
    assert np.array_equal(grad_input, expected_grad), "Backward pass incorrect"
    
    print("ReLU Test Passed!")

if __name__ == "__main__":
    test_relu()
 