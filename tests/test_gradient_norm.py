import torch
import torch.nn as nn
from visdom import compute_gradient_l2_norm

def test_compute_gradient_l2_norm_returns_float():
    model = nn.Linear(5, 1)
    x = torch.randn(3, 5)
    y = torch.randn(3, 1)

    criterion = nn.MSELoss()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()

    norm = compute_gradient_l2_norm(model)

    assert isinstance(norm, float)
    assert norm > 0