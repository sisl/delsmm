import torch
from delsmm.utils import bfill_uppertriangle, bfill_lowertriangle

def test():
    A = torch.zeros(4, 10, 3, 3)
    vec = torch.randn(4, 10, 3)
    B = bfill_uppertriangle(A, vec)
    B = bfill_lowertriangle(B, vec)
    assert torch.allclose(B, B.transpose(3, 2))

    A = torch.zeros(4, 10, 3, 3)
    vec = torch.randn(4, 10, 3)
    B = bfill_lowertriangle(A, vec)    
    B = bfill_uppertriangle(B, vec)
    assert torch.allclose(B, B.transpose(3, 2))

    

if __name__ == "__main__":
    test()