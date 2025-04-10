import torch

from siteflow.model.module import UNet2D, UNet3D

# 	                    2D	            3D
# single-channel	(1, Y, X)	    (Z, Y, X)
# multi-channel	    (C, 1, Y, X)	(C, Z, Y, X)


class TestModel:
    def test_unet2d(self):
        model = UNet2D(1, 1, f_maps=16, final_sigmoid=True)
        model.eval()
        x = torch.rand(1, 1, 65, 65)
        with torch.no_grad():
            y = model(x)

        assert torch.all(0 <= y) and torch.all(y <= 1)

    def test_unet3d(self):
        model = UNet3D(1, 1, f_maps=16, final_sigmoid=True)
        model.eval()
        x = torch.rand(1, 1, 33, 65, 65)
        with torch.no_grad():
            y = model(x)

        assert torch.all(0 <= y) and torch.all(y <= 1)
