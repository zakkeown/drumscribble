import torch
from drumscribble.model.convnext import ConvNeXtBlock


class TestConvNeXtBlock:
    def test_output_shape_preserved(self):
        block = ConvNeXtBlock(dim=64, kernel_size=(1, 7))
        x = torch.randn(2, 64, 1, 625)
        out = block(x)
        assert out.shape == (2, 64, 1, 625)

    def test_residual_connection(self):
        block = ConvNeXtBlock(dim=64, kernel_size=(1, 7))
        x = torch.zeros(1, 64, 1, 100)
        out = block(x)
        assert out.shape == x.shape

    def test_kernel_11(self):
        block = ConvNeXtBlock(dim=256, kernel_size=(1, 11))
        x = torch.randn(2, 256, 1, 312)
        out = block(x)
        assert out.shape == (2, 256, 1, 312)

    def test_expand_ratio(self):
        block = ConvNeXtBlock(dim=64, kernel_size=(1, 7), expand_ratio=4)
        total = sum(p.numel() for p in block.parameters())
        assert total > 30000  # sanity check
