import torch
from drumscribble.model.convnext import ConvNeXtBlock, ConvNeXtBackbone


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


class TestConvNeXtBackbone:
    def test_output_and_skips(self):
        backbone = ConvNeXtBackbone()
        x = torch.randn(2, 1, 128, 624)  # T divisible by 8
        out, skips = backbone(x)
        assert out.shape == (2, 384, 1, 78)  # 624/8 = 78
        assert len(skips) == 3
        assert skips[0].shape == (2, 64, 1, 624)
        assert skips[1].shape == (2, 128, 1, 312)
        assert skips[2].shape == (2, 256, 1, 156)

    def test_output_shape_30s(self):
        backbone = ConvNeXtBackbone()
        x = torch.randn(1, 1, 128, 1872)  # divisible by 8
        out, skips = backbone(x)
        assert out.shape == (1, 384, 1, 234)  # 1872/8

    def test_param_count(self):
        backbone = ConvNeXtBackbone()
        total = sum(p.numel() for p in backbone.parameters())
        assert 8_000_000 < total < 12_000_000  # ~9.4M expected
