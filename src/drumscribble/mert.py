"""MERT-95M feature extraction (optional, requires transformers)."""

import torch
import torch.nn as nn


class MERTExtractor(nn.Module):
    """Extract features from MERT-95M at specified layers.

    Uses layers 5-6 per design Revision 2.  Requires the ``transformers``
    package (install via ``pip install drumscribble[mert]``).
    """

    def __init__(
        self,
        model_name: str = "m-a-p/MERT-v1-95M",
        layer_indices: list[int] | None = None,
    ):
        super().__init__()
        from transformers import AutoModel, AutoFeatureExtractor

        self.layer_indices = layer_indices or [5, 6]
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoFeatureExtractor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract MERT features.

        Args:
            waveform: (B, samples) at 16kHz.

        Returns:
            (B, 768, 1, T_mert) features in ANE 4D format.
        """
        outputs = self.model(waveform, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        # Average selected layers
        selected = [hidden_states[i] for i in self.layer_indices]
        features = torch.stack(selected).mean(dim=0)  # (B, T_mert, 768)

        # Reshape to ANE format: (B, C, 1, T)
        features = features.permute(0, 2, 1).unsqueeze(2)

        return features
