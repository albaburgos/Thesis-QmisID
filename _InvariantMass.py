import torch
from iwpc.encodings.encoding_base import Encoding

# A Custom Dielectron encoding that appends the dielectron invariant mass as an extra feature, to test if it improves kernel performance.

class CustomDielectronEncoding(Encoding):
    """
    Wraps base encoding and appends the dielectron invariant mass as an extra feature.

    Input layout (8D): [l1_pt, l1_eta, l1_phi, l2_pt, l2_eta, l2_phi, flip1, flip2]

    Output: base_encoding(x) concatenated with [invmass] 
    """
    def __init__(self, base_encoding):
        input_dim  = int(base_encoding.input_shape[0])
        output_dim = int(base_encoding.output_shape[0]) + 1
        super().__init__(input_dim, output_dim)
        self.base_encoding = base_encoding

    def _encode(self, x):
        # x[:, 0] = l1_pt,  x[:, 1] = l1_eta,  x[:, 2] = l1_phi
        # x[:, 3] = l2_pt,  x[:, 4] = l2_eta,  x[:, 5] = l2_phi
        # x[:, 6] = flip1,  x[:, 7] = flip2
        base = self.base_encoding(x)
        invmass = torch.sqrt(torch.clamp(
            2 * x[:, 0].abs() * x[:, 3].abs()
            * (torch.cosh(x[:, 1] - x[:, 4]) - torch.cos(x[:, 2] - x[:, 5])),
            min=0.0,
        ))
        return torch.cat([base, invmass.unsqueeze(-1)], dim=-1)
