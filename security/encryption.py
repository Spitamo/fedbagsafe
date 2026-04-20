import numpy as np
import torch
from typing import Dict, Tuple, List

try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False


class HomomorphicEncryption:
    """HE utilities using CKKS scheme"""

    POLY_MODULUS_DEGREE = 8192

    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: tuple = (60, 40, 40, 60),
        global_scale: int = 2**40
    ):
        if not TENSEAL_AVAILABLE:
            raise RuntimeError(
                "TenSEAL is not installed. "
                "Install it with: pip install tenseal"
            )

        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        self.global_scale = global_scale

        self.private_ctx = self._build_private_context()
        self.public_ctx = self._build_public_context()
        self.slot_count = poly_modulus_degree // 2

    def _build_private_context(self):
        """build private CKKS context"""
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=list(self.coeff_mod_bit_sizes),
        )
        ctx.global_scale = self.global_scale
        ctx.generate_galois_keys()
        ctx.generate_relin_keys()
        return ctx

    def _build_public_context(self):
        """build public CKKS context"""
        serialized = self.private_ctx.serialize(save_secret_key=False)
        public_ctx = ts.context_from(serialized)
        public_ctx.make_context_public()
        return public_ctx

    def encrypt_model(
        self,
        model,
        layers_to_aggregate: List[str]
    ) -> Tuple[Dict, Dict]:
        """Enc model parameters"""
        enc_model = {}
        shapes = {}
        sd = model.state_dict()

        for key in layers_to_aggregate:
            flat = sd[key].detach().cpu().float().contiguous().view(
                -1
            ).numpy().tolist()
            chunks = [
                ts.ckks_vector(self.public_ctx, flat[i:i + self.slot_count])
                for i in range(0, len(flat), self.slot_count)
            ]
            enc_model[key] = chunks
            shapes[key] = tuple(sd[key].shape)

        return enc_model, shapes

    def decrypt_model(
        self,
        enc_acc: Dict,
        num_clients: int
    ) -> Dict[str, torch.Tensor]:
        """Dec model parameters"""
        decrypted = {}

        for key, payload in enc_acc.items():
            vals = []
            chunks = payload.get("chunks", [])
            
            for chunk in chunks:
                # check if chunk is already decrypted (string) or encrypted
                if isinstance(chunk, str):
                    # already serialized, deserialize first
                    ch_priv = ts.ckks_vector_from(self.private_ctx, chunk)
                    vals.extend(ch_priv.decrypt())
                else:
                    # still encrypted, serialize then deserialize
                    blob = chunk.serialize()
                    ch_priv = ts.ckks_vector_from(self.private_ctx, blob)
                    vals.extend(ch_priv.decrypt())

            shape = payload.get("shape", None)
            if shape is None:
                raise ValueError(f"Missing shape for key {key}")
            
            needed = int(np.prod(shape))
            vals = vals[:needed]

            tensor = torch.tensor(vals, dtype=torch.float32) / num_clients
            decrypted[key] = tensor.view(*shape)

        return decrypted