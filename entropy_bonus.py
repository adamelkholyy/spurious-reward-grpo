"""verl-style entropy bonus for GRPO: loss = pg_loss - entropy_coeff * entropy, where entropy is the
per-token policy entropy aggregated the same way as the policy loss (token-mean over the whole
generation batch for loss_type="dapo", i.e. verl's loss_agg_mode="token-mean").

TRL computes per-token entropies under torch.no_grad() (they are only logged), so this swaps TRL's
entropy_from_logits for one that is differentiable while the training forward runs. Its backward
recomputes the softmax chunk by chunk from the saved logits (dH/dz = -p * (log p + H)), so no
full-vocab softmax is kept for backward.
"""
import torch
import torch.nn.functional as F
import trl.trainer.grpo_trainer as grpo_module

_trl_entropy_from_logits = grpo_module.entropy_from_logits
_differentiable = False  # set only around the training forward pass


class _EntropyFromLogits(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, chunk_size):
        with torch.no_grad():
            H = _trl_entropy_from_logits(logits, chunk_size)
        ctx.save_for_backward(logits, H)
        ctx.chunk_size = chunk_size
        return H

    @staticmethod
    def backward(ctx, grad_H):
        logits, H = ctx.saved_tensors
        V = logits.shape[-1]
        flat = logits.reshape(-1, V)
        grad = torch.empty_like(flat)
        H_flat, g_flat = H.reshape(-1, 1).float(), grad_H.reshape(-1, 1).float()
        for i in range(0, flat.shape[0], ctx.chunk_size):
            j = i + ctx.chunk_size
            logp = F.log_softmax(flat[i:j].float(), dim=-1)
            grad[i:j] = (-logp.exp() * (logp + H_flat[i:j]) * g_flat[i:j]).to(grad.dtype)
        return grad.view_as(logits), None


def _entropy_from_logits(logits, chunk_size=128):
    if _differentiable:
        with torch.enable_grad():  # TRL calls this inside torch.no_grad()
            return _EntropyFromLogits.apply(logits, chunk_size)
    return _trl_entropy_from_logits(logits, chunk_size)


grpo_module.entropy_from_logits = _entropy_from_logits


class EntropyBonusMixin:
    """Mix in before a GRPOTrainer subclass. Needs self.entropy_coeff > 0."""

    entropy_coeff = 0.0

    def _get_per_token_logps_and_entropies(self, *args, **kwargs):
        global _differentiable
        _differentiable = torch.is_grad_enabled() and self.model.training
        try:
            logps, entropies, aux_loss = super()._get_per_token_logps_and_entropies(*args, **kwargs)
        finally:
            _differentiable = False
        self._last_entropies = entropies
        return logps, entropies, aux_loss

    def _compute_loss(self, model, inputs):
        self._last_entropies = None
        loss = super()._compute_loss(model, inputs)
        H = self._last_entropies
        if H is None or not H.requires_grad:
            return loss
        mask = inputs["completion_mask"]
        if "tool_mask" in inputs:
            mask = mask * inputs["tool_mask"]
        normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
        entropy = (H * mask).sum() / normalizer  # same token-mean as the dapo policy loss
        if not getattr(self, "_entropy_bonus_logged", False):
            self._entropy_bonus_logged = True
            print(f"[entropy bonus] applied: coeff={self.entropy_coeff}, micro-batch term={entropy.item():.4f}, "
                  f"requires_grad={entropy.requires_grad}", flush=True)
        return loss - self.entropy_coeff * entropy


def with_entropy_bonus(trainer_cls, entropy_coeff):
    return type(f"Entropy{trainer_cls.__name__}", (EntropyBonusMixin, trainer_cls),
                {"entropy_coeff": float(entropy_coeff)})
