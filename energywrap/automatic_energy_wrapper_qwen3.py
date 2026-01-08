import torch
import torch.nn as nn
from typing import Tuple, List, Callable, Optional, Union


class EnergyWrapper(nn.Module):
    """Wraps a model to apply energy dynamics to hidden neurons."""

    def __init__(self, model: nn.Module, output_layers: List[nn.Module], delta: float = None, gamma: float = 0.05,
                 lambda_kl: float = 0.01, lambda_l1: float = 0.005, beta: float = 0.05,
                 hidden_layer_types: Optional[Tuple[type, ...]] = None,
                 hidden_layer_filter: Optional[Callable[[nn.Module], bool]] = None):
        """Initialize EnergyWrapper for Qwen3-VL, identifying hidden layers and setting energy parameters.

        Args:
            model: The neural network to wrap (Qwen3VL model).
            output_layers: List of modules producing final outputs (excluded from hidden layers).
            delta: Energy increment per step (default: 1/hidden_dim to prevent self-extinguishing).
            gamma: Energy drain factor, range [0.01, 0.5] for stability.
            lambda_kl: KL-divergence weight for sparsity, range [0.001, 0.1].
            lambda_l1: L1 regularization weight for sparsity, range [0.001, 0.1].
            beta: Target sparsity (~5% active neurons).
            hidden_layer_types: Tuple of module types to consider as hidden layers.
            hidden_layer_filter: Custom function to filter hidden layers (overrides hidden_layer_types).
        """
        super().__init__()
        self.model = model
        self.output_layers = output_layers
        self.gamma = gamma
        self.lambda_kl = lambda_kl
        self.lambda_l1 = lambda_l1
        self.beta = beta

        # Validate output_layers
        module_ids = {id(m) for m in model.modules()}
        for module in output_layers:
            if id(module) not in module_ids:
                raise ValueError(f"Output layer {module} not found in model")

        # For Qwen3-VL, we want to wrap:
        # - Linear layers (MLPs, projections)
        # - Conv2d/Conv3d (vision patch embeddings)
        # - LayerNorm/RMSNorm (normalization neurons)
        self.hidden_layer_types = hidden_layer_types or (
            nn.Linear, nn.Conv2d, nn.Conv3d, nn.LayerNorm,
            type(model.language_model.norm) if hasattr(model, 'language_model') else nn.Module
        )
        self.hidden_layer_filter = hidden_layer_filter or (lambda m: isinstance(m, self.hidden_layer_types))

        # Identify hidden layers
        self.hidden_layers = []
        self.hidden_dims = []
        self.energy_buffers = []
        self.hooks = []
        self.kv_attention_wrapped = False  # Track if we've wrapped KV attention

        for name, module in model.named_modules():
            if self.hidden_layer_filter(module) and module not in output_layers:
                self.hidden_layers.append((name, module))
                hidden_dim = self._get_hidden_dim(module)
                self.hidden_dims.append(hidden_dim)
                self.energy_buffers.append(
                    self.register_buffer(f'energy_{name.replace(".", "_")}', torch.zeros(hidden_dim)))

        if not self.hidden_layers:
            raise ValueError("No hidden layers found in model")

        # Delta per layer
        self.deltas = [1.0 / dim if delta is None else delta for dim in self.hidden_dims]
        for i, (delta, dim) in enumerate(zip(self.deltas, self.hidden_dims)):
            assert delta * dim >= 1, f"delta * hidden_dim < 1 for layer {i}"

        # Assertions for hyperparameter ranges
        assert 0.01 <= gamma <= 0.5, "gamma must be in [0.01, 0.5]"
        assert 0.001 <= lambda_kl <= 0.1, "lambda_kl must be in [0.001, 0.1]"
        assert 0.001 <= lambda_l1 <= 0.1, "lambda_l1 must be in [0.001, 0.1]"

        self.aux_loss = torch.tensor(0.0)

        # Register forward hooks
        for idx, (name, module) in enumerate(self.hidden_layers):
            handle = module.register_forward_hook(self._make_hook(idx, name))
            self.hooks.append(handle)

        # Add special hook for attention KV after softmax (wrap once)
        self._add_kv_attention_hook(model)

    def _add_kv_attention_hook(self, model):
        """Add hook to wrap key-value calculation after softmax in attention modules."""
        # Find attention modules in Qwen3-VL
        for name, module in model.named_modules():
            # Check for both vision and text attention
            if 'Qwen3VLVisionAttention' in str(type(module)) or 'Qwen3VLTextAttention' in str(type(module)):
                # Register hook to intercept attention computation
                handle = module.register_forward_hook(self._make_kv_attention_hook(name))
                self.hooks.append(handle)
                self.kv_attention_wrapped = True

    def _make_kv_attention_hook(self, attn_name: str):
        """Create hook for attention KV computation after softmax."""

        def hook(module, input, output):
            # For Qwen3VL, we need to intercept the value states after attention weights are applied
            # The attention output is already computed, but we can wrap it as "KV neurons"
            if isinstance(output, tuple):
                attn_output = output[0]
            else:
                attn_output = output

            # Treat attention output as neurons to wrap
            original_shape = attn_output.shape
            batch_size = original_shape[0]

            # Flatten to [batch, neurons]
            h = attn_output.reshape(batch_size, -1)
            hidden_dim = h.shape[-1]

            # Create energy buffer if not exists
            energy_key = f'energy_kv_{attn_name.replace(".", "_")}'
            if not hasattr(self, energy_key):
                self.register_buffer(energy_key, torch.zeros(hidden_dim, device=h.device))

            energy = getattr(self, energy_key)

            # Apply energy dynamics
            a = h.clamp(min=0)

            # KL-divergence
            rho = a.gt(0).float().mean(dim=0)
            rho = torch.clamp(rho, 1e-5, 1 - 1e-5)
            kl_loss = self.lambda_kl * (rho * torch.log(rho / self.beta) +
                                        (1 - rho) * torch.log((1 - rho) / (1 - self.beta)))

            # Energy update with smaller delta for attention (more stable)
            delta_kv = 0.5 / hidden_dim
            noise = torch.normal(mean=0.0, std=0.001, size=(hidden_dim,), device=energy.device)
            new_energy = energy.detach() + delta_kv + noise - self.gamma * a.mean(dim=0)

            # Energy penalties
            high_mask = new_energy > 1
            low_mask = new_energy < -1
            if high_mask.any() or low_mask.any():
                self.aux_loss += 0.01 * (new_energy[high_mask].abs() - 1).sum()
                self.aux_loss += 0.01 * (new_energy[low_mask].abs() - 1).sum()

            # Energy thresholds
            fire_mask = new_energy >= 2
            shutoff_mask = new_energy <= -2
            if fire_mask.any():
                h = h.clone()
                h[:, fire_mask] = 2.0
            if shutoff_mask.any():
                h = h.clone()
                h[:, shutoff_mask] = (energy.detach() + 2)[shutoff_mask]
                new_energy[shutoff_mask] = -2

            setattr(self, energy_key, new_energy)
            self.aux_loss += kl_loss.mean()

            # Restore original shape
            h = h.reshape(original_shape)

            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h

        return hook

    def _get_hidden_dim(self, module: nn.Module) -> int:
        """Infer hidden dimension from module."""
        if isinstance(module, nn.Linear):
            return module.out_features
        elif isinstance(module, (nn.RNNCell, nn.LSTM, nn.GRU)):
            return module.hidden_size
        elif isinstance(module, (nn.Conv2d, nn.Conv3d)):
            return module.out_channels
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            return module.normalized_shape[0] if isinstance(module.normalized_shape, tuple) else module.normalized_shape
        elif hasattr(module, 'weight') and module.weight is not None:
            # For RMSNorm and similar
            return module.weight.numel()
        raise NotImplementedError(f"Cannot infer hidden_dim for module type: {type(module)}")

    def process_hidden_output(self, module: nn.Module,
                              output: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        """Extract hidden neuron activations from module output."""
        if isinstance(module, nn.RNNCell):
            return output
        elif isinstance(module, nn.LSTM):
            return output[0]
        elif isinstance(module, nn.GRU):
            return output
        elif isinstance(module, nn.Linear):
            return output
        elif isinstance(module, (nn.Conv2d, nn.Conv3d)):
            return output.flatten(start_dim=1)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            return output
        elif hasattr(module, 'weight'):  # RMSNorm
            return output
        raise NotImplementedError(f"Unsupported module type: {type(module)}")

    def restore_output_shape(self, module: nn.Module, h: torch.Tensor,
                             original_output: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> Union[
        torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Restore modified hidden output to original shape."""
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            batch_size = original_output.size(0)
            if isinstance(module, nn.Conv2d):
                return h.view(batch_size, module.out_channels, *original_output.shape[2:])
            else:  # Conv3d
                return h.view(batch_size, module.out_channels, *original_output.shape[2:])
        elif isinstance(module, nn.LSTM):
            return (h, original_output[1])
        return h

    def _make_hook(self, layer_idx: int, layer_name: str):
        """Create forward hook to apply energy dynamics to hidden layer output."""

        def hook(module, input, output):
            h = self.process_hidden_output(module, output)
            original_shape = h.shape

            # Flatten to [batch, neurons] if needed
            if h.dim() > 2:
                batch_size = h.shape[0]
                h = h.reshape(batch_size, -1)

            a = h.clamp(min=0)

            # KL-divergence
            rho = a.gt(0).float().mean(dim=1).mean()
            rho = torch.clamp(rho, 1e-5, 1 - 1e-5)
            kl_loss = self.lambda_kl * (rho * torch.log(rho / self.beta) +
                                        (1 - rho) * torch.log((1 - rho) / (1 - self.beta)))

            # L1 sparsity
            l1_loss = 0.0
            for param in module.parameters():
                if param.ndim >= 2:
                    l1_loss += self.lambda_l1 * param.abs().sum()

            # Energy update
            energy = getattr(self, f'energy_{layer_name.replace(".", "_")}')
            noise = torch.normal(mean=0.0, std=0.001, size=(self.hidden_dims[layer_idx],),
                                 device=energy.device)
            new_energy = energy.detach() + self.deltas[layer_idx] + noise - self.gamma * a.mean(dim=0)

            # Energy penalties
            high_mask = new_energy > 1
            low_mask = new_energy < -1
            if high_mask.any() or low_mask.any():
                self.aux_loss += 0.01 * (new_energy[high_mask].abs() - 1).sum()
                self.aux_loss += 0.01 * (new_energy[low_mask].abs() - 1).sum()

            # Energy thresholds
            fire_mask = new_energy >= 2
            shutoff_mask = new_energy <= -2
            if fire_mask.any():
                h = h.clone()
                h[:, fire_mask] = 2.0
            if shutoff_mask.any():
                h = h.clone()
                h[:, shutoff_mask] = (energy.detach() + 2)[shutoff_mask]
                new_energy[shutoff_mask] = -2

            setattr(self, f'energy_{layer_name.replace(".", "_")}', new_energy)
            self.aux_loss += kl_loss + l1_loss

            # Restore original shape
            h = h.reshape(original_shape)

            return self.restore_output_shape(module, h, output)

        return hook

    def forward(self, *args, **kwargs):
        """Forward pass through wrapped model, applying energy dynamics via hooks."""
        self.aux_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        output = self.model(*args, **kwargs)
        return output

    def __getattr__(self, name: str):
        """Delegate attribute access to wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def wrap_qwen3vl_energy_dynamics(model: nn.Module, output_layers: List[nn.Module], delta: float = None,
                                 gamma: float = 0.05, lambda_kl: float = 0.01, lambda_l1: float = 0.005,
                                 beta: float = 0.05) -> nn.Module:
    """Wrap a Qwen3-VL model to add energy dynamics to its hidden neurons.

    This wraps all neurons including:
    - Linear layers (MLPs, projections)
    - Convolutional layers (vision encoders)
    - Normalization layers (LayerNorm, RMSNorm)
    - Key-Value computation after softmax in attention (wrapped once)
    """
    return EnergyWrapper(model, output_layers, delta, gamma, lambda_kl, lambda_l1, beta)