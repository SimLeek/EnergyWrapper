import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Union, Any, Callable, Dict
from dataclasses import dataclass


def _infer_hidden_dim(module: nn.Module) -> Optional[int]:
    """Infer hidden dimension from common layer types."""
    if isinstance(module, (nn.Linear, nn.LazyLinear)):
        return module.out_features if hasattr(module, 'out_features') else None
    elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
        return module.hidden_size
    elif isinstance(module, nn.MultiheadAttention):
        return module.embed_dim
    elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return module.out_channels
    elif isinstance(module, nn.TransformerEncoderLayer):
        return module.linear1.out_features if hasattr(module, 'linear1') else None
    return None


def _apply_energy_dynamics(
        h: torch.Tensor,
        energy: torch.Tensor,
        delta: float,
        gamma: float,
        lambda_kl: float,
        beta: float,
        hidden_dim: int
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Apply energy dynamics to a tensor and return modified tensor, new energy, and aux loss."""

    # Handle missing batch dimension
    original_shape = h.shape
    if h.dim() == 1:
        h = h.unsqueeze(0)

    a = h.clamp(min=0)

    # KL sparsity - flatten all non-feature dimensions
    flat_a = a.reshape(-1, a.shape[-1])
    rho = flat_a.gt(0).float().mean(dim=0).mean()
    rho = torch.clamp(rho, 1e-5, 1 - 1e-5)
    kl_loss = lambda_kl * (
            rho * torch.log(rho / beta) +
            (1 - rho) * torch.log((1 - rho) / (1 - beta))
    )

    # Energy update
    noise = torch.normal(mean=0.0, std=0.001, size=(hidden_dim,), device=energy.device)
    new_energy = energy.detach() + delta + noise - gamma * flat_a.mean(dim=0)

    # Energy penalties
    aux_loss = kl_loss.item()
    high_mask = new_energy > 1
    low_mask = new_energy < -1
    if high_mask.any():
        aux_loss += 0.01 * (new_energy[high_mask].abs() - 1).sum().item()
    if low_mask.any():
        aux_loss += 0.01 * (new_energy[low_mask].abs() - 1).sum().item()

    # Apply thresholds
    fire_mask = new_energy >= 2
    shutoff_mask = new_energy <= -2
    if fire_mask.any() or shutoff_mask.any():
        h = h.clone()
        if fire_mask.any():
            h[..., fire_mask] = 2.0
        if shutoff_mask.any():
            h[..., shutoff_mask] = (energy.detach() + 2)[shutoff_mask]
            new_energy[shutoff_mask] = -2

    # Restore original shape if needed
    if len(original_shape) == 1:
        h = h.squeeze(0)

    return h, new_energy, aux_loss


class EnergyHookLayer(nn.Module):
    """A dummy layer marking where to apply energy dynamics."""

    def __init__(self,
                 hidden_dim: Optional[Union[int, Tuple[int, ...]]] = None,
                 delta: Optional[float] = None,
                 gamma: float = 0.05,
                 lambda_kl: float = 0.01,
                 beta: float = 0.05,
                 output_selector: Optional[Callable] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.delta = delta
        self.gamma = gamma
        self.lambda_kl = lambda_kl
        self.beta = beta
        self.output_selector = output_selector or (lambda x: x)

        self.energy = None
        self.aux_loss = 0.0

    def forward(self, x: Any) -> Any:
        self.aux_loss = 0.0

        # Extract tensor to process
        try:
            h = self.output_selector(x)
            if not isinstance(h, torch.Tensor):
                return x
        except (TypeError, IndexError, KeyError):
            return x

        # Infer hidden_dim if not provided
        if self.hidden_dim is None:
            self.hidden_dim = h.shape[-1]

        # Initialize energy buffer on first forward pass
        if self.energy is None:
            self.energy = torch.zeros(self.hidden_dim, device=h.device)

        # Handle shape compatibility
        if h.shape[-1] != self.hidden_dim:
            compatible_dim = None
            for dim_idx in range(len(h.shape)):
                if h.shape[dim_idx] == self.hidden_dim:
                    compatible_dim = dim_idx
                    break

            if compatible_dim is None:
                raise ValueError(
                    f"Cannot find dimension matching hidden_dim={self.hidden_dim} "
                    f"in tensor with shape {h.shape}"
                )

            h = h.transpose(-1, compatible_dim)

        # Compute delta
        delta = self.delta if self.delta is not None else 1.0 / self.hidden_dim

        # Apply energy dynamics
        h, self.energy, aux_loss = _apply_energy_dynamics(
            h, self.energy, delta, self.gamma, self.lambda_kl, self.beta, self.hidden_dim
        )
        self.aux_loss = aux_loss

        return h


class EnergyHookLayerWrapper(nn.Module):
    """Wraps an arbitrary module and applies energy dynamics + L1 regularization."""

    def __init__(self,
                 module: nn.Module,
                 hidden_dim: Optional[Union[int, Tuple[int, ...]]] = None,
                 delta: Optional[float] = None,
                 gamma: float = 0.05,
                 lambda_kl: float = 0.01,
                 lambda_l1: float = 0.005,
                 beta: float = 0.05,
                 output_selector: Optional[Callable] = None,
                 output_reconstructor: Optional[Callable] = None):
        super().__init__()
        self.module = module
        self.hidden_dim = hidden_dim if hidden_dim is not None else _infer_hidden_dim(module)
        self.delta = delta
        self.gamma = gamma
        self.lambda_kl = lambda_kl
        self.lambda_l1 = lambda_l1
        self.beta = beta
        self.output_selector = output_selector or (lambda x: x)
        self.output_reconstructor = output_reconstructor or (lambda x, orig: x)

        self.energy = None
        self.aux_loss = 0.0

    def forward(self, *args, **kwargs) -> Any:
        self.aux_loss = 0.0

        output = self.module(*args, **kwargs)

        # Extract tensor to process
        try:
            h = self.output_selector(output)
            if not isinstance(h, torch.Tensor):
                return output
        except (TypeError, IndexError, KeyError):
            return output

        # Infer hidden_dim if not provided
        if self.hidden_dim is None:
            self.hidden_dim = h.shape[-1]

        # Initialize energy buffer on first forward pass
        if self.energy is None:
            self.energy = torch.zeros(self.hidden_dim, device=h.device)

        # Handle shape compatibility
        if h.shape[-1] != self.hidden_dim:
            compatible_dim = None
            for dim_idx in range(len(h.shape)):
                if h.shape[dim_idx] == self.hidden_dim:
                    compatible_dim = dim_idx
                    break

            if compatible_dim is None:
                raise ValueError(
                    f"Cannot find dimension matching hidden_dim={self.hidden_dim} "
                    f"in tensor with shape {h.shape}"
                )

            h = h.transpose(-1, compatible_dim)

        # Compute delta
        delta = self.delta if self.delta is not None else 1.0 / self.hidden_dim

        # Apply energy dynamics
        h, self.energy, aux_loss = _apply_energy_dynamics(
            h, self.energy, delta, self.gamma, self.lambda_kl, self.beta, self.hidden_dim
        )

        # L1 regularization
        l1_loss = 0.0
        if self.lambda_l1 > 0.0:
            for param in self.module.parameters():
                if param.ndim >= 2:
                    l1_loss += self.lambda_l1 * param.abs().sum().item()

        self.aux_loss = aux_loss + l1_loss

        # Reconstruct output
        return self.output_reconstructor(h, output)


class EnergyWrapper(nn.Module):
    """Wraps a model with energy dynamics hooks."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        # Find all hooked layers
        self.hooked_layers: List[Tuple[str, nn.Module]] = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, (EnergyHookLayer, EnergyHookLayerWrapper))
        ]

        if not self.hooked_layers:
            raise ValueError("No EnergyHookLayer or EnergyHookLayerWrapper modules found in model.")

    def get_total_aux_loss(self) -> float:
        """Get total auxiliary loss across all hooked layers."""
        return sum(module.aux_loss for _, module in self.hooked_layers)

    def get_layer_stats(self) -> dict:
        """Get statistics for all hooked layers."""
        stats = {}
        for name, module in self.hooked_layers:
            if module.energy is not None:
                stats[name] = {
                    'aux_loss': module.aux_loss,
                    'avg_energy': module.energy.mean().item(),
                    'min_energy': module.energy.min().item(),
                    'max_energy': module.energy.max().item(),
                    'delta': module.delta if module.delta is not None else 1.0 / module.hidden_dim
                }
        return stats

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)



# Stress test with simplified Transformer
if __name__ == "__main__":

    class EnergyOrchestrator:
        """Monitors and adjusts energy dynamics across all hooked layers."""

        def __init__(self,
                     wrapped_model: EnergyWrapper,
                     aux_loss_threshold: float = 0.1,
                     aux_loss_increase_threshold: float = 0.05,
                     energy_threshold: float = -0.5,
                     delta_increase: float = 0.01):
            self.wrapped_model = wrapped_model
            self.aux_loss_threshold = aux_loss_threshold
            self.aux_loss_increase_threshold = aux_loss_increase_threshold
            self.energy_threshold = energy_threshold
            self.delta_increase = delta_increase

            # Track previous aux losses
            self.prev_aux_losses = {name: 0.0 for name, _ in wrapped_model.hooked_layers}

        def adjust_deltas(self):
            """Adjust delta values for layers with problematic energy dynamics."""
            for name, module in self.wrapped_model.hooked_layers:
                if module.energy is None:
                    continue

                aux_loss = module.aux_loss
                avg_energy = module.energy.mean().item()
                prev_aux_loss = self.prev_aux_losses[name]

                # Check if aux loss is spiking and energy is low
                aux_loss_increase = aux_loss - prev_aux_loss

                if (aux_loss > self.aux_loss_threshold and
                        aux_loss_increase > self.aux_loss_increase_threshold and
                        avg_energy < self.energy_threshold):
                    # Increase delta
                    current_delta = module.delta if module.delta is not None else 1.0 / module.hidden_dim
                    new_delta = current_delta + self.delta_increase
                    module.delta = new_delta
                    print(f"Layer {name}: Increased delta from {current_delta:.4f} to {new_delta:.4f} "
                          f"(aux_loss={aux_loss:.4f}, avg_energy={avg_energy:.4f})")

                self.prev_aux_losses[name] = aux_loss

    class SimplifiedTransformer(nn.Module):
        def __init__(self, d_model=64, nhead=4, dim_feedforward=256, num_layers=2):
            super().__init__()
            self.d_model = d_model

            # Embedding
            self.embedding = nn.Embedding(100, d_model)

            # Transformer layers with energy hooks
            self.layers = nn.ModuleList()
            for i in range(num_layers):
                layer = nn.ModuleDict({
                    'attn': EnergyHookLayerWrapper(
                        nn.MultiheadAttention(d_model, nhead, batch_first=True),
                        output_selector=lambda x: x[0],  # Get attention output
                        output_reconstructor=lambda h, orig: (h, orig[1]),  # Preserve attn weights
                        gamma=0.08,
                        lambda_l1=0.01
                    ),
                    'ff1': EnergyHookLayerWrapper(
                        nn.Linear(d_model, dim_feedforward),
                        gamma=0.1,
                        lambda_l1=0.01
                    ),
                    'ff2': EnergyHookLayerWrapper(
                        nn.Linear(dim_feedforward, d_model),
                        gamma=0.1,
                        lambda_l1=0.01
                    )
                })
                self.layers.append(layer)

            self.output = nn.Linear(d_model, 100)

        def forward(self, x):
            x = self.embedding(x)

            for layer in self.layers:
                # Self-attention
                attn_out, _ = layer['attn'](x, x, x)
                x = x + attn_out

                # Feedforward
                ff = torch.relu(layer['ff1'](x))
                ff = layer['ff2'](ff)
                x = x + ff

            return self.output(x)


    # Create model and wrap it
    model = SimplifiedTransformer(d_model=64, nhead=4, dim_feedforward=256, num_layers=2)
    wrapped_model = EnergyWrapper(model)
    orchestrator = EnergyOrchestrator(
        wrapped_model,
        aux_loss_threshold=0.1,
        aux_loss_increase_threshold=0.05,
        energy_threshold=-0.5,
        delta_increase=0.01
    )

    print("=== Transformer Energy Dynamics Test ===\n")
    print(f"Found {len(wrapped_model.hooked_layers)} hooked layers:")
    for name, _ in wrapped_model.hooked_layers:
        print(f"  - {name}")
    print()

    # Training loop
    optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=0.001)

    for step in range(10):
        # Forward pass
        x = torch.randint(0, 100, (8, 20))  # (batch, seq_len)
        output = wrapped_model(x)

        # Compute loss
        target = torch.randint(0, 100, (8, 20))
        loss = nn.functional.cross_entropy(output.transpose(1, 2), target)

        # Add auxiliary loss
        aux_loss = wrapped_model.get_total_aux_loss()
        total_loss = loss + aux_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Print statistics
        print(f"Step {step + 1}:")
        print(f"  Loss: {loss.item():.4f}, Aux Loss: {aux_loss:.4f}, Total: {total_loss.item():.4f}")

        stats = wrapped_model.get_layer_stats()
        for name, stat in stats.items():
            print(f"  {name}: aux={stat['aux_loss']:.4f}, "
                  f"avg_energy={stat['avg_energy']:.3f}, "
                  f"delta={stat['delta']:.4f}")

        # Adjust deltas if needed
        orchestrator.adjust_deltas()
        print()