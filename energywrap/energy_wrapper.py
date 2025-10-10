import torch
import torch.nn as nn
from typing import Tuple, List, Callable, Optional, Union


class EnergyWrapper(nn.Module):
    """Wraps a model to apply energy dynamics to hidden neurons."""

    def __init__(self, model: nn.Module, output_layers: List[nn.Module], delta: float = None, gamma: float = 0.05,
                 lambda_kl: float = 0.01, lambda_l1: float = 0.005, beta: float = 0.05,
                 hidden_layer_types: Optional[Tuple[type, ...]] = None,
                 hidden_layer_filter: Optional[Callable[[nn.Module], bool]] = None):
        """Initialize EnergyWrapper, identifying hidden layers and setting energy parameters.

        Args:
            model: The neural network to wrap.
            output_layers: List of modules producing final outputs (excluded from hidden layers).
            delta: Energy increment per step (default: 1/hidden_dim to prevent self-extinguishing).
            gamma: Energy drain factor, range [0.01, 0.5] for stability (https://arxiv.org/abs/1904.13228).
            lambda_kl: KL-divergence weight for sparsity, range [0.001, 0.1] (https://jmlr.csail.mit.edu/papers/v11/lee10a.html).
            lambda_l1: L1 regularization weight for sparsity, range [0.001, 0.1] (https://arxiv.org/abs/1904.13228).
            beta: Target sparsity (~5% active neurons, https://arxiv.org/abs/1803.10915).
            hidden_layer_types: Tuple of module types to consider as hidden layers (default: Linear, RNNCell, LSTM, GRU, Conv2d).
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
        # Default hidden layer types
        self.hidden_layer_types = hidden_layer_types or (nn.Linear, nn.RNNCell, nn.LSTM, nn.GRU, nn.Conv2d)
        self.hidden_layer_filter = hidden_layer_filter or (lambda m: isinstance(m, self.hidden_layer_types))
        # Identify hidden layers: Modules passing filter and not in output_layers
        self.hidden_layers = []
        self.hidden_dims = []
        self.energy_buffers = []
        self.hooks = []
        for name, module in model.named_modules():
            if self.hidden_layer_filter(module) and module not in output_layers:
                self.hidden_layers.append((name, module))
                hidden_dim = self._get_hidden_dim(module)
                self.hidden_dims.append(hidden_dim)
                self.energy_buffers.append(self.register_buffer(f'energy_{name}', torch.zeros(hidden_dim)))
        if not self.hidden_layers:
            raise ValueError("No hidden layers found in model")
        # Delta per layer: Scaled to ensure delta * hidden_dim >= 1
        self.deltas = [1.0 / dim if delta is None else delta for dim in self.hidden_dims]
        for i, (delta, dim) in enumerate(zip(self.deltas, self.hidden_dims)):
            assert delta * dim >= 1, f"delta * hidden_dim < 1 for layer {i} (https://arxiv.org/abs/1904.13228)"
        # Assertions for hyperparameter ranges
        assert 0.01 <= gamma <= 0.5, "gamma must be in [0.01, 0.5] (https://arxiv.org/abs/1904.13228)"
        assert 0.001 <= lambda_kl <= 0.1, "lambda_kl must be in [0.001, 0.1] (https://jmlr.csail.mit.edu/papers/v11/lee10a.html)"
        assert 0.001 <= lambda_l1 <= 0.1, "lambda_l1 must be in [0.001, 0.1] (https://arxiv.org/abs/1904.13228)"
        self.beta = beta  # Target 5% active neurons (https://arxiv.org/abs/1803.10915)
        #self.hidden_history: List[List[torch.Tensor]] = [[] for _ in self.hidden_layers]
        #self.energy_history: List[List[torch.Tensor]] = [[] for _ in self.hidden_layers]
        self.aux_loss = torch.tensor(0.0)
        # Register forward hooks to apply energy dynamics
        for idx, (name, module) in enumerate(self.hidden_layers):
            handle = module.register_forward_hook(self._make_hook(idx, name))
            self.hooks.append(handle)

    def _get_hidden_dim(self, module: nn.Module) -> int:
        """Infer hidden dimension from module.

        Returns dimension of output neurons (e.g., out_features for Linear, hidden_size for RNNCell).
        Raises NotImplementedError for unsupported types to ensure explicit handling.
        """
        if isinstance(module, nn.Linear):
            return module.out_features
        elif isinstance(module, (nn.RNNCell, nn.LSTM, nn.GRU)):
            return module.hidden_size
        elif isinstance(module, nn.Conv2d):
            return module.out_channels  # Flattened in hook
        raise NotImplementedError(f"Cannot infer hidden_dim for module type: {type(module)}")

    def process_hidden_output(self, module: nn.Module,
                              output: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        """Extract hidden neuron activations from module output.

        Converts module output (e.g., RNNCell, LSTM, Conv2d) into a 2D tensor [batch, hidden_dim].
        Override for custom modules. Raises NotImplementedError for unsupported types.
        """
        if isinstance(module, nn.RNNCell):
            return output  # (batch, hidden_size)
        elif isinstance(module, nn.LSTM):
            return output[0]  # h: (batch, hidden_size)
        elif isinstance(module, nn.GRU):
            return output  # (batch, hidden_size)
        elif isinstance(module, nn.Linear):
            return output  # (batch, out_features)
        elif isinstance(module, nn.Conv2d):
            return output.flatten(start_dim=1)  # (batch, channels*H*W)
        raise NotImplementedError(f"Unsupported module type: {type(module)}")

    def restore_output_shape(self, module: nn.Module, h: torch.Tensor,
                             original_output: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> Union[
        torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Restore modified hidden output to original shape for downstream compatibility.

        Ensures modified activations (e.g., after energy dynamics) match the module's expected output format.
        For example, Conv2d outputs are reshaped to [batch, channels, H, W], LSTM outputs include cell state.
        """
        if isinstance(module, nn.Conv2d):
            batch_size = original_output.size(0)
            return h.view(batch_size, module.out_channels, *original_output.shape[2:])
        elif isinstance(module, nn.LSTM):
            return (h, original_output[1])  # Restore (h, c)
        return h

    def _make_hook(self, layer_idx: int, layer_name: str):
        """Create forward hook to apply energy dynamics to hidden layer output.

        Applies delta increment, gamma drain, KL-divergence (target ~5% active neurons),
        L1 sparsity on weights, and energy thresholds (≥2 fire, ≤-2 shutoff).
        Modifies activations in-place and accumulates auxiliary loss.
        Energy penalties (|x| > 1) and thresholds (≥2, ≤-2) mimic biological neural dynamics:
        - |x| > 1 penalties: >1 destroys pathological attractors (repetitive loops, as in BasicRNN.init_seizure
          where W_hh[:k,:k] = 0.9*eye(k) creates self-reinforcing cycles) [web:10].
        - |x| < 1 penalties: Encourages recovering dead (zero-output) neurons to maintain network capacity [web:5].
        - Energy ≥ 2: Forces firing (h=2) to recover dead/zero-output neurons, ensuring they contribute [web:7].
        - Energy ≤ -2: Extinguishes seizures/pathological attractors (repetitive RNN loops) to stabilize network [web:1].
        """

        def hook(module, input, output):
            h = self.process_hidden_output(module, output)
            a = h.clamp(min=0)
            # KL-divergence: Encourages ~5% active neurons
            rho = a.gt(0).float().mean(dim=1).mean()
            rho = torch.clamp(rho, 1e-5, 1 - 1e-5)
            kl_loss = self.lambda_kl * (rho * torch.log(rho / self.beta) +
                                        (1 - rho) * torch.log((1 - rho) / (1 - self.beta)))
            # L1 sparsity: Promotes sparse weights
            l1_loss = 0.0
            for param in module.parameters():
                if param.ndim >= 2:
                    l1_loss += self.lambda_l1 * param.abs().sum()
            # Energy update: Delta increment, noise, and gamma drain
            energy = getattr(self, f'energy_{layer_name}')
            noise = torch.normal(mean=0.0, std=0.001, size=(self.hidden_dims[layer_idx],),
                                 device=energy.device)
            new_energy = energy.detach() + self.deltas[layer_idx] + noise - self.gamma * a.mean(dim=0)
            # Energy penalties: |x| > 1 destroys pathological attractors, <1 encourages recovering dead neurons
            high_mask = new_energy > 1
            low_mask = new_energy < -1
            if high_mask.any() or low_mask.any():
                self.aux_loss += 0.01 * (new_energy[high_mask].abs() - 1).sum()
                self.aux_loss += 0.01 * (new_energy[low_mask].abs() - 1).sum()
            # Energy thresholds: ≥2 forces firing to recover dead neurons, ≤-2 extinguishes seizures
            fire_mask = new_energy >= 2
            shutoff_mask = new_energy <= -2
            if fire_mask.any():
                h = h.clone()
                h[:, fire_mask] = 2.0
            if shutoff_mask.any():
                h = h.clone()
                h[:, shutoff_mask] = (energy.detach() + 2)[shutoff_mask]
                new_energy[shutoff_mask] = -2
            setattr(self, f'energy_{layer_name}', new_energy)
            #self.energy_history[layer_idx].append(new_energy.clone().detach())
            #self.hidden_history[layer_idx].append(h.clone().detach())
            self.aux_loss += kl_loss + l1_loss
            return self.restore_output_shape(module, h, output)

        return hook

    def forward(self, *args, **kwargs):
        """Forward pass through wrapped model, applying energy dynamics via hooks.

        Resets auxiliary loss and clears history buffers before each pass.
        Returns the model's output unchanged.
        """
        self.aux_loss = torch.tensor(0.0)
        output = self.model(*args, **kwargs)
        return output

    def __getattr__(self, name: str):
        """Delegate attribute access to wrapped model.

        Allows seamless access to model attributes (e.g., hidden_size) while maintaining wrapper encapsulation.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def wrap_energy_dynamics(model: nn.Module, output_layers: List[nn.Module], delta: float = None, gamma: float = 0.05,
                         lambda_kl: float = 0.01, lambda_l1: float = 0.005, beta: float = 0.05,
                         hidden_layer_types: Optional[Tuple[type, ...]] = None,
                         hidden_layer_filter: Optional[Callable[[nn.Module], bool]] = None) -> nn.Module:
    """Wrap a model to add energy dynamics to its hidden neurons.

    Returns an EnergyWrapper instance that applies delta increments, gamma drains,
    KL/L1 sparsity, and energy thresholds to hidden layer outputs via forward hooks.
    """
    return EnergyWrapper(model, output_layers, delta, gamma, lambda_kl, lambda_l1, beta,
                         hidden_layer_types, hidden_layer_filter)