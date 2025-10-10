import torch
import torch.nn as nn

class BasicRNN(nn.Module):
    """Vanilla RNN with RNNCell and Linear output layer."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Process sequence, return outputs and final hidden state.
        x: (batch_size, T, input_size), h: (batch_size, hidden_size).
        """
        batch_size, T, _ = x.size()
        x = x.permute(1, 0, 2)  # (T, batch_size, input_size)
        outputs = []
        h_t = h.detach()
        for t in range(T):
            h_t = self.rnn_cell(x[t], h_t)
            out = self.fc(h_t)
            outputs.append(out)
        return torch.stack(outputs).permute(1, 0, 2), h_t

    def init_seizure(self, k: int = 10):
        """Set W_hh[:k,:k] to 0.9*eye(k) to model pathological attractors.
        In RNNs: Strong diagonal weights create self-reinforcing cycles, mimicking
        epileptic seizures in biological brains where neurons fire synchronously in loops.
        Used to test robustness against such attractors (repetitive, unwanted outputs).
        """
        with torch.no_grad():
            self.rnn_cell.weight_ih.zero_()
            self.rnn_cell.bias_ih.zero_()
            self.rnn_cell.bias_hh.zero_()
            self.rnn_cell.weight_hh.zero_()
            self.rnn_cell.weight_hh[:k, :k] = torch.eye(k) * 0.9
            self.fc.weight.zero_()
            self.fc.bias.zero_()