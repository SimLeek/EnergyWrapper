import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from example.basic_rnn import BasicRNN
from energywrap.automatic import wrap_energy_dynamics

# Dataset for sine wave prediction
def generate_sine_wave_seq(T: int = 150) -> torch.Tensor:
    """Generate T-step sine wave sequence with random phase offset."""
    t = torch.arange(T, dtype=torch.float32) * 0.1
    phi = torch.rand(1).item() * 2 * torch.pi
    x = torch.stack([torch.cos(t + phi), torch.sin(t + phi)], dim=1)
    return x

# Dataset for sequence copying
def generate_copy_seq(T: int = 150) -> torch.Tensor:
    """Generate 15-step random binary sub-sequence, repeat to fill T."""
    sub_len = 15
    bits = torch.randint(0, 2, (sub_len,))
    x = torch.zeros(T, 2)
    for t in range(T):
        x[t, bits[t % sub_len]] = 1
    return x

# Dataset for next-character prediction
def generate_char_seq(T: int = 150) -> torch.Tensor:
    """Generate T-step random digit sequence, map to even/odd."""
    digits = torch.randint(0, 10, (T,))
    x = torch.zeros(T, 2)
    for t in range(T):
        x[t, digits[t] % 2] = 1
    return x

# Dataset for rare event
def generate_rare_event_seq(T: int = 150) -> torch.Tensor:
    """Generate T-step sequence with geometrically decreasing 1s."""
    x = torch.zeros(T, 2)
    k = torch.randint(0, T // 2 + 1, (1,)).item()
    indices = [k]
    current = k
    gap = k // 2
    while current + gap + 1 < T and gap > 0:
        current += gap + 1
        indices.append(current)
        gap //= 2
    for idx in indices:
        x[idx, 1] = 1
        x[idx, 0] = 0
    for t in range(T):
        if x[t, 1] == 0:
            x[t, 0] = 1
    return x

class SequenceDataset:
    def __init__(self, T: int = 150, task: str = 'sine'):
        self.task = task
        if task == 'sine':
            self.x = generate_sine_wave_seq(T)
        elif task == 'copy':
            self.x = generate_copy_seq(T)
        elif task == 'char':
            self.x = generate_char_seq(T)
        elif task == 'rare':
            self.x = generate_rare_event_seq(T)

    def get_sequence(self) -> torch.Tensor:
        return self.x

# Training and testing function for wrapped model
def train_and_test(model, train_dataset, test_dataset, small_test_dataset, task: str, task_output_size: int, max_epochs: int = 200):
    criterion = nn.MSELoss() if task == 'sine' else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    T = 150
    consecutive_90 = 0
    x_train = train_dataset.get_sequence()
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_aux = 0.0
        correct = 0
        total = 0
        h = torch.zeros(1, model.hidden_size)
        for t in range(T):
            optimizer.zero_grad()
            x_t = x_train[t:t+1].unsqueeze(0)
            target = x_train[t, 1:2].unsqueeze(0) if task == 'sine' else x_train[t].unsqueeze(0)
            out, h = model(x_t, h)
            out = out[:, :, :task_output_size]
            if task == 'sine':
                main_loss = criterion(out.squeeze(-1), target)
                low_energy_loss = 0
            else:
                main_loss = criterion(out.view(-1, task_output_size), target.argmax(dim=-1).long())
                output_probs = torch.softmax(out, dim=-1)
                low_energy_loss = 0.005 * torch.clamp(0.1 - output_probs.sum(dim=-1), min=0).mean()
            total_loss = main_loss + model.aux_loss + low_energy_loss
            epoch_loss += total_loss.item()
            epoch_aux += model.aux_loss.item()
            total_loss.backward()
            optimizer.step()
            if task == 'sine':
                error = torch.abs(out.squeeze(-1) - target).item()
                correct += (error < 0.1)
            elif task == 'rare':
                pred = out.view(-1, task_output_size).argmax(dim=-1)
                if target.argmax(dim=-1).item() == 1:
                    total += 1
                    correct += (pred == 1).sum().item()
            else:
                pred = out.view(-1, task_output_size).argmax(dim=-1)
                correct += (pred == target.argmax(dim=-1)).sum().item()
                total += 1
        avg_loss = epoch_loss / T
        avg_aux = epoch_aux / T
        acc = correct / total if total > 0 else 0.0
        model.eval()
        test_correct = 0
        test_total = 0
        x_test = small_test_dataset.get_sequence()
        with torch.no_grad():
            h = torch.zeros(1, model.hidden_size)
            out_seq = []
            for t in range(T):
                x_t = x_test[t:t+1].unsqueeze(0)
                target = x_test[t, 1:2].unsqueeze(0) if task == 'sine' else x_test[t].unsqueeze(0)
                out, h = model(x_t, h)
                out = out[:, :, :task_output_size]
                out_seq.append(out)
                if task == 'sine':
                    error = torch.abs(out.squeeze(-1) - target).item()
                    test_correct += (error < 0.1)
                elif task == 'rare':
                    pred = out.view(-1, task_output_size).argmax(dim=-1)
                    if target.argmax(dim=-1).item() == 1:
                        test_total += 1
                        test_correct += (pred == 1).sum().item()
                else:
                    pred = out.view(-1, task_output_size).argmax(dim=-1)
                    test_correct += (pred == target.argmax(dim=-1)).sum().item()
                    test_total += 1
        test_acc = test_correct / test_total if test_total > 0 else 0.0
        model.train()
        if epoch % 20 == 0 or (test_acc > 0.9 and consecutive_90 >= 9):
            num_non_zero = sum((p.abs() > 1e-6).sum().item() for p in model.parameters() if p.ndim >= 2)
            first_h = [0] * 10
            out = torch.cat(out_seq, dim=1)
            if task == 'sine':
                pred_seq = out[0, -6:, 0].tolist()
                true_seq = x_test[-6:, 1].tolist()
                pred_str = ', '.join(f'{x:.4f}' for x in pred_seq)
                true_str = ', '.join(f'{x:.4f}' for x in true_seq)
            else:
                pred_seq = out[0, -6:, :].argmax(dim=-1).tolist()
                true_seq = x_test[-6:].argmax(dim=-1).tolist()
                pred_str = ', '.join(f'{x}' for x in pred_seq)
                true_str = ', '.join(f'{x}' for x in true_seq)
            print(f'Epoch {epoch}, Task {task}, Avg Loss: {avg_loss:.6f}, '
                  f'Avg Aux: {avg_aux:.6f}, Test Acc: {test_acc*100:.6f}%')
            print(f'Pred:\t{pred_str}')
            print(f'True:\t{true_str}')
            if task == 'rare':
                true_ones = (x_test.argmax(dim=-1) == 1).sum().item()
                correct_ones = sum(1 for p, t in zip(out[0].argmax(dim=-1), x_test.argmax(dim=-1)) if t == 1 and p == 1)
                print(f'Correct 1s: {correct_ones}/{true_ones} ({correct_ones/true_ones*100:.2f}%)')
            print(f'Num Non-Zero Weights: {num_non_zero}, First 10 Hidden: {first_h}')
            if epoch == 0:
                print(f'First 10 steps raw outputs (Wrapped):')
                for t in range(min(10, T)):
                    out_t = out[0, t, :].tolist()
                    if task == 'sine':
                        print(f'Step {t}: Pred [{out_t[0]:.4f}], True [{x_test[t, 1]:.4f}]')
                    else:
                        print(f'Step {t}: Pred [{out_t[0]:.4f}, {out_t[1]:.4f}], True [{x_test[t, 0]:.4f}, {x_test[t, 1]:.4f}]')
        if test_acc > 0.9:
            consecutive_90 += 1
            if consecutive_90 >= 10:
                print(f'Task {task} achieved >90% accuracy for 10 epochs. Moving to next task.')
                out = torch.cat(out_seq, dim=1)
                if task == 'sine':
                    pred_seq = out[0, -6:, 0].tolist()
                    true_seq = x_test[-6:, 1].tolist()
                else:
                    pred_seq = out[0, -6:, :].argmax(dim=-1).tolist()
                    true_seq = x_test[-6:].argmax(dim=-1).tolist()
                pred_str = ', '.join(f'{x:.4f}' if task == 'sine' else f'{x}' for x in pred_seq)
                true_str = ', '.join(f'{x:.4f}' if task == 'sine' else f'{x}' for x in true_seq)
                print(f'Pred:\t{pred_str}')
                print(f'True:\t{true_str}')
                if task == 'rare':
                    true_ones = (x_test.argmax(dim=-1) == 1).sum().item()
                    correct_ones = sum(1 for p, t in zip(out[0].argmax(dim=-1), x_test.argmax(dim=-1)) if t == 1 and p == 1)
                    print(f'Correct 1s: {correct_ones}/{true_ones} ({correct_ones/true_ones*100:.2f}%)')
                return True
        else:
            consecutive_90 = 0
    print(f'Task {task} reached max epochs ({max_epochs}). Moving to next task.')
    return False

# Training and testing function for vanilla BasicRNN (control)
def train_and_test_no_wrapper(model, train_dataset, test_dataset, small_test_dataset, task: str, task_output_size: int, max_epochs: int = 200):
    criterion = nn.MSELoss() if task == 'sine' else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    T = 150
    consecutive_90 = 0
    x_train = train_dataset.get_sequence()
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        h = torch.zeros(1, model.hidden_size)
        for t in range(T):
            optimizer.zero_grad()
            x_t = x_train[t:t+1].unsqueeze(0)
            target = x_train[t, 1:2].unsqueeze(0) if task == 'sine' else x_train[t].unsqueeze(0)
            out, h = model(x_t, h)
            out = out[:, :, :task_output_size]
            loss = criterion(out.squeeze(-1), target) if task == 'sine' else criterion(out.view(-1, task_output_size), target.argmax(dim=-1).long())
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            if task == 'sine':
                error = torch.abs(out.squeeze(-1) - target).item()
                correct += (error < 0.1)
            elif task == 'rare':
                pred = out.view(-1, task_output_size).argmax(dim=-1)
                if target.argmax(dim=-1).item() == 1:
                    total += 1
                    correct += (pred == 1).sum().item()
            else:
                pred = out.view(-1, task_output_size).argmax(dim=-1)
                correct += (pred == target.argmax(dim=-1)).sum().item()
                total += 1
        avg_loss = epoch_loss / T
        acc = correct / total if total > 0 else 0.0
        model.eval()
        test_correct = 0
        test_total = 0
        x_test = small_test_dataset.get_sequence()
        with torch.no_grad():
            h = torch.zeros(1, model.hidden_size)
            out_seq = []
            for t in range(T):
                x_t = x_test[t:t+1].unsqueeze(0)
                target = x_test[t, 1:2].unsqueeze(0) if task == 'sine' else x_test[t].unsqueeze(0)
                out, h = model(x_t, h)
                out = out[:, :, :task_output_size]
                out_seq.append(out)
                if task == 'sine':
                    error = torch.abs(out.squeeze(-1) - target).item()
                    test_correct += (error < 0.1)
                elif task == 'rare':
                    pred = out.view(-1, task_output_size).argmax(dim=-1)
                    if target.argmax(dim=-1).item() == 1:
                        test_total += 1
                        test_correct += (pred == 1).sum().item()
                else:
                    pred = out.view(-1, task_output_size).argmax(dim=-1)
                    test_correct += (pred == target.argmax(dim=-1)).sum().item()
                    test_total += 1
        test_acc = test_correct / test_total if test_total > 0 else 0.0
        model.train()
        if epoch % 20 == 0 or (test_acc > 0.9 and consecutive_90 >= 9):
            num_non_zero = sum((p.abs() > 1e-6).sum().item() for p in model.parameters() if p.ndim >= 2)
            first_h = [0] * 10
            out = torch.cat(out_seq, dim=1)
            if task == 'sine':
                pred_seq = out[0, -6:, 0].tolist()
                true_seq = x_test[-6:, 1].tolist()
                pred_str = ', '.join(f'{x:.4f}' for x in pred_seq)
                true_str = ', '.join(f'{x:.4f}' for x in true_seq)
            else:
                pred_seq = out[0, -6:, :].argmax(dim=-1).tolist()
                true_seq = x_test[-6:].argmax(dim=-1).tolist()
                pred_str = ', '.join(f'{x}' for x in pred_seq)
                true_str = ', '.join(f'{x}' for x in true_seq)
            print(f'Epoch {epoch}, Task {task} (Control), Avg Loss: {avg_loss:.6f}, Test Acc: {test_acc*100:.6f}%')
            print(f'Pred:\t{pred_str}')
            print(f'True:\t{true_str}')
            if task == 'rare':
                true_ones = (x_test.argmax(dim=-1) == 1).sum().item()
                correct_ones = sum(1 for p, t in zip(out[0].argmax(dim=-1), x_test.argmax(dim=-1)) if t == 1 and p == 1)
                print(f'Correct 1s: {correct_ones}/{true_ones} ({correct_ones/true_ones*100:.2f}%)')
            print(f'Num Non-Zero Weights: {num_non_zero}, First 10 Hidden: {first_h}')
            if epoch == 0:
                print(f'First 10 steps raw outputs (Control):')
                for t in range(min(10, T)):
                    out_t = out[0, t, :].tolist()
                    if task == 'sine':
                        print(f'Step {t}: Pred [{out_t[0]:.4f}], True [{x_test[t, 1]:.4f}]')
                    else:
                        print(f'Step {t}: Pred [{out_t[0]:.4f}, {out_t[1]:.4f}], True [{x_test[t, 0]:.4f}, {x_test[t, 1]:.4f}]')
        if test_acc > 0.9:
            consecutive_90 += 1
            if consecutive_90 >= 10:
                print(f'Task {task} (Control) achieved >90% accuracy for 10 epochs. Moving to next task.')
                out = torch.cat(out_seq, dim=1)
                if task == 'sine':
                    pred_seq = out[0, -6:, 0].tolist()
                    true_seq = x_test[-6:, 1].tolist()
                else:
                    pred_seq = out[0, -6:, :].argmax(dim=-1).tolist()
                    true_seq = x_test[-6:].argmax(dim=-1).tolist()
                pred_str = ', '.join(f'{x:.4f}' if task == 'sine' else f'{x}' for x in pred_seq)
                true_str = ', '.join(f'{x:.4f}' if task == 'sine' else f'{x}' for x in true_seq)
                print(f'Pred:\t{pred_str}')
                print(f'True:\t{true_str}')
                if task == 'rare':
                    true_ones = (x_test.argmax(dim=-1) == 1).sum().item()
                    correct_ones = sum(1 for p, t in zip(out[0].argmax(dim=-1), x_test.argmax(dim=-1)) if t == 1 and p == 1)
                    print(f'Correct 1s: {correct_ones}/{true_ones} ({correct_ones/true_ones*100:.2f}%)')
                return True
        else:
            consecutive_90 = 0
    print(f'Task {task} (Control) reached max epochs ({max_epochs}). Moving to next task.')
    return False

def main():
    tasks = ['sine', 'copy', 'char', 'rare']
    input_size = 2
    hidden_size = 64
    output_sizes = {'sine': 1, 'copy': 2, 'char': 2, 'rare': 2}
    T = 150
    max_epochs = 200
    window_size = 10

    for init_name in ['zero', 'seizure']:
        print(f'\nTraining with {init_name} initialization')
        # Wrapped model
        model_wrapped = BasicRNN(input_size, hidden_size, max(output_sizes.values()))
        if init_name == 'zero':
            with torch.no_grad():
                for param in model_wrapped.parameters():
                    param.zero_()
        elif init_name == 'seizure':
            model_wrapped.init_seizure(k=10)
        model_wrapped = wrap_energy_dynamics(model_wrapped, output_layers=[model_wrapped.fc])
        # Control model (vanilla BasicRNN)
        model_control = BasicRNN(input_size, hidden_size, max(output_sizes.values()))
        if init_name == 'zero':
            with torch.no_grad():
                for param in model_control.parameters():
                    param.zero_()
        elif init_name == 'seizure':
            model_control.init_seizure(k=10)
        for task in tasks:
            print(f'\nStarting task: {task} (Wrapped)')
            train_dataset = SequenceDataset(T, task)
            test_dataset = SequenceDataset(T, task)
            small_test_dataset = SequenceDataset(T, task)
            train_and_test(model_wrapped, train_dataset, test_dataset, small_test_dataset, task, output_sizes[task], max_epochs)
            print(f'\nStarting task: {task} (Control)')
            train_and_test_no_wrapper(model_control, train_dataset, test_dataset, small_test_dataset, task, output_sizes[task], max_epochs)
            # Final evaluation for both models
            for model_type, model in [('Wrapped', model_wrapped), ('Control', model_control)]:
                model.eval()
                correct = 0
                total = 0
                x = test_dataset.get_sequence()
                with torch.no_grad():
                    h = torch.zeros(1, model.hidden_size)
                    out_seq = []
                    for t in range(T):
                        x_t = x[t:t+1].unsqueeze(0)
                        target = x[t, 1:2].unsqueeze(0) if task == 'sine' else x[t].unsqueeze(0)
                        out, h = model(x_t, h)
                        out = out[:, :, :output_sizes[task]]
                        out_seq.append(out)
                        if task == 'sine':
                            error = torch.abs(out.squeeze(-1) - target).item()
                            correct += (error < 0.1)
                        elif task == 'rare':
                            pred = out.view(-1, output_sizes[task]).argmax(dim=-1)
                            if target.argmax(dim=-1).item() == 1:
                                total += 1
                                correct += (pred == 1).sum().item()
                        else:
                            pred = out.view(-1, output_sizes[task]).argmax(dim=-1)
                            correct += (pred == target.argmax(dim=-1)).sum().item()
                            total += 1
                    acc = correct / total if total > 0 else 0.0
                    print(f'{init_name} {task} {model_type} Final Test Acc: {acc*100:.6f}%')
                    out = torch.cat(out_seq, dim=1)
                    print(f'\n{init_name} {task} {model_type} Example Predictions (first 100 steps):')
                    if task == 'sine':
                        pred_seq = out[0, :100, 0].tolist()
                        true_seq = x[:100, 1].tolist()
                        pred_str = ', '.join(f'{x:.4f}' for x in pred_seq)
                        true_str = ', '.join(f'{x:.4f}' for x in true_seq)
                        print(f'Pred:\t{pred_str}')
                        print(f'True:\t{true_str}')
                    else:
                        pred_seq = out[0, :100, :].argmax(dim=-1).tolist()
                        true_seq = x[:100].argmax(dim=-1).tolist()
                        pred_str = ', '.join(f'{x}' for x in pred_seq)
                        true_str = ', '.join(f'{x}' for x in true_seq)
                        print(f'Pred:\t{pred_str}')
                        print(f'True:\t{true_str}')
                    if task == 'rare':
                        true_ones = (x.argmax(dim=-1) == 1).sum().item()
                        correct_ones = sum(1 for p, t in zip(out[0].argmax(dim=-1), x.argmax(dim=-1)) if t == 1 and p == 1)
                        print(f'Correct 1s: {correct_ones}/{true_ones} ({correct_ones/true_ones*100:.2f}%)')
                    print(f'First 10 steps raw outputs ({model_type}):')
                    for t in range(min(10, T)):
                        out_t = out[0, t, :].tolist()
                        if task == 'sine':
                            print(f'Step {t}: Pred [{out_t[0]:.4f}], True [{x[t, 1]:.4f}]')
                        else:
                            print(f'Step {t}: Pred [{out_t[0]:.4f}, {out_t[1]:.4f}], True [{x[t, 0]:.4f}, {x[t, 1]:.4f}]')
            if task != 'sine':
                for model_type, model in [('Wrapped', model_wrapped), ('Control', model_control)]:
                    errors = []
                    x = test_dataset.get_sequence()
                    with torch.no_grad():
                        h = torch.zeros(1, model.hidden_size)
                        for t in range(T):
                            x_t = x[t:t+1].unsqueeze(0)
                            target = x[t, 1:2].unsqueeze(0) if task == 'sine' else x[t].unsqueeze(0)
                            out, h = model(x_t, h)
                            out = out[:, :, :output_sizes[task]]
                            if task == 'sine':
                                error = torch.abs(out.squeeze(-1) - target).item()
                            else:
                                pred = out.view(-1, output_sizes[task]).argmax(dim=-1)
                                error = 1.0 if pred.item() != target.argmax(dim=-1).item() else 0.0
                            errors.append(error)
                    moving_avg = [sum(errors[max(0, i-window_size+1):i+1]) / len(errors[max(0, i-window_size+1):i+1]) for i in range(T)]
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(range(T), moving_avg, label='Moving Avg Error (window=10)')
                    ax.set_title(f'Moving Average Error Over Steps ({init_name} init, {task} task, {model_type})')
                    ax.set_xlabel('Step')
                    ax.set_ylabel('Error')
                    ax.legend()
                    plt.tight_layout()
                    plt.savefig(f'rnn_error_plot_{init_name}_{task}_{model_type.lower()}.png')
                    plt.close()
                    print(f'Plot saved to rnn_error_plot_{init_name}_{task}_{model_type.lower()}.png')
            else:
                print(f'Skipping error plot for {task} as requested.')

if __name__ == '__main__':
    main()