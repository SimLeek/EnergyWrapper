# EnergyWrapper

A Python library for wrapping neural networks in an energy system that:
* ensures neurons fire at some point (zero init hidden layers and RL action output layers can learn)
* stops pathalogical attractors from forming and undoes them
* Causes Curiosity RL on every set of neurons or layers it is applied to
* Causes hidden layers to act as input predictors when given no other learning functions

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Results

Larger error is worse.

Post training results from a "seizure init" RNN with a pathalogical attractor built in. The pathalogical attractor could not be removed through normal training:
[![RNN plot](plots/rnn_error_plot_seizure_char_control.png)]()
Post training results from the energy wrapped RNN:
[![RNN plot](plots/rnn_error_plot_seizure_char_wrapped.png)]()
copy task that's more repetitive. This time with a zero-init RNN. It was unable to learn:
[![RNN plot](plots/rnn_error_plot_zero_copy_control.png)]()
Post training on a wrapped neural network:
[![RNN plot](plots/rnn_error_plot_zero_copy_wrapped.png
I'm pretty sure they overfit, but they didn't output the same thing over and over again, unlike the other RNNs.

## Installation

### Local Development (Editable Install)

```bash
# Clone the repository
git clone https://github.com/simleek/energywrapper.git
cd curious-rl

# Install in editable mode
pip install -e .
```

### Direct from GitHub

```bash
# Install latest from main branch
pip install git+https://github.com/simleek/energywrapper.git
```

## License

MIT License - see LICENSE file for details.

manual_qwen3_vl.py is under the apache 2.0 license

## Citation

If you use this library in your research, please cite:

```bibtex
@software{energywrapper,
  author = {SimLeek},
  title = {EnergyWrapper: A Python library for wrapping neural networks in an energy system},
  year = {2026},
  url = {https://github.com/simleek/energywrapper}
}
```
