# Hybrid Second-Order Optimization of Quantum Circuits

**Iterative Preconditioned Gradient Descent (IPG) and Limited-Memory BFGS (L-BFGS) for variational state preparation, validated on IBM quantum hardware.**

Author: [Takudzwa Chitsa](https://github.com/nowins3), World Science Scholars

---

## Overview

Useful quantum subroutines such as state initialization and the Quantum Fourier Transform often need deep circuits, and the multi-qubit gates in them dominate the noise on NISQ hardware. This project treats gate-angle selection as a classical optimization problem. A parameterized 3-qubit circuit is tuned until it prepares a target state with high fidelity, using second-order information to move through a non-convex landscape with local minima and barren plateaus.

Four optimizers are implemented from scratch and compared on the same ansatz and cost function:

| Method | Order | Idea |
|---|---|---|
| Gradient Descent (GD) | First | Step against the gradient |
| Adam | First | Adaptive per-parameter step sizes from gradient moments |
| **IPG** | Second-order (preconditioned) | Gradient is multiplied by a preconditioner $K_t$ that is itself iteratively updated toward $H^{-1}$ |
| **L-BFGS** | Quasi-Newton | Approximates the inverse Hessian from the last *m* parameter and gradient differences |

The optimized parameters for the GHZ and W states were then run on `ibm_brisbane` to check how they hold up on real hardware.

## Hybrid optimization loop

1. A classical optimizer proposes the 27 rotation angles.
2. The angles are loaded into the circuit on a noiseless simulator.
3. The output state is compared with the target state.
4. The cost is fed back to the optimizer, and the loop repeats until the target state is approximated with high fidelity.

## Problem setup

**Targets (3 qubits)**

$$|\Psi_{GHZ}\rangle = \frac{|000\rangle + |111\rangle}{\sqrt{2}}, \qquad |\Psi_W\rangle = \frac{|001\rangle + |010\rangle + |100\rangle}{\sqrt{3}}$$

**Ansatz.** 27 trainable parameters: layers of $R_Z$ and $R_Y$ rotations on each qubit, interleaved with three rings of CNOT gates (`0→1, 1→2, 2→0`, then `0→2, 1→0, 2→1`, then `0→1, 1→2, 2→0` again).

**Cost function**

$$C(\theta) = 1 - \left|\,\mathrm{Re}\,\langle \Psi_{target} | \Psi(\theta) \rangle\,\right|$$

**Gradients and Hessians** are computed with the parameter-shift rule in the Wolfram version, and with PennyLane autodiff (`qml.grad`, `qml.jacobian`) in the Python version.

## Methods

### Iterative Preconditioned Gradient Descent (IPG)

Plain gradient descent with a preconditioner that is refined at every step:

$$x_{t+1} = x_t - \delta_t K_t g_t, \qquad K_{t+1} = K_t - \alpha_t\big((H_t + \beta_t I)K_t - I\big)$$

Convergence conditions on the hyperparameters:

$$\delta_t \le 1, \qquad \beta_t > -\lambda_{\min}(H_t), \qquad \alpha_t < \frac{1}{\lambda_{\max}(H_t) + \beta_t}$$

In the code, $\beta$ and $\alpha$ are derived from the Hessian eigenvalues at the initial point and a small constant $\zeta$, and $K_0$ is a random Gaussian matrix. This avoids the $O(N^3)$ cost of explicitly inverting the Hessian at every iteration and softens the ill-conditioning problem when the eigenvalue spread is large.

### L-BFGS

Two-loop recursion over the last *m* = 10 pairs $(s_i, y_i)$ with $s_t = x_t - x_{t-1}$ and $y_t = \nabla f(x_t) - \nabla f(x_{t-1})$, an initial Hessian scaling $\gamma = \frac{s^\top y}{y^\top y}$, and a backtracking (Armijo) line search.

## Results

All numbers below are from the Wolfram Quantum Framework notebook (see the PDF). Values are read off the published plots, so treat them as approximate.

- **Quasi-Newton and preconditioned methods beat first-order methods.** After 100 iterations on the GHZ target, GD and Adam are still around the 0.03 to 0.05 cost range, while L-BFGS and IPG reach the 10⁻³ range, which is orders of magnitude lower.
- **L-BFGS converges fastest.** It drops to its floor within roughly the first 10 iterations.
- **IPG converges more slowly but ends with lower noise.** It produced states with lower residual error for both the GHZ and W targets.
- **W is harder than GHZ.** L-BFGS best cost is about 0.0024 to 0.0036 for GHZ versus about 0.016 for W. The W state needed longer runs to reach adequate fidelity.

### Hardware validation (`ibm_brisbane`)

The best IPG parameters were decomposed through Qiskit and run on `ibm_brisbane`. For GHZ, the dominant outcomes were `000` and `111` (about 22/47 each, roughly 47% each, against an ideal 50%), with small leakage into other basis states. The W-state run is also included in the notebook and PDF.

## Repository contents

```
.
├── README.md
├── Optimization_of_Quantum_Circuits.ipynb      # PennyLane + Qiskit reimplementation
└── Optimizing_Quantum_Circuits_published.pdf   # Original Wolfram Language write-up with plots
```

- **PDF.** The original Wolfram Language notebook, exported. It contains the derivations, all convergence plots, the optimizer comparison and the IBM hardware histograms.
- **Notebook.** A Python reimplementation: PennyLane for the circuit and autodiff, custom GD / Adam / IPG / L-BFGS optimizers, and Qiskit + `qiskit_ibm_runtime` for running on IBM hardware.

## Getting started

```bash
pip install pennylane pennylane-lightning autograd numpy scipy matplotlib
pip install qiskit qiskit_aer qiskit_ibm_runtime pylatexenc
```

Open `Optimization_of_Quantum_Circuits.ipynb` in Jupyter or Google Colab and run the cells in order. The notebook writes intermediate results to JSON; edit the output paths (currently Google Drive paths) to suit your setup.

To change the target state, edit the `ghz_state` assignment in `cost_function`. It currently points at `w3_state`.

### Running on IBM hardware

Do **not** hard-code your API token. Save it once and load it from the saved account or an environment variable:

```python
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN", overwrite=True)
service = QiskitRuntimeService()
backend = service.backend("ibm_brisbane")
```

Backend availability changes over time, so substitute any backend you have access to.

## Future work

- Extend the methods to full unitary synthesis, especially with unknown initial conditions such as the Quantum Fourier Transform.
- Study matrix-conditioning techniques for better numerical stability across circuit architectures.
- Test noise-aware cost functions and larger qubit counts.

## References

1. D. Srinivasan, K. Chakrabarti, N. Chopra, A. Dutt. *Quantum Circuit Optimization through Iteratively Pre-Conditioned Gradient Descent.* [arXiv:2309.09957](https://arxiv.org/abs/2309.09957)

## Acknowledgements

Research carried out through World Science Scholars.
