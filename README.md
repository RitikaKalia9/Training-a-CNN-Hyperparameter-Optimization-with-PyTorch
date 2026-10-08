# Training a CNN: Hyperparameter Optimization with PyTorch

A hands-on, self-checking lab that builds hyperparameter optimization (HPO) from first principles. Each concept is derived on paper, implemented in plain Python/PyTorch, and **verified numerically** (hand calculation vs. code vs. simulation). It ends with a real random search under successive halving on CIFAR-10.

📓 **Notebook:** [`LCO23393_Ritika_Kalia.ipynb`](./LCO23393_Ritika_Kalia.ipynb)

---

## What's Inside

| Part | Topic | What it shows |
|---|---|---|
| 0 | Setup | Device selection (CUDA/CPU), seeding for reproducibility |
| 1 | Parameters vs. hyperparameters | `SmallCNN` with 10 tensors / **620,362** trainable scalars; 6 hyperparameters live outside the model. The nested HPO problem |
| 2 | Learning-rate stability window | For a quadratic loss, only `0 < η < 2/a` converges; best `η = 1/a`. Verified by simulation and formula |
| 3 | Learning-rate schedules | Step, exponential, cosine annealing, linear warmup, matched exactly against `torch.optim.lr_scheduler` |
| 4 | Batch size | Updates per epoch, measured **1/B** gradient-variance law, linear scaling rule |
| 5 | Grid vs. random search | Grid cost = ∏kⱼ; random search `P = 1 − (1 − p)ⁿ`, confirmed by 20,000-run simulation |
| 6 | Expected Improvement | Exploitation vs. exploration terms of a Bayesian-optimization acquisition function |
| 7 | Successive halving | Rung generation, budget accounting, reduction-factor trade-offs |
| 8 | **Real search on CIFAR-10** | 9 random configs (lr, weight decay, dropout) reduced by successive halving, with no test-set leakage |
| 9 | Putting it together | Summary tables of key numbers, hand-calculation checks, conclusions, viva Q&A |

---

## Key Results

- **Stability window:** with curvature `a = 4`, training converges only for `0 < η < 0.5` (best `η = 0.25`). At `η = 0.5` the iterates bounce between ±1 forever; at `η = 0.6` they diverge.
- **Learning rate matters multiplicatively:** `η = 0.02` needs 56 iterations to reach `|w| ≤ 0.01`; `η = 0.2` needs 3. This is why lr is searched on a **log scale**.
- **One large curvature limits everything:** for `a = 100` the window shrinks to `η < 0.02` and the flat direction needs 459 iterations.
- **Batch size:** gradient variance falls as `1/B` (measured 15.5× for B: 8 → 128, predicted 16×); updates per epoch drop from 391 (B=128) to 98 (B=512).
- **Random vs. grid:** 90 random trials give 99% confidence of hitting a 5% good region, versus 625 trials for a 4-hyperparameter × 5-value grid (6.9× more).
- **Successive halving savings:** 108 vs. 729 epoch-units (n=27, 6.75×) and 405 vs. 6,561 (n=81, 16.2×).
- **CIFAR-10 search (Part 8):** 27 epoch-units instead of 81 (3× saving). Best config: `lr = 2.87e-3`, `weight_decay = 5.30e-4`, `dropout = 0.30` → validation loss **1.2248** after 9 epochs.
  All four configs with `lr ≥ 5e-3` collapsed to chance level (loss ≈ 2.303 = ln 10) and were eliminated at rung 0, which is the stability window showing up on a real network.

> The validation loss is a *selection score*, not a generalisation estimate. The CIFAR-10 test split is **never loaded** in this notebook.

---

## Setup

**Requirements**

- Python 3.9+
- `torch`, `torchvision`, `matplotlib`

```bash
pip install torch torchvision matplotlib
```

**Run**

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
jupyter notebook LCO23393_Ritika_Kalia.ipynb
```

Or open it directly in **Google Colab** and enable a GPU runtime (`Runtime → Change runtime type → GPU`). Everything also runs on CPU; only Part 8 benefits from a GPU.

> The first run of Part 8 downloads CIFAR-10 (~170 MB) into `./data`.

---

## Experimental Protocol (Part 8)

- **Data:** 5,000 training images (indices 0–4,999) and 2,000 validation images (indices 5,000–6,999), both from CIFAR-10's *train* split. Non-overlapping.
- **Model:** `SmallCNN`, three conv blocks followed by two FC layers, trained with Adam.
- **Search space:**
  - `lr` ~ 10^U(−4, −1.5) *(log scale)*
  - `weight_decay` ~ 10^U(−6, −3) *(log scale)*
  - `dropout` ~ U(0, 0.6) *(linear scale)*
- **Successive halving (η_SH = 3):** 9 configs × 1 epoch → best 3 × 3 epochs → best 1 × 9 epochs. Each rung retrains from scratch.
- **Reproducibility:** fixed seeds (`torch.manual_seed(0)`, `random.Random(0)`) give the same nine configurations on every run. Exact losses can still vary with hardware and PyTorch version.

---

## Known Limitations

- Rungs retrain from scratch instead of resuming from checkpoints (matches the budget formula, but is costlier than a checkpointed implementation).
- Single seed and a subsampled dataset, chosen to keep the search fast.
- The selected configuration is not retrained and evaluated on the test set here. The correct next step is to retrain it from scratch, ideally over several seeds, and evaluate on the test set **once**.

---

## Concepts Covered

`hyperparameters` · `learning-rate stability` · `LR schedules` · `batch size & gradient noise` · `grid search` · `random search` · `Bayesian optimization` · `Expected Improvement` · `successive halving` · `Hyperband (discussed)` · `CIFAR-10` · `PyTorch`

---

## Author

**Ritika Kalia**
Computer Science and Engineering, CCET Chandigarh
