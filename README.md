# ⟨⟨ MATRIX FORGE ⟩⟩
### Linear Algebra Workbench — Python Desktop Application

> A full-featured, graphical linear algebra solver built with NumPy · SciPy · SymPy · Matplotlib.  
> Solve any matrix system, decompose it, analyse it, and visualise it — all from one desktop app.

---

## Screenshot Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ⟨⟨ MATRIX FORGE ⟩⟩    NumPy · SciPy · SymPy · Matplotlib              │
├──────────────┬──────────────────────────────────────────────────────────┤
│  INPUT       │  📄 Results  │  🔢 Steps  │  📊 Graphs  │  ↗ Vectors    │
│              │                                                           │
│  Matrix A    │  SOLVE  Ax = b                                            │
│  ─────────   │  ✓ UNIQUE SOLUTION                                        │
│  2  1  -1    │  x = [2.0000  3.0000  -1.0000]                           │
│  -3 -1  2    │  Residual ‖Ax−b‖ = 0.00e+00                              │
│  -2  1  2    │                                                           │
│              │                                                           │
│  Vector b    │                                                           │
│  8  11  3    │                                                           │
│              │                                                           │
│  25+ buttons │                                                           │
└──────────────┴───────────────────────────────────────────────────────────┘
```

---

## Requirements

```bash
pip install numpy scipy matplotlib sympy pillow
```

| Library    | Version  | Purpose                                  |
|------------|----------|------------------------------------------|
| numpy      | ≥ 1.24   | Core matrix operations                   |
| scipy      | ≥ 1.10   | LU, Schur, matrix exp/log/sqrt           |
| matplotlib | ≥ 3.7    | All visualisation & graphs               |
| sympy      | ≥ 1.12   | Exact symbolic RREF, Jordan form, eigenvalues |
| pillow     | ≥ 9.0    | (optional) image utilities               |
| tkinter    | built-in | GUI framework (Python standard library)  |

---

## Installation & Run

```bash
# 1. Clone or download the file
git clone https://github.com/yourname/matrix-forge.git
cd matrix-forge

# 2. Install dependencies
pip install numpy scipy matplotlib sympy pillow

# 3. Launch the app
python linear_algebra_solver.py
```

> **Windows / macOS / Linux** — all supported. Tkinter ships with standard Python.  
> Recommended Python version: **3.9 or later**.

---

## Features

### Systems of Equations

| Operation | Description |
|-----------|-------------|
| **Solve Ax = b** | Detects unique, infinite, and inconsistent solutions. Shows rank, nullity, particular solution, null-space basis, and residual norm. |
| **Gaussian Elimination** | Full step-by-step row-reduction log. Supports partial pivot (default) or full pivot (checkbox). |
| **RREF** | Reduced Row Echelon Form with pivot positions, rank, and nullity. |
| **Least Squares Ax ≈ b** | Solves overdetermined systems. Returns optimal x̂, fitted values, residuals, and R² score. |
| **Iterative Solvers** | Jacobi and Gauss-Seidel side-by-side with convergence plots showing ‖Δx‖ per iteration. |

### Decompositions

| Operation | Description |
|-----------|-------------|
| **LU (PA = LU)** | Partial-pivot LU via SciPy. Returns P, L, U with verification ‖PA−LU‖. |
| **QR (Householder)** | Orthogonal × upper-triangular. Verifies ‖QᵀQ−I‖ and ‖A−QR‖. |
| **SVD** | Full U, Σ, Vᵀ. Shows effective rank, condition number, cumulative energy plot, and ‖A−Aₖ‖ error curve. |
| **Cholesky** | Checks positive definiteness, returns L where A = LLᵀ. |
| **Schur** | A = QTQ\* decomposition via SciPy. |
| **Hessenberg** | Upper Hessenberg reduction A = QHQᵀ. |

### Eigenanalysis

| Operation | Description |
|-----------|-------------|
| **Eigenvalues / Eigenvectors** | Full complex eigenvalue support. Shows trace, determinant, and spectral radius ρ(A). |
| **Hermitian Eigen** | Guaranteed real eigenvalues for symmetric/Hermitian matrices (numpy.eigh). |
| **Jordan Normal Form** | Exact symbolic Jordan form A = PJP⁻¹ via SymPy. |

### Matrix Functions

| Operation | Description |
|-----------|-------------|
| **Matrix Exponential eᴬ** | scipy.linalg.expm — Padé approximation. |
| **Matrix Square Root √A** | scipy.linalg.sqrtm with verification ‖√A·√A−A‖. |
| **Matrix Logarithm ln(A)** | scipy.linalg.logm. |
| **Matrix Power Aᵏ** | Integer power k (dialog prompt). Supports negative powers. |

### Analysis & Algorithms

| Operation | Description |
|-----------|-------------|
| **Matrix Properties** | Shape, rank, nullity, trace, determinant, condition number, Frobenius/spectral/nuclear norms, symmetry, skew-symmetry, orthogonality, positive-definiteness, idempotency, invertibility, inverse, pseudoinverse, null-space basis. |
| **Gram-Schmidt** | Orthonormalises columns of A. Verifies QᵀQ ≈ I. |
| **PCA** | Principal Component Analysis — PC directions, eigenvalues, variance explained (%), cumulative chart, biplot. |
| **Vector Operations** | Dot product, norms, angle (degrees), u+v, u−v, projection of v onto u, cross product (3D). Dedicated Vectors tab with visualisation. |
| **Markov Chain** | Finds steady-state distribution π for row-stochastic A. Shows A^10, A^50, and π convergence plot. Auto-normalises non-stochastic input. |

### Applications

| Operation | Description |
|-----------|-------------|
| **Matrix Arithmetic** | Aᵀ, A², A+B, A−B, A·B, Hadamard A⊙B (element-wise). |
| **Symbolic (SymPy exact)** | Exact rational RREF, exact determinant, exact eigenvalues, exact system solution. |
| **SVD Image Compression** | Low-rank approximation demo at k=1, 2, 5, 10 — visualises compression quality vs. storage ratio. |
| **Quadratic Form xᵀAx** | Evaluates the quadratic form for vector u (entered in the Vector u field). |
| **Linear Regression** | Fits y = β₀ + Xβ via least squares. Returns intercept, coefficients, R². Plots fit and residuals. |

---

## Visualisations (Graphs Tab)

Every operation auto-generates relevant plots. The **"📊 All Graphs"** button produces the full dashboard:

| Plot | What it shows |
|------|---------------|
| Matrix Heatmap | Colour-coded entry values with annotations |
| Column Vectors | 2D arrow plot of all column vectors |
| Singular Values | Bar chart of σ₁ ≥ σ₂ ≥ … ≥ σₙ |
| Linear Transformation | Unit circle and square transformed by A (2×2) |
| Eigenvalue Spectrum | Complex plane scatter with unit circle reference |
| A² or AᵀA | Second-order matrix heatmap |
| Matrix Norms | Frobenius, Spectral, Nuclear, Max |
| Condition Number | Visual bar with actual κ(A) value |

Operation-specific dashboards:
- **SVD**: U, Σ, Vᵀ heatmaps + cumulative energy + ‖A−Aₖ‖ error curve
- **Eigen**: Spectrum + eigenvector heatmap + spectral radius bar
- **PCA**: Variance bar + cumulative + biplot + correlation matrix heatmap
- **Iterative**: Convergence curves (Jacobi vs. Gauss-Seidel)
- **Markov**: Transition heatmap + steady-state bar + convergence trace
- **Regression**: Fit line + residuals scatter

---

## Input Format

### Matrix A (and Matrix B)
```
2  1  -1
-3 -1  2
-2  1  2
```
- Rows separated by **newlines**
- Values separated by **spaces** or **commas**
- Supports Python expressions: `1/3`, `sqrt(2)`, `pi`, `-2.5e-3`
- Lines starting with `#` are ignored (comments)

### Vector b / u / v
```
8  11  3
```
Single line, space- or comma-separated.

### Options
| Control | Description |
|---------|-------------|
| **Decimals** | Spinner (1–12) — controls output precision |
| **Full pivot** | Checkbox — enables complete pivoting in Gaussian elimination |

---

## UI Tabs

| Tab | Contents |
|-----|----------|
| 📄 **Results** | Full computation output with colour-coded formatting. Copy button. |
| 🔢 **Steps** | Step-by-step row operations for Gaussian elimination. |
| 📊 **Graphs** | All Matplotlib visualisations with navigation toolbar (zoom, pan, save). |
| ↗ **Vectors** | Dedicated 2D vector visualisation for u and v operations. |
| 🕓 **History** | Log of every operation run (timestamp + matrix snippet). Clearable. |
| 📖 **Reference** | Built-in cheat sheet — formulas, identities, and algorithm descriptions. |

---

## Keyboard & Toolbar

- **Matplotlib toolbar** (in Graphs tab): zoom, pan, reset view, save PNG
- **Copy button**: copies full Results text to clipboard
- **Export button**: saves Results to a `.txt` file (file dialog)
- **Sample 5×5**: loads a pre-built 5×5 positive-definite test matrix
- **Clear**: wipes all input fields and output panels

---

## Project Structure

```
matrix-forge/
│
├── linear_algebra_solver.py    # Single-file application — run this
└── README.md                   # This file
```

No external config files, no database, no internet connection required.  
Everything runs locally.

---

## How It Works — Architecture

```
MatrixForge (tk.Tk)
│
├── Left Panel
│   ├── Input fields  (Matrix A, B, vectors b/u/v)
│   ├── Options       (precision, full-pivot toggle)
│   └── Button panel  (25 operation buttons, scrollable by category)
│
├── Right Notebook (6 tabs)
│   ├── Results tab   — scrolled text with tagged colour formatting
│   ├── Steps tab     — row-operation log
│   ├── Graphs tab    — embedded Matplotlib FigureCanvasTkAgg
│   ├── Vectors tab   — dedicated 2D vector figure
│   ├── History tab   — HistoryManager log
│   └── Reference tab — static cheat sheet
│
└── Core Modules
    ├── LA_Engine     — all linear algebra (pure NumPy/SciPy/SymPy, no GUI)
    ├── HistoryManager — operation log stack
    └── Plot helpers   — reusable Matplotlib functions
```

---

## Examples

### Solve a 3×3 system
**Matrix A:**
```
2  1  -1
-3 -1  2
-2  1  2
```
**Vector b:** `8  11  3`

**Result:** `x = [2.0  3.0  -1.0]`

---

### Find eigenvalues of a 4×4 matrix
**Matrix A:**
```
4  2  0  1
2  6  1  0
0  1  5  2
1  0  2  7
```
Press **◉ Eigenvalues / Vectors** — get all λᵢ, eigenvectors, spectral radius, and eigenvalue spectrum plot.

---

### PCA on a data matrix
Enter your data matrix (rows = samples, columns = features), press **⊠ PCA**.  
Get: principal components, variance explained per PC, biplot, and correlation matrix heatmap.

---

### Markov chain steady state
Enter a row-stochastic transition matrix (rows sum to 1), press **⊠ Markov Chain**.  
Get: steady-state distribution π, long-run A^50, and convergence plot.

---

## Algorithms Reference

| Algorithm | Library | Method |
|-----------|---------|--------|
| Gaussian elimination | NumPy | Manual RREF with partial/full pivot |
| LU decomposition | SciPy | `scipy.linalg.lu` |
| QR decomposition | NumPy | `numpy.linalg.qr` (Householder) |
| SVD | NumPy | `numpy.linalg.svd` |
| Cholesky | NumPy | `numpy.linalg.cholesky` |
| Schur | SciPy | `scipy.linalg.schur` |
| Eigenvalues | NumPy | `numpy.linalg.eig` / `eigh` |
| Jordan form | SymPy | `Matrix.jordan_form()` |
| Matrix exponential | SciPy | `scipy.linalg.expm` (Padé) |
| Matrix square root | SciPy | `scipy.linalg.sqrtm` |
| Matrix logarithm | SciPy | `scipy.linalg.logm` |
| Least squares | NumPy | `numpy.linalg.lstsq` |
| Null space | SciPy | `scipy.linalg.null_space` |
| Gram-Schmidt | NumPy | Manual iterative projection |
| PCA | NumPy | Covariance → `eigh` |
| Jacobi / Gauss-Seidel | NumPy | Manual iterative |
| Markov steady-state | NumPy | Left eigenvector of Pᵀ for λ=1 |
| Symbolic RREF | SymPy | `Matrix.rref()` |
| Exact eigenvalues | SymPy | `Matrix.eigenvals()` |

---

## Licence

MIT — free to use, modify, and distribute.

---

## Contributing

Pull requests welcome. Suggested additions:
- Sparse matrix support (`scipy.sparse`)
- Power method / QR iteration for large eigenproblems
- Complex matrix input support
- LaTeX export of results
- Dark / light theme toggle
