# IONspectra
calculation of absorption spectra in the quantum Ising model for the estimation of trapped ion quantum simulators; ref.: https://arxiv.org/abs/2107.09071
______________________________________

main files:

- IsingSpectrum_ED_sparse.jl
solves the nearest-neighbor and long-range Ising model on a finite chain of size N with open and periodic boundary conditions using iterative diagonalization methods with sparse data

- fidelity.jl
calculation of fidelity with finite size scaling and fidelity susceptibility 
