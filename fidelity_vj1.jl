### Solve the nearest-neighbor (NN) and long-range (LR) Ising model
### on a finite chain of size N with PBC by exact diagonalization
### and calculate the fidelity (overlap) of the first meson between both models

### export JULIA_NUM_THREADS=8

# include("layout.jl")
# using layout
# using LinearMaps
# using BSON
using Plots
using LinearAlgebra
using SparseArrays
using ArnoldiMethod
# using Base.Threads
LinearAlgebra.BLAS.set_num_threads(1)

sz = [1 0; 0 -1] |> sparse
sx = [0 1; 1 0] |> sparse
si = [1 0; 0 1] |> sparse
ZZ = kron(sz,sz)
XI = kron(sx,si)
IX = kron(si,sx)
ZI = kron(sz,si)
IZ = kron(si,sz)

## chain parameters:
N = 12
d = 2
num_states = 10*N # number of lowest eigenstates/energies

## Ising parameters:
J = 1.0
h = 1.0
g = 3.0
alpha_vals = range(0,3,length=16) # LR param

## output file:
fidelity_file = "fidelity_g3"




### **********************************   functions   *************************************************
function constr_NN_ham_PBC(d,N) # for PBC in NN (as an alternative to the gate application, by explicit matrix construction)
    H = spzeros(d^N,d^N)
    println("construct NN Hamiltonian PBC")
    for k=1:N-1
        idL = sparse(I,d^(k-1),d^(k-1))
        idR = sparse(I,d^(N-k-1),d^(N-k-1))

        auxH2 = -J*kron(idL,kron(ZZ,idR)) # interaction term -J*ZZ
        auxH1 = kron(idL,kron(-h*XI-g*ZI,idR)) # single terms -h*X-g*Z

        H = H+auxH2+auxH1
    end
    ## last periodic term Z_N Z_1:
    H_period = -J*kron(kron(sz,sparse(I,d^(N-2),d^(N-2))),sz)
    H = H + H_period

    ## last single term:
    lastT = -h*kron(sparse(I,d^(N-1),d^(N-1)),sx) - g*kron(sparse(I,d^(N-1),d^(N-1)),sz)
    H = H + lastT
    return H
end

function constr_LR_ham_PBC(d,N,alpha) # for PBC in LR
    H = spzeros(d^N,d^N)
    for j=1:N # single terms
        for i=1:j-1 # 2-site interaction terms
            idL = sparse(I,d^(i-1),d^(i-1))      # Id left of i
            idM = sparse(I,d^(j-i-1),d^(j-i-1))    # Id between i and j
            idR = sparse(I,d^(N-j),d^(N-j))      # Id right of j

            dist = min(abs(i-j), abs(i-(j-N))) # periodic distances on circle
            J_ij = -J/(dist^alpha) # LR weights on circle
            auxH2 = J_ij*kron(kron(kron(kron(idL,sz),idM),sz),idR) # = Z_i Z_j / dist^alpha
            H = H+auxH2
        end
        ## single terms:
        idL = sparse(I,d^(j-1),d^(j-1))
        idR = sparse(I,d^(N-j),d^(N-j))
        auxH1 = kron(kron(idL,-h*sx-g*sz),idR) # = -h*X-g*Z
        H = H+auxH1
    end
    return H
end

"""
Binary `BitArray` representation of the given integer `num`, padded to length `N`.
"""
bit_rep(num::Integer, N::Integer) = BitArray(parse(Bool, i) for i in string(num, base=2, pad=N))

function magnetization(state1, state2=state1)
    M = 0.
    for i in 1:length(state1)
        bstate = bit_rep(i-1,N)
        bstate_M = 0.
        for spin in bstate
            bstate_M += (state1[i]*state2[i] * (spin ? 1 : -1))
        end
        # @assert abs(bstate_M) <= 1
        M += abs(bstate_M)
    end
    return M
end
function overlap(state1, state2=state1)
    M = 0.
    for i in 1:length(state1)
        bstate = bit_rep(i-1,N)
        bstate_M = 0.
        for spin in bstate
            bstate_M += (state1[i]*state2[i] * (spin ? 1 : 1))
        end
        # @assert abs(bstate_M) <= 1
        M += abs(bstate_M)
    end
    return M
end

function eigen_sparse(x, num_eigvec)
    decomp, history = partialschur(x, nev=num_eigvec, which=SR()); # only solve for the ground state
    vals, vecs = partialeigen(decomp);
    return vals, vecs
end

### *******************++++******   fidelity between NN and LR   ***************
## NN:
hamNN_pbc = constr_NN_ham_PBC(d,N)
println("diagonalize NN Hamiltonian PBC")
# @time E_finite_NN_pbc, evecs_NN_pbc = eigen_sparse(hamNN_pbc, num_states)
@time E_finite_NN_pbc, evecs_NN_pbc = eigen(Matrix(hamNN_pbc))



## LR:
fidelity_vals = Array{Float64}(undef, length(alpha_vals),4)
evecs_LR_pbc_N1 = Array{Float64}(undef, 2^N,length(alpha_vals))

Threads.@threads for j = 1:length(alpha_vals)
    alpha = alpha_vals[j]
    hamLR_pbc = constr_LR_ham_PBC(d,N,alpha)
    # @time E_finite_LR_pbc, evecs_LR_pbc = eigen_sparse(hamLR_pbc, num_states)
    E_finite_LR_pbc, evecs_LR_pbc = eigen(Matrix(hamLR_pbc))
    evecs_LR_pbc_N1[:,j] = evecs_LR_pbc[:,N+1]

    fidelity1a=0.0; fidelity1b=0.0; fidelity2a=0.0; fidelity2b=0.0;
    ## loop over N eigenstates of first meson:
    for n = 2:N+1
        fidelity1a += overlap(evecs_NN_pbc[:,n], evecs_LR_pbc[:,n])
        fidelity1b += abs(evecs_NN_pbc[:,n]'*evecs_LR_pbc[:,n])
        for m = 2:N+1
            fidelity2a += overlap(evecs_NN_pbc[:,n], evecs_LR_pbc[:,m])
            fidelity2b += abs(evecs_NN_pbc[:,n]'*evecs_LR_pbc[:,m])
        end
    end
    fidelity_vals[j,1] = fidelity1a/N^2; fidelity_vals[j,2] = fidelity1b/N;
    fidelity_vals[j,3] = fidelity2a/N^3; fidelity_vals[j,4] = fidelity2b/N^2;
end

plotly()
p1 = plot()
plot!(p1, alpha_vals, fidelity_vals[:,1], legend=false)
plot!(p1, alpha_vals, fidelity_vals[:,2], legend=false)

# p2 = plot()
# plot!(p2, alpha_vals, fidelity_vals[:,1], legend=false)
# savefig(string(@__DIR__,"/figures/fidelityv1"))

p3 = plot()
plot!(p3, alpha_vals, fidelity_vals[:,3], legend=false)
plot!(p3, alpha_vals, fidelity_vals[:,4], legend=false)

p4 = plot()
plot!(p4, alpha_vals, fidelity_vals[:,3], legend=false)
savefig(string(@__DIR__,"/figures/fidelityv1(2)"))


### ------------------------------------------   save data   -----------------------
# LR_data = Dict(:E_excitations_LR_obc => E_excitations_LR_obc, :E_excitations_LR_pbc => E_excitations_LR_pbc,
#                :E_ratios_LR_obc => E_ratios_LR_obc, :E_ratios_LR_pbc => E_ratios_LR_pbc,
#                :omega => omega, :chi => susceptibilities,
#                :N=>N, :d=>d, :num_states=>num_states, :J=>J, :h=>h, :g=>g, :alpha=>alpha, :Gamma=>Γ,
#                :info => "N=$N, d=$d, J=$J, h=$h, g=$g, α=$alpha, Γ=$Γ")
# BSON.bson(string(@__DIR__,"/data/"*LR_file*".bson"), LR_data)






println("done: fidelity_vj1.jl")
# show()
;
