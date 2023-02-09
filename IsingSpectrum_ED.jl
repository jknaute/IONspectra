### Solve the nearest-neighbor (NN) and long-range (LR) Ising model
### on a finite chain of size N with OBC and PBC by exact diagonalization

include("layout.jl")
using layout
using LinearMaps
using PyPlot
using Base.Threads
# Base.BLAS.set_num_threads(1)

sz = [1 0; 0 -1]
sx = [0 1; 1 0]
si = [1 0; 0 1]
ZZ = kron(sz,sz)
XI = kron(sx,si)
IX = kron(si,sx)
ZI = kron(sz,si)
IZ = kron(si,sz)

## chain parameters:
N = 12
d = 2
num_states = 100 # number of lowest eigenstates/energies

## Ising parameters:
J = 1.0
h = 1.0
g = 3.0
alpha = 3.0 # LR param



function build_ising_gate(J,h,g) # for PBC in NN
    gate = -(J*ZZ + h/2*(XI+IX) + g/2*(ZI+IZ))
    return gate
end

function apply_spin_hamiltonian{T}(v::AbstractVector{T}, ham::Matrix{T}) # for PBC in NN
    D = length(v)
    N = convert(Int64, log2(D))
    tensor_shp = tuple(fill(2, N)...)  # tensor_shp =(2,2,2, ..., 2)
    cyclperm = tuple(2:N..., 1)  #tuple for cyclic permutation cyclperm = (2, 3, ... N, 1)
    out = zeros(tensor_shp)
    for n=1:N
        v, out = reshape(v, (4, 2^(N-2))), reshape(out, (4, 2^(N-2)))
        out += ham*v
        v, out = reshape(v, tensor_shp), reshape(out, tensor_shp)
        v, out = permutedims(v, cyclperm), permutedims(out, cyclperm)
    end
    out = reshape(out, D)
    return out
end

function constr_NN_ham_OBC(d,N) # for OBC in NN
    H = zeros(d^N,d^N)
    println("construct NN Hamiltonian")
    for k=1:N-1
        idL = eye(d^(k-1))
        idR = eye(d^(N-k-1))

        auxH2 = -J*kron(idL,kron(ZZ,idR)) # interaction term -J*ZZ
        auxH1 = kron(idL,kron(-h*XI-g*ZI,idR)) # single terms -h*X-g*Z

        H = H+auxH2+auxH1
    end
    ## last single term:
    lastT = -h*kron(eye(d^(N-1)),sx) - g*kron(eye(d^(N-1)),sz)
    H = H + lastT
    return H
end

function constr_NN_ham_PBC(d,N) # for PBC in NN (as an alternative to the gate application by explicit matrix construction)
    H = zeros(d^N,d^N)
    println("construct NN Hamiltonian")
    for k=1:N-1
        idL = eye(d^(k-1))
        idR = eye(d^(N-k-1))

        auxH2 = -J*kron(idL,kron(ZZ,idR)) # interaction term -J*ZZ
        auxH1 = kron(idL,kron(-h*XI-g*ZI,idR)) # single terms -h*X-g*Z

        H = H+auxH2+auxH1
    end
    ## last periodic term Z_N Z_1:
    H_period = -J*kron(kron(sz,eye(d^(N-2))),sz)
    # p = d
    # q = d^(N-1)
    # r = p*q
    # Ir = eye(r)
    # S = []
    # for k=1:q
    #     S = cat(1, S,Ir[k:q:r,:])
    # end
    # H_period = S*H_period
    # H_period = H_period*S'
    H = H + H_period

    ## last single term:
    lastT = -h*kron(eye(d^(N-1)),sx) - g*kron(eye(d^(N-1)),sz)
    H = H + lastT
    return H
end

function constr_LR_ham_OBC(d,N) # for OBC in LR
    H = zeros(d^N,d^N)
    println("construct LR Hamiltonian OBC")
    for j=1:N # single terms
        for i=1:j-1 # 2-site interaction terms
            idL = eye(d^(i-1))      # Id left of i
            idM = eye(d^(j-i-1))    # Id between i and j
            idR = eye(d^(N-j))      # Id right of j
            J_ij = -J/((j-i)^alpha) # LR weights
            auxH2 = J_ij*kron(kron(kron(kron(idL,sz),idM),sz),idR) # = Z_i Z_j / |i-j|^alpha
            H = H+auxH2
        end
        idL = eye(d^(j-1))
        idR = eye(d^(N-j))
        auxH1 = kron(kron(idL,-h*sx-g*sz),idR) # = -h*X-g*Z
        H = H+auxH1
    end
    return H
end

function constr_LR_ham_PBC(d,N) # for PBC in LR
    H = zeros(d^N,d^N)
    println("construct LR Hamiltonian PBC")
    for j=1:N # single terms
        for i=1:j-1 # 2-site interaction terms
            idL = eye(d^(i-1))      # Id left of i
            idM = eye(d^(j-i-1))    # Id between i and j
            idR = eye(d^(N-j))      # Id right of j

            dist = min(abs(i-j), abs(i-(j-N))) # periodic distances on circle
            println("i,j,dist = ",i,", ",j,", ",dist)
            J_ij = -J/(dist^alpha) # LR weights on circle
            auxH2 = J_ij*kron(kron(kron(kron(idL,sz),idM),sz),idR) # = Z_i Z_j / dist^alpha
            H = H+auxH2
        end
        ## single terms:
        idL = eye(d^(j-1))
        idR = eye(d^(N-j))
        auxH1 = kron(kron(idL,-h*sx-g*sz),idR) # = -h*X-g*Z
        H = H+auxH1
    end
    println("construct pbc")
    ## consider additional term in periodicity Z_{N+1}=Z_1 separately:
    ##     H ~ H + (Z_2+...+Z_N)Z_1
    ## H_period := (Z_2+...+Z_N)Z_1 = S*[Z_1(Z_2+...+Z_N)]*S'
    ## where S is a permutation matrix, see https://en.wikipedia.org/wiki/Kronecker_product ### <== !!! NOT NECESSARY  !!!
    j = 1
    H_period = Array{Float64}(d^N,d^N)
    for i=2:N
        idM = eye(d^(i-j-1))    # Id between j and i
        idR = eye(d^(N-i))      # Id right of i

        dist = min(abs(i-j), abs(j-(i-N))) # periodic distances on circle
        println("i,j,dist = ",i,", ",j,", ",dist)
        J_ij = -J/(dist^alpha) # LR weights on circle
        auxH2 = J_ij*kron(kron(kron(sz,idM),sz),idR) # = Z_1 Z_i / dist^alpha
        H_period = H_period+auxH2
    end
    # p = d
    # q = d^(N-1)
    # r = p*q
    # Ir = eye(r)
    # S = []
    # for k=1:q
    #     S = cat(1, S,Ir[k:q:r,:])
    # end
    # H_period = S*H_period
    # H_period = H_period*S'

    H = H + H_period
    return H
end



### **********************************   NN   *************************************************
### ------------------------------------------   OBC   -------------------------
## diagonalization:
hamNN_obc = constr_NN_ham_OBC(d,N)
println("diagonalize NN Hamiltonian")
apply_hamNN_obc(v) = hamNN_obc*v
hamNN_obc_linmap = LinearMap{Float64}(apply_hamNN_obc, d^N, ishermitian=true)
@time E_finite_NN_obc, evecs = eigs(hamNN_obc, nev=num_states, which=:SR) # finite size energies

## finite size energies:
E_finite_NN_obc = real(E_finite_NN_obc)                               # eigenvalues from ED = finite size energies
E_excitations_NN_obc = E_finite_NN_obc-E_finite_NN_obc[1]             # mass gaps to ground state
E_ratios_NN_obc = E_excitations_NN_obc[2:end]/E_excitations_NN_obc[2] # ratio of mass gaps to first one



### ------------------------------------------   PBC   -------------------------
ising_h = build_ising_gate(J,h,g)
apply_ising_hamiltonian(v) = apply_spin_hamiltonian(v, ising_h)
Hmap = LinearMap(apply_ising_hamiltonian, 2^N)
@time E_finite_NN_pbc, evecs = eigs(Hmap, nev=num_states, which=:SR)

## finite size energies:
E_finite_NN_pbc = real(E_finite_NN_pbc)                               # eigenvalues from ED = finite size energies
E_excitations_NN_pbc = E_finite_NN_pbc-E_finite_NN_pbc[1]             # mass gaps to ground state
E_ratios_NN_pbc = E_excitations_NN_pbc[2:end]/E_excitations_NN_pbc[2] # ratio of mass gaps to first one

## comparison with explicit matrix construction:
hamNN_pbc = constr_NN_ham_PBC(d,N)
@time E_finite_NN_pbc_explicit, evecs = eigs(hamNN_pbc, nev=num_states, which=:SR)
E_finite_NN_pbc_explicit = real(E_finite_NN_pbc_explicit)
E_excitations_NN_pbc_explicit = E_finite_NN_pbc_explicit-E_finite_NN_pbc_explicit[1]
E_ratios_NN_pbc_explicit = E_excitations_NN_pbc_explicit[2:end]/E_excitations_NN_pbc_explicit[2]


### ------------------------------------------   Plots   -----------------------
figure(1)
E8_masses = [1., 1.61803, 1.98904, 2.40487, 2.9563, 3.21834, 3.89116, 4.78339] # analytical E8 spectrum (m_n/m_1)
for i in 1:8 axhline(E8_masses[i], ls="--", c="grey",zorder=-1) end
plot(1:num_states-1,E_ratios_NN_obc,ls="",marker="o", label="NN obc")
plot(1:num_states-1,E_ratios_NN_pbc,ls="",marker="v", label="NN pbc")
plot(1:num_states-1,E_ratios_NN_pbc_explicit,ls="",marker="^", label="NN pbc explicit")

xlabel("\$n\$ (excited state)")
ylabel("\$m_n/m_1\$")
legend(loc = "best", numpoints=3, frameon=0, fancybox=0, columnspacing=1, title="\$N = $N, g=$g\$")
layout.nice_ticks()
savefig(string(@__DIR__,"/figures/IsingSpectrum_NN_ED.pdf"))




### **********************************   LR   *************************************************
### ------------------------------------------   OBC   -------------------------
## diagonalization:
hamLR_obc = constr_LR_ham_OBC(d,N)
println("diagonalize LR Hamiltonian")
apply_hamLR_obc(v) = hamLR_obc*v
hamLR_obc_linmap = LinearMap{Float64}(apply_hamLR_obc, d^N, ishermitian=true)
@time E_finite_LR_obc, evecs = eigs(hamLR_obc, nev=num_states, which=:SR) # finite size energies

## finite size energies:
E_finite_LR_obc = real(E_finite_LR_obc)                               # eigenvalues from ED = finite size energies
E_excitations_LR_obc = E_finite_LR_obc-E_finite_LR_obc[1]             # mass gaps to ground state
E_ratios_LR_obc = E_excitations_LR_obc[2:end]/E_excitations_LR_obc[2] # ratio of mass gaps to first one



### ------------------------------------------   PBC   -------------------------
## diagonalization:
@time hamLR_pbc = constr_LR_ham_PBC(d,N)
println("diagonalize LR Hamiltonian PBC")
@time E_finite_LR_pbc, evecs = eigs(hamLR_pbc, nev=num_states, which=:SR) # finite size energies

## finite size energies:
E_finite_LR_pbc = real(E_finite_LR_pbc)                               # eigenvalues from ED = finite size energies
E_excitations_LR_pbc = E_finite_LR_pbc-E_finite_LR_pbc[1]             # mass gaps to ground state
E_ratios_LR_pbc = E_excitations_LR_pbc[2:end]/E_excitations_LR_pbc[2] # ratio of mass gaps to first one



### ------------------------------------------   Plots   -----------------------
figure(2)
E8_masses = [1., 1.61803, 1.98904, 2.40487, 2.9563, 3.21834, 3.89116, 4.78339] # analytical E8 spectrum (m_n/m_1)
for i in 1:8 axhline(E8_masses[i], ls="--", c="grey",zorder=-1) end
plot(1:num_states-1,E_ratios_LR_obc,ls="",marker="o", label="LR obc")
plot(1:num_states-1,E_ratios_LR_pbc,ls="",marker="v", label="LR pbc")

xlabel("\$n\$ (excited state)")
ylabel("\$m_n/m_1\$")
legend(loc = "best", numpoints=3, frameon=0, fancybox=0, columnspacing=1, title="\$N = $N, g=$g, \\alpha=$alpha \$")
layout.nice_ticks()
savefig(string(@__DIR__,"/figures/IsingSpectrum_LR_ED.pdf"))







println("done: IsingSpectrum_ED.jl")
show()
;
