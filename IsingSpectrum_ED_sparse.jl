### Solve the nearest-neighbor (NN) and long-range (LR) Ising model
### on a finite chain of size N with OBC and PBC by exact diagonalization

### export JULIA_NUM_THREADS=8

include("layout.jl")
using layout
using LinearMaps
using BSON
using PyPlot
using Base.Threads
# Base.BLAS.set_num_threads(1)

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
num_states = 400 # number of lowest eigenstates/energies

## Ising parameters:
J = 1.0
h = 1.0
g = 3.0
alpha = 3.0 # LR param

## spectroscopy parameters:
Γ = 0.1         # energy resolution width
omega_max = 4.0 # maximal probe frequency in units of m1
omega_numpts = 1001

## params for ground state preparation:
t_ramp = 1.0
dt = 0.01
calculate_adiabatic_gs = false
use_adiabatic_gs = true

## choose what to do:
calculate_NN = false
NN_file = "NN_data_g3"
calculate_LR = true
LR_file = "LR_data_g3_alpha3_adiabatic_tramp1"




### **********************************   functions   *************************************************
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

function constr_NN_ham_OBC(d,N,J,h,g) # for OBC in NN
    H = spzeros(d^N,d^N)
    println("construct NN Hamiltonian OBC")
    for k=1:N-1
        idL = speye(d^(k-1))
        idR = speye(d^(N-k-1))

        auxH2 = -J*kron(idL,kron(ZZ,idR)) # interaction term -J*ZZ
        auxH1 = kron(idL,kron(-h*XI-g*ZI,idR)) # single terms -h*X-g*Z

        H = H+auxH2+auxH1
    end
    ## last single term:
    lastT = -h*kron(speye(d^(N-1)),sx) - g*kron(speye(d^(N-1)),sz)
    H = H + lastT
    return H
end

function constr_NN_ham_PBC(d,N,J,h,g) # for PBC in NN (as an alternative to the gate application, by explicit matrix construction)
    H = spzeros(d^N,d^N)
    println("construct NN Hamiltonian PBC")
    for k=1:N-1
        idL = speye(d^(k-1))
        idR = speye(d^(N-k-1))

        auxH2 = -J*kron(idL,kron(ZZ,idR)) # interaction term -J*ZZ
        auxH1 = kron(idL,kron(-h*XI-g*ZI,idR)) # single terms -h*X-g*Z

        H = H+auxH2+auxH1
    end
    ## last periodic term Z_N Z_1:
    H_period = -J*kron(kron(sz,speye(d^(N-2))),sz)
    H = H + H_period

    ## last single term:
    lastT = -h*kron(speye(d^(N-1)),sx) - g*kron(speye(d^(N-1)),sz)
    H = H + lastT
    return H
end

function constr_LR_ham_OBC(d,N,J,h,g) # for OBC in LR
    H = spzeros(d^N,d^N)
    println("construct LR Hamiltonian OBC")
    for j=1:N # single terms
        for i=1:j-1 # 2-site interaction terms
            idL = speye(d^(i-1))      # Id left of i
            idM = speye(d^(j-i-1))    # Id between i and j
            idR = speye(d^(N-j))      # Id right of j
            J_ij = -J/((j-i)^alpha) # LR weights
            auxH2 = J_ij*kron(kron(kron(kron(idL,sz),idM),sz),idR) # = Z_i Z_j / |i-j|^alpha
            H = H+auxH2
        end
        idL = speye(d^(j-1))
        idR = speye(d^(N-j))
        auxH1 = kron(kron(idL,-h*sx-g*sz),idR) # = -h*X-g*Z
        H = H+auxH1
    end
    return H
end

function constr_LR_ham_PBC(d,N,J,h,g) # for PBC in LR
    H = spzeros(d^N,d^N)
    println("construct LR Hamiltonian PBC")
    for j=1:N # single terms
        for i=1:j-1 # 2-site interaction terms
            idL = speye(d^(i-1))      # Id left of i
            idM = speye(d^(j-i-1))    # Id between i and j
            idR = speye(d^(N-j))      # Id right of j

            dist = min(abs(i-j), abs(i-(j-N))) # periodic distances on circle
            # dist1, dist2 = abs(i-j), abs(i-(j-N))
            # println("i,j,dist = ",i,", ",j,", ",dist)
            # println("i,j,dist1,dist2 = ",i,", ",j,", ",dist1,", ",dist2)
            J_ij = -J/(dist^alpha) # LR weights on circle
            # J_ij = -( J/(dist1^alpha) + J/(dist2^alpha) )
            auxH2 = J_ij*kron(kron(kron(kron(idL,sz),idM),sz),idR) # = Z_i Z_j / dist^alpha
            H = H+auxH2
        end
        ## single terms:
        idL = speye(d^(j-1))
        idR = speye(d^(N-j))
        auxH1 = kron(kron(idL,-h*sx-g*sz),idR) # = -h*X-g*Z
        H = H+auxH1
    end
    return H
end

"""
Binary `BitArray` representation of the given integer `num`, padded to length `N`.
"""
bit_rep(num::Integer, N::Integer) = BitArray(parse(Bool, i) for i in base(2,num,N))

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

function Lorentz(ω, Em, E0, Γ)
    return Γ ./ ((ω-(Em-E0)).^2 + Γ^2)
end


function prepare_ground_state(d,N,J,g)
    polar = zeros(d^N)
    polar[1] = 1.0
    dh = dt/t_ramp

    ## time evolution:
    for i = 1:Int(t_ramp/dt)
        hi = dh*i
        println("hi = ",hi)
        # Ht = constr_NN_ham_PBC(d,N,J,hi,g)
        Ht = constr_LR_ham_PBC(d,N,J,hi,g)
        Ut = expm(-im*dt*Matrix(Ht))
        polar = Ut*polar
    end

    ## cancel phase:
    polar_mat = reshape(polar, Int(d^(N/2)),Int(d^(N/2)))
    polar_tr = trace(polar_mat)
    polar_ph = polar_tr/abs(polar_tr)
    polar_mat = polar_mat/polar_ph
    polar = vec(polar_mat)

    ## save state:
    polar_data = Dict(:polar => polar, :dt => dt, :dh => dh, :t_ramp => t_ramp,
                      :J=>J, :h=>h, :g=>g, :alpha => alpha,
                      :info => "N=$N, d=$d, J=$J, h=$h, g=$g, dt=$dt, dh=$dh, t_ramp=$t_ramp")
    BSON.bson(string(@__DIR__,"/data/adiabaticGS_tramp"*string(t_ramp)*".bson"), polar_data)

    return polar
end




### **********************************   adiabatic ground state   *****************************
if calculate_adiabatic_gs
    polar = prepare_ground_state(d,N,J,g)
end




### **********************************   NN   *************************************************
if calculate_NN
    ### ------------------------------------------   OBC   -------------------------
    ## diagonalization:
    hamNN_obc = constr_NN_ham_OBC(d,N,J,h,g)
    println("diagonalize NN Hamiltonian OBC")
    @time E_finite_NN_obc, evecs_NN_obc = eigs(hamNN_obc, nev=num_states, which=:SR) # finite size energies

    ## finite size energies:
    E_finite_NN_obc = real(E_finite_NN_obc)                               # eigenvalues from ED = finite size energies
    E_excitations_NN_obc = E_finite_NN_obc-E_finite_NN_obc[1]             # mass gaps to ground state
    E_ratios_NN_obc = E_excitations_NN_obc[2:end]/E_excitations_NN_obc[2] # ratio of mass gaps to first one



    ### ------------------------------------------   PBC   -------------------------
    ## diagonalization:
    hamNN_pbc = constr_NN_ham_PBC(d,N,J,h,g)
    println("diagonalize NN Hamiltonian PBC")
    @time E_finite_NN_pbc, evecs_NN_pbc = eigs(hamNN_pbc, nev=num_states, which=:SR)

    ## finite size energies:
    E_finite_NN_pbc = real(E_finite_NN_pbc)
    E_excitations_NN_pbc = E_finite_NN_pbc-E_finite_NN_pbc[1]
    E_ratios_NN_pbc = E_excitations_NN_pbc[2:end]/E_excitations_NN_pbc[2]


    ### ------------------------------------------   absorption rate   -----------------------
    ground = evecs_NN_pbc[:,1]
    omega = linspace(0, omega_max, omega_numpts)*E_excitations_NN_pbc[2] # 0 ... 5*m1
    susceptibilities = Array{Complex128}(length(omega))

    @time @threads for i=1:length(omega)
        Core.println("i = ",i)
        ω_i = omega[i]
        Imχ = 0.0
        for m = 1:length(E_excitations_NN_pbc)
            L_mn = Lorentz(ω_i, E_finite_NN_pbc[m], E_finite_NN_pbc[1], Γ) - Lorentz(ω_i, -E_finite_NN_pbc[m], -E_finite_NN_pbc[1], Γ)
            Imχ += magnetization(evecs_NN_pbc[:,m],ground)^2 * L_mn
        end
        susceptibilities[i] = pi*Imχ
        # println("Imχ = ",Imχ)
    end


    ### ------------------------------------------   save data   -----------------------
    NN_data = Dict(:E_excitations_NN_obc => E_excitations_NN_obc, :E_excitations_NN_pbc => E_excitations_NN_pbc,
                   :E_ratios_NN_obc => E_ratios_NN_obc, :E_ratios_NN_pbc => E_ratios_NN_pbc,
                   :omega => omega, :chi => susceptibilities,
                   :N=>N, :d=>d, :num_states=>num_states, :J=>J, :h=>h, :g=>g, :Gamma=>Γ,
                   :info => "N=$N, d=$d, J=$J, h=$h, g=$g, Γ=$Γ")
    BSON.bson(string(@__DIR__,"/data/"*NN_file*".bson"), NN_data)
end



### **********************************   LR   *************************************************
if calculate_LR
    ### ------------------------------------------   OBC   -------------------------
    println("\n")
    ## diagonalization:
    hamLR_obc = constr_LR_ham_OBC(d,N,J,h,g)
    println("diagonalize LR Hamiltonian OBC")
    @time E_finite_LR_obc, evecs_LR_obc = eigs(hamLR_obc, nev=num_states, which=:SR) # finite size energies

    ## finite size energies:
    E_finite_LR_obc = real(E_finite_LR_obc)                               # eigenvalues from ED = finite size energies
    E_excitations_LR_obc = E_finite_LR_obc-E_finite_LR_obc[1]             # mass gaps to ground state
    E_ratios_LR_obc = E_excitations_LR_obc[2:end]/E_excitations_LR_obc[2] # ratio of mass gaps to first one



    ### ------------------------------------------   PBC   -------------------------
    ## diagonalization:
    hamLR_pbc = constr_LR_ham_PBC(d,N,J,h,g)
    println("diagonalize LR Hamiltonian PBC")
    @time E_finite_LR_pbc, evecs_LR_pbc = eigs(hamLR_pbc, nev=num_states, which=:SR) # finite size energies

    ## finite size energies:
    E_finite_LR_pbc = real(E_finite_LR_pbc)                               # eigenvalues from ED = finite size energies
    E_excitations_LR_pbc = E_finite_LR_pbc-E_finite_LR_pbc[1]             # mass gaps to ground state
    E_ratios_LR_pbc = E_excitations_LR_pbc[2:end]/E_excitations_LR_pbc[2] # ratio of mass gaps to first one


    ### ------------------------------------------   absorption rate   -----------------------
    if use_adiabatic_gs
        polar_data =  BSON.load(string(@__DIR__,"/data/adiabaticGS_tramp"*string(t_ramp)*".bson"))
        ground = polar_data[:polar]
        println(polar_data[:info])
    else
        ground = evecs_LR_pbc[:,1]
    end
    omega = linspace(0, omega_max, omega_numpts)*E_excitations_LR_pbc[2] # 0 ... 5*m1
    susceptibilities = Array{Complex128}(length(omega))

    @time @threads for i=1:length(omega)
        Core.println("i = ",i)
        ω_i = omega[i]
        Imχ = 0.0
        for m = 1:length(E_excitations_LR_pbc)
            L_mn = Lorentz(ω_i, E_finite_LR_pbc[m], E_finite_LR_pbc[1], Γ) - Lorentz(ω_i, -E_finite_LR_pbc[m], -E_finite_LR_pbc[1], Γ)
            Imχ += magnetization(evecs_LR_pbc[:,m],ground)^2 * L_mn
        end
        susceptibilities[i] = pi*Imχ
        # println("Imχ = ",Imχ)
    end


    ### ------------------------------------------   save data   -----------------------
    LR_data = Dict(:E_excitations_LR_obc => E_excitations_LR_obc, :E_excitations_LR_pbc => E_excitations_LR_pbc,
                   :E_ratios_LR_obc => E_ratios_LR_obc, :E_ratios_LR_pbc => E_ratios_LR_pbc,
                   :omega => omega, :chi => susceptibilities,
                   :N=>N, :d=>d, :num_states=>num_states, :J=>J, :h=>h, :g=>g, :alpha=>alpha, :Gamma=>Γ,
                   :info => "N=$N, d=$d, J=$J, h=$h, g=$g, α=$alpha, Γ=$Γ")
    BSON.bson(string(@__DIR__,"/data/"*LR_file*".bson"), LR_data)
end








# ## other method
# susceptibilities2 = Array{Complex128}(length(omega))
# for k=1:length(omega)
#     println("k = ",k)
#     ω_k = omega[k]
#     Imχ = 0.0
#     for m = 1:length(E_excitations_NN_pbc)
#         for i=1:N
#             idL = speye(d^(i-1))
#             idR = speye(d^(N-i))
#             O_Zi = kron(kron(idL,sz),idR)
#             Imχ += abs2(ground' * O_Zi * evecs_NN_pbc[:,m]) * Lorentz(ω_k, E_finite_NN_pbc[m], E_finite_NN_pbc[1], Γ)
#         end
#     end
#     susceptibilities2[k] = Imχ
#     println("Imχ = ",Imχ)
# end
#
# println("M = ",magnetization(ground))
# ground = evecs_NN_pbc[:,1]
# D,U = eig(Matrix(hamNN_obc))
# ground = U[:,1]
# M = 0.0
# for i = 1:N
#     idL = speye(d^(i-1))
#     idR = speye(d^(N-i))
#     O_Zi = kron(kron(idL,sz),idR)
#     # O_Zi = U'*O_Zi*U
#     println(i,", ",ground' * O_Zi * ground)
#     M += abs(ground' * O_Zi * ground)
# end
# println("M = ",M)
# println("M/N = ",M/N)




println("done: IsingSpectrum_ED_sparse.jl")
show()
;
