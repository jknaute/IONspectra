### Solve the nearest-neighbor (NN) and long-range (LR) Ising model
### on a finite chain of size N with PBC by exact diagonalization
### and calculate the fidelity (absolute overlap) of the first excited state
### (as a proxy for the of the first meson with N states) between both models

### export JULIA_NUM_THREADS=8

include("layout.jl")
using layout
# using LinearMaps
using BSON
using PyPlot
using Optim
using NLSolversBase
# using LsqFit # Pkg.checkout("LsqFit.jl")
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
N_vals = [21]
d = 2
num_states = 2 # number of lowest eigenstates/energies

## Ising parameters:
J = 1.0
h = 1.0
g = 3.0

## choose what to do:
fidelity_file = "fidelity_g3_sparse"
alpha_vals = linspace(0,3,31)         # for fidelity calculation
calculate_fidelity = false
plot_fidelity = false

scaling_file = "scaling_g3_sparse_power01"
power = 0.1
do_finite_size_scaling = false

fidelity_suscep_file = "fidelity_suscep_g3_sparse"
alpha_vals_suscep = linspace(0,3,301) # more finegrained for fidelity susceptibility
calculate_fidelity_suscep = false
plot_fidelity_suscep = true




### ************************************************   functions   *************
function constr_NN_ham_PBC(d,N) # for PBC in NN
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

function constr_LR_ham_PBC(d,N,alpha) # for PBC in LR
    H = spzeros(d^N,d^N)
    for j=1:N # single terms
        for i=1:j-1 # 2-site interaction terms
            idL = speye(d^(i-1))      # Id left of i
            idM = speye(d^(j-i-1))    # Id between i and j
            idR = speye(d^(N-j))      # Id right of j

            dist = min(abs(i-j), abs(i-(j-N))) # periodic distances on circle
            J_ij = -J/(dist^alpha) # LR weights on circle
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



### *******************++++******   fidelity between NN and LR   ***************
if calculate_fidelity
    fidelity_vals = Array{Float64}(length(alpha_vals), length(N_vals))
    # evecs_LR_pbc_N1 = Array{Float64}(2^N,length(alpha_vals))

    for i = 1:length(N_vals) # N-loop
        N = N_vals[i]
        println("N = ",N)

        ## NN:
        hamNN_pbc = constr_NN_ham_PBC(d,N)
        println("diagonalize NN Hamiltonian PBC")
        @time E_finite_NN_pbc, evecs_NN_pbc = eigs(hamNN_pbc, nev=num_states, which=:SR)
        # @time E_finite_NN_pbc, evecs_NN_pbc = eig(Matrix(hamNN_pbc))

        ## LR:
        for j = 1:length(alpha_vals) # α-loop
            alpha = alpha_vals[j]
            hamLR_pbc = constr_LR_ham_PBC(d,N,alpha)
            @time E_finite_LR_pbc, evecs_LR_pbc = eigs(hamLR_pbc, nev=num_states, which=:SR)
            # E_finite_LR_pbc, evecs_LR_pbc = eig(Matrix(hamLR_pbc))
            # evecs_LR_pbc_N1[:,j] = evecs_LR_pbc[:,N+1]

            # fidelity_vals[j,1] = overlap(evecs_NN_pbc[:,2], evecs_LR_pbc[:,2])/N
            fidelity_vals[j,i] = abs(evecs_NN_pbc[:,2]'*evecs_LR_pbc[:,2])

            # ## loop over N eigenstates of first meson:
            # fidelity1a=0.0; fidelity1b=0.0; fidelity2a=0.0; fidelity2b=0.0;
            # for n = 2:N+1
            #     fidelity1a += overlap(evecs_NN_pbc[:,n], evecs_LR_pbc[:,n])
            #     fidelity1b += abs(evecs_NN_pbc[:,n]'*evecs_LR_pbc[:,n])
            #     for m = 2:N+1
            #         fidelity2a += overlap(evecs_NN_pbc[:,n], evecs_LR_pbc[:,m])
            #         fidelity2b += abs(evecs_NN_pbc[:,n]'*evecs_LR_pbc[:,m])
            #     end
            # end
            # fidelity_vals[j,1] = fidelity1a/N; fidelity_vals[j,2] = fidelity1b;
            # fidelity_vals[j,3] = fidelity2a/N^3; fidelity_vals[j,4] = fidelity2b/N^2;
        end
    end

    ## ------------------------------------------   save data   -----------------------
    fidelity_data = Dict(:N => N_vals, :alpha => alpha_vals, :fidelity => fidelity_vals,
                   :d=>d, :J=>J, :h=>h, :g=>g,
                   :info => "N=$N_vals, d=$d, J=$J, h=$h, g=$g, α=$alpha_vals")
    BSON.bson(string(@__DIR__,"/data/"*fidelity_file*".bson"), fidelity_data)
end



###--- PLOTS:
if plot_fidelity
    fidelity_data =  BSON.load(string(@__DIR__,"/data/"*fidelity_file*".bson"))
    fidelity_vals =  fidelity_data[:fidelity]
    N_vals        = fidelity_data[:N]
    alpha_vals    = fidelity_data[:alpha]

    ## power law scaling expectations:
    scaling_data_power1 =  BSON.load(string(@__DIR__,"/data/scaling_g3_sparse_power1.bson"))
    fpower1_infty =  scaling_data_power1[:fpower_infty]
    scaling_data_power2 =  BSON.load(string(@__DIR__,"/data/scaling_g3_sparse_power2.bson"))
    fpower2_infty =  scaling_data_power2[:fpower_infty]
    scaling_data_power15 =  BSON.load(string(@__DIR__,"/data/scaling_g3_sparse_power15.bson"))
    fpower15_infty =  scaling_data_power15[:fpower_infty]
    scaling_data_power01 =  BSON.load(string(@__DIR__,"/data/scaling_g3_sparse_power01.bson"))
    fpower01_infty =  scaling_data_power01[:fpower_infty]
    scaling_data_free1_NM =  BSON.load(string(@__DIR__,"/data/scaling_g3_sparse_free1_NM.bson"))
    fpower_free1_NM =  scaling_data_free1_NM[:fpower_infty]
    scaling_data_free1_LBFGS =  BSON.load(string(@__DIR__,"/data/scaling_g3_sparse_free1_LBFGS.bson"))
    fpower_free1_LBFGS =  scaling_data_free1_LBFGS[:fpower_infty]

    figure(1) # F(α)
    for i = 2:length(N_vals) # N-loop
        N = N_vals[i]
        plot(alpha_vals, fidelity_vals[:,i], label="\$N = $N\$")
    end
    # plot(alpha_vals, fidelity_vals[:,2])
    axhline(1.0, ls="--", c="grey",zorder=-1)
    xlim(0,3)
    # ylim(0.9,1.01)
    xlabel("\$\\alpha\$")
    ylabel("\$F\\, (\\alpha)\$")
    legend(loc = "lower right", numpoints=3, frameon=1, facecolor="white", fancybox=1, columnspacing=1, title="\$g=3.0\$\n NN + LR pbc")
    layout.nice_ticks()
    savefig(string(@__DIR__,"/figures/fidelity.pdf"))


    figure(2) # log(F)/N
    for i = 2:length(N_vals) # N-loop
        N = N_vals[i]
        plot(alpha_vals, log.(fidelity_vals[:,i])/N, label="\$N = $N\$")
    end
    axhline(0.0, ls="--", c="grey",zorder=-1)
    xlim(0,3)
    ax = subplot(111)
    ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-2,1), useOffset=true)
    xlabel("\$\\alpha\$")
    ylabel("\$\\log[F\\, (\\alpha)] / N\$")
    legend(loc = "lower right", numpoints=3, frameon=1, facecolor="white", fancybox=1, columnspacing=1, title="\$g=3.0\$\n NN + LR pbc")
    layout.nice_ticks()
    savefig(string(@__DIR__,"/figures/fidelity_log.pdf"))


    figure(3) # F^(1/N)
    for i = 2:length(N_vals) # N-loop
        N = N_vals[i]
        plot(alpha_vals, fidelity_vals[:,i].^(1.0/N), label="\$N = $N\$")
    end
    # plot(alpha_vals, fpower01_infty, c="k", label="b=1") # label="\$N = \\infty\$"
    # plot(alpha_vals, fpower2_infty, c="grey", label="b=2")
    # plot(alpha_vals, fpower_free1_NM, c="k", ls="--", label="b=free NM")
    plot(alpha_vals, fpower_free1_LBFGS, c="k", ls="--", zorder=-2, label="\$N = \\infty\$") # "b=free LBFGS")
    # fill_between(alpha_vals, fpower_free1_NM, fpower_free1_LBFGS, color="k", alpha=1.0, label="\$N = \\infty\$")
    # fill_between(alpha_vals, fpower1_infty, fpower2_infty, color="C7", alpha=0.6, label="\$N = \\infty\$")
    axhline(1.0, ls="--", c="grey",zorder=-1)
    xlim(0,3)
    ylim(0.988,1.0005)
    # ax = subplot(111)
    # ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-3,1), useOffset=true)
    xlabel("\$\\alpha\$")
    ylabel("\$F\\, (\\alpha)^{1/N}\$")
    legend(loc = "lower right", numpoints=3, frameon=1, facecolor="white", fancybox=1, columnspacing=1, title="\$g=3.0\$\n NN + LR pbc")
    layout.nice_ticks()
    savefig(string(@__DIR__,"/figures/fidelity_power.pdf"))
end




### *******************++++**********   finite size scaling   ******************
if do_finite_size_scaling
    fidelity_data =  BSON.load(string(@__DIR__,"/data/"*fidelity_file*".bson"))
    fidelity_vals =  fidelity_data[:fidelity]
    N_vals = fidelity_data[:N]
    alpha_vals = fidelity_data[:alpha]

    f_extrapolated = Array{Float64}(length(alpha_vals))

    for i = 1:length(alpha_vals) # α-loop
        alpha = alpha_vals[i]

        ## chi^2 from fit ansatz:
        ## fit also power-law decay of N:
        # f(x) = sum(( (fidelity_vals[i,:].^(1.0 ./ N_vals)) - (x[1] + x[2] ./ (N_vals.^x[3])) ).^2) / length(N_vals)
        # model(x,p) = p[1] + p[2] ./ (x.^p[3])
        ## keep fixed power in N:
        f(x) = sum(( (fidelity_vals[i,:].^(1.0 ./ N_vals)) - (x[1] + x[2] ./ (N_vals.^power)) ).^2) / length(N_vals)

        ## fit type 1:
        x0 = [1.0,1.0,1.0]
        min_f = optimize(f,x0,LBFGS(),Optim.Options(g_tol=1e-12))
        params_f,minval_f = Optim.minimizer(min_f),Optim.minimum(min_f)
        f_extrapolated[i] = params_f[1]
        # f_fitted = params_f[1] + params_f[2] ./ (N_vals.^params_f[3])
        f_fitted = params_f[1] + params_f[2] ./ (N_vals.^power)
        println("α, fit: ",alpha,", ",params_f,", ",minval_f)

        ## fit type 2:
        # p0 = [1.0, 1.0, 1.0]
        # lb = [0.0, 0.0, 0.0]
        # ub = [1.0, 1.0, 3.0]
        # xdata = N_vals
        # ydata = fidelity_vals[i,:].^(1.0 ./ N_vals)
        # fit2 = curve_fit(model,xdata,ydata,p0, lower=lb, upper=ub)
        # params_f = fit2.param
        # f_extrapolated[i] = params_f[1]
        # f_fitted = model(N_vals,params_f)
        # println("α, fit2: ",alpha,", ",params_f,", ",f(params_f))
        # # println(fit2.resid)
        # # println(confidence_interval(fit2, 0.1))
        # # println(LsqFit.stderror(fit2))

        ## plot:
        figure(4)
        plot(N_vals, fidelity_vals[i,:].^(1.0 ./ N_vals), ls="", marker="s", label="\$\\alpha = $alpha\$")
        plot(N_vals, f_fitted, ls=":", c="k")

        figure(5)
        plot(alpha, params_f[1], ls="", marker="s", c="k")
    end

    ## save:
    scaling_data = Dict(:N => N_vals, :alpha => alpha_vals, :fpower_infty => f_extrapolated,
                   :d=>d, :J=>J, :h=>h, :g=>g,
                   :info => "N=$N_vals, d=$d, J=$J, h=$h, g=$g, α=$alpha_vals")
    BSON.bson(string(@__DIR__,"/data/"*scaling_file*".bson"), scaling_data)
end




### ********************************   fidelity susceptibility   ***************
if calculate_fidelity_suscep
    fidelity_suscep_vals = Array{Float64}(length(alpha_vals_suscep)-1, length(N_vals))

    for i = 1:length(N_vals) # N-loop
        N = N_vals[i]
        println("N = ",N)

        exc_α = Array{Float64}(d^N)  # = |phi_1(α)>
        exc_dα = Array{Float64}(d^N) # = |phi_1(α+dα)>

        ## α=0:
        hamLR_pbc = constr_LR_ham_PBC(d,N,0.0)
        @time E_finite_LR_pbc, evecs_LR_pbc = eigs(hamLR_pbc, nev=num_states, which=:SR)
        exc_α = evecs_LR_pbc[:,2]

        for j = 2:length(alpha_vals_suscep) # α-loop
            alpha = alpha_vals_suscep[j]
            hamLR_pbc = constr_LR_ham_PBC(d,N,alpha)
            @time E_finite_LR_pbc, evecs_LR_pbc = eigs(hamLR_pbc, nev=num_states, which=:SR)
            exc_dα = evecs_LR_pbc[:,2]

            fidelity_suscep_vals[j-1,i] = abs(exc_α' * exc_dα)
            exc_α = deepcopy(exc_dα)
        end
    end

    ## ------------------------------------------   save data   -----------------------
    fidelity_suscep_data = Dict(:N => N_vals, :alpha => alpha_vals_suscep, :fidelity_suscep => fidelity_suscep_vals,
                   :d=>d, :J=>J, :h=>h, :g=>g,
                   :info => "N=$N_vals, d=$d, J=$J, h=$h, g=$g, α=$alpha_vals_suscep")
    BSON.bson(string(@__DIR__,"/data/"*fidelity_suscep_file*".bson"), fidelity_suscep_data)
end



###--- PLOTS:
if plot_fidelity_suscep
    fidelity_suscep_data = BSON.load(string(@__DIR__,"/data/"*fidelity_suscep_file*".bson"))
    fidelity_suscep_vals = fidelity_suscep_data[:fidelity_suscep]
    N_vals               = fidelity_suscep_data[:N]
    alpha_vals_suscep    = fidelity_suscep_data[:alpha]
    dα = alpha_vals_suscep[2]-alpha_vals_suscep[1]
    alpha_peak = Array{Float64}(length(N_vals))

    for i = 2:length(N_vals) # N-loop
        N = N_vals[i]
        dχ = diff(fidelity_suscep_vals[:,i]) / dα
        ddχ = diff(dχ) / dα

        figure(5)
        plot(alpha_vals_suscep[1:end-1], fidelity_suscep_vals[:,i], label="\$N = $N\$")

        figure(6)
        plot(alpha_vals_suscep[2:end-2], -ddχ, label="\$N = $N\$")

        figure(7)
        plot(alpha_vals_suscep[1:end-1], -2*log.(fidelity_suscep_vals[:,i]) / (dα^2), label="\$N = $N\$")

        ## peak position:
        ind_peak = indmax(-2*log.(fidelity_suscep_vals[:,i]) / (dα^2))
        alpha_peak[i] = alpha_vals_suscep[ind_peak]
    end

    figure(7)
    # axhline(0.0, ls="--", c="grey",zorder=-1)
    xlim(0,3)
    ylim(0,0.095)
    # ax = subplot(111)
    # ax[:ticklabel_format](axis="y", style="scientific", scilimits=(0,0), useOffset=true)
    xlabel("\$\\alpha\$")
    ylabel("\$\\chi_F\\, (\\alpha)\$")
    legend(loc = "best", numpoints=3, frameon=1, facecolor="white", fancybox=1, columnspacing=1, title="\$g=3.0\$\n LR pbc")
    layout.nice_ticks()
    savefig(string(@__DIR__,"/figures/fidelity_suscep.pdf"))

    figure(8)
    linmodel(x,p) = p[1] + p[2]*x
    # p0 = [1.0, 1.0]
    # linfit = curve_fit(linmodel,1 ./ N_vals[2:end],alpha_peak[2:end],p0)
    # linparams = linfit.param
    linparams = linreg(1 ./ N_vals[2:end], alpha_peak[2:end])
    plot(1 ./ N_vals[2:end], alpha_peak[2:end], ls="", marker="s")
    plot(linspace(0,1/N_vals[1]), linmodel(linspace(0,1/N_vals[1]),linparams), ls=":", c="k")
    xlim(left=0.0)
    println("linfit: ",linparams)
end



# x=linspace(0,3,301)
# dx=x[2]-x[1]
# y=sin(x)
# dy=diff(y) / dx
# ddy=diff(dy) / dx
# plot(x,y)
# plot(x[1:end-1],dy)
# plot(x[1:end-2],ddy)





println("done: fidelity.jl")
show()
;
