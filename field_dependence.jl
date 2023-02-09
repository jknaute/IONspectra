include("layout.jl")
using layout
using BSON
using PyPlot
using LsqFit # Pkg.add("LsqFit",v"0.3.0")

NN_files = ["g4_N18_n600_Gamma0.1",
            "g3.5_N18_n600_Gamma0.1",
            "g3_N18_n600_Gamma0.1",
            "g2.5_N18_n600_Gamma0.1",
            "g2_N18_n600_Gamma0.1",
            "g1.5_N18_n600_Gamma0.1",
            "g1_N18_n600_Gamma0.1"
           ]
LR_files = ["g4_N18_n600_alpha3_Gamma0.1",
            "g3.5_N18_n600_alpha3_Gamma0.1",
            "g3_N18_n600_alpha3_Gamma0.1",
            "g2.5_N18_n600_alpha3_Gamma0.1",
            "g2_N18_n600_alpha3_Gamma0.1",
            "g1.5_N18_n600_alpha3_Gamma0.1",
            "g1_N18_n600_alpha3_Gamma0.1"
           ]

g_comp = 3.0   # chosen g field value for comparison with E8 spectrum
num_levels = 5 # number of E8 levels to analyze
omega_plot_max = 3.2 # max omega/m1


NN_masses = Array{Float64}(length(NN_files),num_levels)
NN_mass_uncert = Array{Float64}(length(NN_files),num_levels)
NN_ratio_uncert = Array{Float64}(length(NN_files),num_levels)
NN_g_vals = Array{Float64}(length(NN_files))
NN_masses[:,:] = NN_mass_uncert[:,:] = NN_ratio_uncert[:,:] = NaN

LR_masses = Array{Float64}(length(LR_files),num_levels)
LR_mass_uncert = Array{Float64}(length(LR_files),num_levels)
LR_ratio_uncert = Array{Float64}(length(LR_files),num_levels)
LR_g_vals = Array{Float64}(length(LR_files))
LR_masses[:,:] = LR_mass_uncert[:,:] = LR_ratio_uncert[:,:] = NaN

E8_masses = [1., 1.61803, 1.98904, 2.40487, 2.9563, 3.21834, 3.89116, 4.78339] # analytical E8 spectrum (m_n/m_1)
include(string(@__DIR__,"/figures/2103.09128/Dzz.jl")) # = Dzz

model1(x,p) = p[3] ./     ((x-p[1]).^2 .+    p[2]^2)
model2(x,p) = p[3] * exp.(-(x-p[1]).^2 ./ (2*p[2]^2)) ./ sqrt(2*pi*p[2]^2)


### ------------------------------------------   NN data   -----------------------
for j = 1:length(NN_files)
    NN_data =  BSON.load(string(@__DIR__,"/data/NN_data_"*NN_files[j]*".bson"))

    println("NN_data: ",NN_data[:info])
    N = NN_data[:N]; d = NN_data[:d]; num_states = NN_data[:num_states]; J = NN_data[:J]; h = NN_data[:h]; g = NN_data[:g]; Γ = NN_data[:Gamma]
    E_excitations_NN_obc = NN_data[:E_excitations_NN_obc]; E_excitations_NN_pbc = NN_data[:E_excitations_NN_pbc]
    E_ratios_NN_obc = NN_data[:E_ratios_NN_obc]; E_ratios_NN_pbc = NN_data[:E_ratios_NN_pbc]
    omega = NN_data[:omega]; susceptibilities = NN_data[:chi]
    omega_max = omega[end]/E_excitations_NN_pbc[2]
    NN_g_vals[j] = g

    for k=1:num_levels
        if k < 4
            p0 = [1.0, 1.0, 20.0]
            lb = [0.5, 0.0, 10.0]
            ub = [3.5, 2.0, 3000.0]
            k_ind = Int(floor(E8_masses[k]/omega_max*length(omega)))
            halfwindow = 75
            xdata = (omega/E_excitations_NN_pbc[2])[k_ind-halfwindow:k_ind+halfwindow]
            ydata = real(susceptibilities[k_ind-halfwindow:k_ind+halfwindow])
            # fit1 = curve_fit(model1,xdata,ydata,p0)
            # p1 = fit1.param
            # println("Lorentz fit: ",p1)

            fit2 = curve_fit(model2,xdata,ydata,p0, lower=lb, upper=ub)
            p2 = fit2.param
            FWHM = 2.355*abs(p2[2])
            NN_masses[j,k] = p2[1]
            NN_mass_uncert[j,k] = 0.5*FWHM
            println("Gauss fit, FWHM: ",p2,", ",FWHM)

            figure(j)
            # plot(xdata, model1(xdata,p1), ls="--", c="r")
            plot(xdata, model2(xdata,p2), ls="--", c="k", zorder=-3)
            plot(linspace(p2[1]-FWHM/2,p2[1]+FWHM/2,100), 0.5*model2(p2[1],p2)*ones(100), c="b", zorder=-1)
        elseif k==4
            if g >= 2.0
                p0 = [2.3, 1.0, 10.0]
                lb = [2.0, 0.0, 1.0]
                ub = [2.5, 2.0, 1000.0]
                if g<=3.5 pos=2.3 else pos=2.4 end
                k_ind = Int(floor(pos/omega_max*length(omega)))
                halfwindow = 25
                xdata = (omega/E_excitations_NN_pbc[2])[k_ind-halfwindow:k_ind+halfwindow]
                ydata = real(susceptibilities[k_ind-halfwindow:k_ind+halfwindow])
                # fit1 = curve_fit(model1,xdata,ydata,p0)
                # p1 = fit1.param
                # println("Lorentz fit: ",p1)

                fit2 = curve_fit(model2,xdata,ydata,p0, lower=lb, upper=ub)
                p2 = fit2.param
                FWHM = 2.355*abs(p2[2])
                NN_masses[j,k] = p2[1]
                NN_mass_uncert[j,k] = 0.5*FWHM
                println("Gauss fit, FWHM: ",p2,", ",FWHM)

                figure(j)
                # plot(xdata, model1(xdata,p1), ls="--", c="r")
                plot(xdata, model2(xdata,p2), ls="--", c="k", zorder=-3)
                plot(linspace(p2[1]-FWHM/2,p2[1]+FWHM/2,100), 0.5*model2(p2[1],p2)*ones(100), c="b", zorder=-1)
            end
        elseif k==5
            if g >= 1.5
                p0 = [3.0, 1.0, 100.0]
                lb = [2.7, 0.0, 1.0]
                ub = [3.3, 2.0, 1000.0]
                k_ind = Int(floor(E8_masses[k]/omega_max*length(omega)))
                halfwindow = 50
                xdata = (omega/E_excitations_NN_pbc[2])[k_ind-halfwindow:k_ind+halfwindow]
                ydata = real(susceptibilities[k_ind-halfwindow:k_ind+halfwindow])
                # fit1 = curve_fit(model1,xdata,ydata,p0)
                # p1 = fit1.param
                # println("Lorentz fit: ",p1)

                fit2 = curve_fit(model2,xdata,ydata,p0, lower=lb, upper=ub)
                p2 = fit2.param
                FWHM = 2.355*abs(p2[2])
                NN_masses[j,k] = p2[1]
                NN_mass_uncert[j,k] = 0.5*FWHM
                println("Gauss fit, FWHM: ",p2,", ",FWHM)

                figure(j)
                # plot(xdata, model1(xdata,p1), ls="--", c="r")
                plot(xdata, model2(xdata,p2), ls="--", c="k", zorder=-3)
                plot(linspace(p2[1]-FWHM/2,p2[1]+FWHM/2,100), 0.5*model2(p2[1],p2)*ones(100), c="b", zorder=-1)
            end
        end
    end

    figure(j)
    for i in 1:8 axvline(E8_masses[i], ls="--", c="grey",zorder=-1) end
    plot(omega/E_excitations_NN_pbc[2], susceptibilities, label="\$g = $g\$", zorder=-2)
    legend(loc = "best", numpoints=3, frameon=1, facecolor="white", fancybox=1, columnspacing=1, title="NN pbc")

    if g==g_comp
        figure(500) # comparison with E8 spectrum
        plot(omega/E_excitations_NN_pbc[2], susceptibilities/maximum(real(susceptibilities)), c="C1", label="NN pbc", zorder=-3)
    end
end


### ------------------------------------------   LR data   -----------------------
for j = 1:length(LR_files)
    LR_data =  BSON.load(string(@__DIR__,"/data/LR_data_"*LR_files[j]*".bson"))

    println("LR_data: ",LR_data[:info])
    N = LR_data[:N]; d = LR_data[:d]; num_states = LR_data[:num_states]; J = LR_data[:J]; h = LR_data[:h]; g = LR_data[:g]; alpha = LR_data[:alpha]; Γ = LR_data[:Gamma]
    E_excitations_LR_obc = LR_data[:E_excitations_LR_obc]; E_excitations_LR_pbc = LR_data[:E_excitations_LR_pbc]
    E_ratios_LR_obc = LR_data[:E_ratios_LR_obc]; E_ratios_LR_pbc = LR_data[:E_ratios_LR_pbc]
    omega = LR_data[:omega]; susceptibilities = LR_data[:chi]
    omega_max = omega[end]/E_excitations_LR_pbc[2]
    LR_g_vals[j] = g

    for k=1:num_levels
        if k < 4
            p0 = [1.0, 1.0, 20.0]
            lb = [0.5, 0.0, 10.0]
            ub = [3.0, 2.0, 3000.0]
            k_ind = Int(floor(E8_masses[k]/omega_max*length(omega)))
            halfwindow = 75
            xdata = (omega/E_excitations_LR_pbc[2])[k_ind-halfwindow:k_ind+halfwindow]
            ydata = real(susceptibilities[k_ind-halfwindow:k_ind+halfwindow])
            # fit1 = curve_fit(model1,xdata,ydata,p0)
            # p1 = fit1.param
            # println("Lorentz fit: ",p1)

            fit2 = curve_fit(model2,xdata,ydata,p0, lower=lb, upper=ub)
            p2 = fit2.param
            FWHM = 2.355*abs(p2[2])
            LR_masses[j,k] = p2[1]
            LR_mass_uncert[j,k] = 0.5*FWHM
            println("Gauss fit, FWHM: ",p2,", ",FWHM)

            figure(200+j)
            # plot(xdata, model1(xdata,p1), ls="--", c="r")
            plot(xdata, model2(xdata,p2), ls="--", c="k", zorder=-3)
            plot(linspace(p2[1]-FWHM/2,p2[1]+FWHM/2,100), 0.5*model2(p2[1],p2)*ones(100), c="b", zorder=-1)
        elseif k==4
            if g >= 2.0
                p0 = [2.3, 1.0, 5.0]
                lb = [2.0, 0.0, 0.0]
                ub = [2.5, 2.0, 1000.0]
                if g<=3.5 pos=2.3 else pos=2.4 end
                k_ind = Int(floor(pos/omega_max*length(omega)))
                halfwindow = 25
                xdata = (omega/E_excitations_LR_pbc[2])[k_ind-halfwindow:k_ind+halfwindow]
                ydata = real(susceptibilities[k_ind-halfwindow:k_ind+halfwindow])
                # fit1 = curve_fit(model1,xdata,ydata,p0)
                # p1 = fit1.param
                # println("Lorentz fit: ",p1)

                fit2 = curve_fit(model2,xdata,ydata,p0, lower=lb, upper=ub)
                p2 = fit2.param
                FWHM = 2.355*abs(p2[2])
                LR_masses[j,k] = p2[1]
                LR_mass_uncert[j,k] = 0.5*FWHM
                println("Gauss fit, FWHM: ",p2,", ",FWHM)

                figure(200+j)
                # plot(xdata, model1(xdata,p1), ls="--", c="r")
                plot(xdata, model2(xdata,p2), ls="--", c="k", zorder=-3)
                plot(linspace(p2[1]-FWHM/2,p2[1]+FWHM/2,100), 0.5*model2(p2[1],p2)*ones(100), c="b", zorder=-1)
            end
        elseif k==5
            if g >= 1.5
                p0 = [3.0, 1.0, 10.0]
                lb = [2.7, 0.0, 1.0]
                ub = [3.3, 2.0, 1000.0]
                k_ind = Int(floor(E8_masses[k]/omega_max*length(omega)))
                halfwindow = 50
                xdata = (omega/E_excitations_LR_pbc[2])[k_ind-halfwindow:k_ind+halfwindow]
                ydata = real(susceptibilities[k_ind-halfwindow:k_ind+halfwindow])
                # fit1 = curve_fit(model1,xdata,ydata,p0)
                # p1 = fit1.param
                # println("Lorentz fit: ",p1)

                fit2 = curve_fit(model2,xdata,ydata,p0, lower=lb, upper=ub)
                p2 = fit2.param
                FWHM = 2.355*abs(p2[2])
                LR_masses[j,k] = p2[1]
                LR_mass_uncert[j,k] = 0.5*FWHM
                println("Gauss fit, FWHM: ",p2,", ",FWHM)

                figure(200+j)
                # plot(xdata, model1(xdata,p1), ls="--", c="r")
                plot(xdata, model2(xdata,p2), ls="--", c="k", zorder=-3)
                plot(linspace(p2[1]-FWHM/2,p2[1]+FWHM/2,100), 0.5*model2(p2[1],p2)*ones(100), c="b", zorder=-1)
            end
        end
    end

    figure(200+j)
    for i in 1:8 axvline(E8_masses[i], ls="--", c="grey",zorder=-1) end
    plot(omega/E_excitations_LR_pbc[2], susceptibilities, label="\$g = $g\$", zorder=-2)
    legend(loc = "best", numpoints=3, frameon=1, facecolor="white", fancybox=1, columnspacing=1, title="LR pbc")

    if g==g_comp
        figure(500) # comparison with E8 spectrum
        plot(omega/E_excitations_LR_pbc[2], susceptibilities/maximum(real(susceptibilities)), ls="-.", c="C2", label="LR pbc", zorder=-2)
    end
end


## extracted mass uncertainties:
figure(300)
elw = 3
cs = 5
colNN = ["b","g","r","c","m"]
for i in 1:num_levels axhline(E8_masses[i], ls="--", c="grey",zorder=-1) end
errorbar(NN_g_vals, NN_masses[:,1], yerr=NN_mass_uncert[:,1], ls="", marker="s", c="k", ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=-1, label="NN pbc")
for k=1:num_levels
    errorbar(NN_g_vals, NN_masses[:,k], yerr=NN_mass_uncert[:,k], ls="", marker="s", c=colNN[k], ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=1)
end
shift = 0.07
colLR = ["C0","C2","C3","C0","C6"]
ebLRd = errorbar(LR_g_vals+shift, LR_masses[:,1], yerr=LR_mass_uncert[:,1], ls="", marker="s", c="k", ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=-1, label="LR pbc")
ebLRd[end][1][:set_linestyle](":")
for k=1:num_levels
    ebLRk = errorbar(LR_g_vals+shift, LR_masses[:,k], yerr=LR_mass_uncert[:,k], ls="", marker="s", c=colLR[k], ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=1)
    ebLRk[end][1][:set_linestyle](":")
end
ylim(0.75,3.15)
xlabel("\$g\$")
ylabel("\$\\widetilde{M}_n/m_1\$")
layout.nice_ticks()
# legend(loc = "best", numpoints=3, frameon=1, facecolor="white", fancybox=1, columnspacing=1)
savefig(string(@__DIR__,"/figures/masses_field_dependence.pdf"))


## extracted mass ratios from peak positions:
## Gauss'sche Fehlerfortpflanzung:
for j=1:length(NN_g_vals)
    for k=1:num_levels
        NN_ratio_uncert[j,k] = sqrt((NN_mass_uncert[j,k]/NN_masses[j,1])^2 + (NN_masses[j,k]*NN_mass_uncert[j,1]/NN_masses[j,1]^2)^2)
    end
end
for j=1:length(LR_g_vals)
    for k=1:num_levels
        LR_ratio_uncert[j,k] = sqrt((LR_mass_uncert[j,k]/LR_masses[j,1])^2 + (LR_masses[j,k]*LR_mass_uncert[j,1]/LR_masses[j,1]^2)^2)
    end
end

figure(400)
elw = 3
cs = 5
for i in 1:num_levels axhline(E8_masses[i], ls="--", c="grey",zorder=-1) end
errorbar(NN_g_vals, NN_masses[:,2]./NN_masses[:,1], yerr=NN_ratio_uncert[:,2], ls="", marker="s", c="k", ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=-1, label="NN pbc")
for k=1:num_levels
    errorbar(NN_g_vals, NN_masses[:,k]./NN_masses[:,1], yerr=NN_ratio_uncert[:,k], ls="", marker="s", c=colNN[k], ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=1)
end
ebLRd = errorbar(LR_g_vals+shift, LR_masses[:,2]./LR_masses[:,1], yerr=LR_ratio_uncert[:,2], ls="", marker="s", c="k", ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=-1, label="LR pbc")
ebLRd[end][1][:set_linestyle](":")
for k=1:num_levels
    ebLRk = errorbar(LR_g_vals+shift, LR_masses[:,k]./LR_masses[:,1], yerr=LR_ratio_uncert[:,k], ls="", marker="s", c=colLR[k], ms=0,mew=3,elinewidth=elw,capsize=cs,zorder=1)
    ebLRk[end][1][:set_linestyle](":")
end
ylim(0.75,3.15)
xlabel("\$g\$")
ylabel("\$\\widetilde{M}_n/\\widetilde{M}_1\$")
layout.nice_ticks()
# legend(loc = "best", numpoints=3, frameon=1, facecolor="white", fancybox=1, columnspacing=1)
savefig(string(@__DIR__,"/figures/masspeaks_field_dependence.pdf"))



## comparison with E8 spectrum:
figure(500)
for i in 1:8 axvline(E8_masses[i], ls="--", c="grey",zorder=-4) end
axvline(E8_masses[1]+E8_masses[2], ls=":", c="silver",zorder=-4)
axvline(E8_masses[1]+E8_masses[3], ls=":", c="silver",zorder=-4)
axvline(E8_masses[2]+E8_masses[2], ls=":", c="silver",zorder=-4)
plot(Dzz[:,1], Dzz[:,2]/maximum(Dzz[:,2]), ls=":", c="b", label="E\$_8\$ QFT", zorder=1)
xlim(0,omega_plot_max)
ylim(0,1.1)
xlabel("\$\\omega / m_1\$")
ylabel("\$\\chi^{\\prime\\prime}(\\omega,k=0)\$ (scaled)")
legend(loc = "upper right", numpoints=3, frameon=1, facecolor="white", fancybox=1, columnspacing=1)
layout.nice_ticks()
savefig(string(@__DIR__,"/figures/absorptionrate_E8comparison.pdf"))
