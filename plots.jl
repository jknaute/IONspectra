# include("layout.jl")
include("layout_subfigures.jl") # for 4-subpanel Figures
using layout
using BSON
using PyPlot


NN_files = ["g4_N18_n600_Gamma0.1", # 1
            "g3_N18_n600_Gamma0.1", # 2
            "g3_N12_n400_Gamma0.1", # 3
            "g2_N18_n600_Gamma0.1", # 4
            "g1_N18_n600_Gamma0.1"  # 5
           ]
LR_files = ["g4_N18_n600_alpha3_Gamma0.1",  # 1
            "g3_N18_n600_alpha3_Gamma0.1",  # 2
            "g3_N18_n600_alpha2_Gamma0.1",  # 3
            "g3_N18_n600_alpha1_Gamma0.1",  # 4
            "g3_N18_n400_alpha0_Gamma0.1",  # 5
            "g3_N12_n400_alpha3_Gamma0.1",  # 6
            "g2_N18_n600_alpha3_Gamma0.1",  # 7
            "g1_N18_n600_alpha3_Gamma0.1"   # 8
           ]

E8_masses = [1., 1.61803, 1.98904, 2.40487, 2.9563, 3.21834, 3.89116, 4.78339] # analytical E8 spectrum (m_n/m_1)
omega_plot_max = 3.2 # max omega/m1

plot_energy_spectra = true
plot_absorption_spectra = false
plot_adiabatic_comparison = false

### ------------------------------------------   NN data   -----------------------------------------------
for j = 1:length(NN_files)
    NN_data =  BSON.load(string(@__DIR__,"/data/NN_data_"*NN_files[j]*".bson"))

    println("NN_data: ",NN_data[:info])
    N = NN_data[:N]; d = NN_data[:d]; num_states = NN_data[:num_states]; J = NN_data[:J]; h = NN_data[:h]; g = NN_data[:g]; Γ = NN_data[:Gamma]
    E_excitations_NN_obc = NN_data[:E_excitations_NN_obc]; E_excitations_NN_pbc = NN_data[:E_excitations_NN_pbc]
    E_ratios_NN_obc = NN_data[:E_ratios_NN_obc]; E_ratios_NN_pbc = NN_data[:E_ratios_NN_pbc]
    omega = NN_data[:omega]; susceptibilities = NN_data[:chi]
    omega_max = omega[end]/E_excitations_NN_pbc[2]
    omega_plot_max_ind = Int(floor(omega_plot_max/omega_max*length(omega)))

    ## energy spectrum:
    if plot_energy_spectra
        figure(j)
        for i in 1:8 axhline(E8_masses[i], ls="--", c="grey",zorder=-1) end
        axhline(E8_masses[1]+E8_masses[2], ls=":", c="silver",zorder=-1)
        axhline(E8_masses[1]+E8_masses[3], ls=":", c="silver",zorder=-1)
        axhline(E8_masses[2]+E8_masses[2], ls=":", c="silver",zorder=-1)
        plot(1:num_states-1,E_ratios_NN_obc,ls="",marker="o", label="NN obc")
        plot(1:num_states-1,E_ratios_NN_pbc,ls="",marker="v", label="NN pbc")
        ylim(top=4.5)
        xlabel("\$n\$", labelpad=-2)
        ylabel("\$m_n/m_1\$")
        # legend(loc = "lower right", numpoints=3, frameon=1, facecolor="white", fancybox=1, columnspacing=1, title="\$N = $N, g=$g\$")
        text(0,4, "\$N = $N, g=$g\$")
        layout.nice_ticks()
        savefig(string(@__DIR__,"/figures/IsingSpectrum_NN_"*NN_files[j]*".pdf"))
    end

    ## absorption spectrum:
    if plot_absorption_spectra
        if j==2
            figure(20)
            for i in 1:8 axvline(E8_masses[i], ls="--", c="grey",zorder=-1) end
            axvline(E8_masses[1]+E8_masses[2], ls=":", c="silver",zorder=-1)
            axvline(E8_masses[1]+E8_masses[3], ls=":", c="silver",zorder=-1)
            axvline(E8_masses[2]+E8_masses[2], ls=":", c="silver",zorder=-1)
            plot(omega/E_excitations_NN_pbc[2], susceptibilities, c="C1", label="NN pbc", zorder=9)
            xlim(0,omega_plot_max)
            ylim(0,14000)
            # ylim(bottom=0)
            xlabel("\$\\omega / m_1\$")
            ylabel("\$\\chi^{\\prime\\prime}(\\omega,k=0)\$")
            legend(loc = "upper right", numpoints=3, frameon=1, facecolor="white", fancybox=1, columnspacing=1, title="\$N = $N, g=$g, \\Gamma/J=$Γ\$")
            layout.nice_ticks()
            savefig(string(@__DIR__,"/figures/absorptionrate_NN_"*NN_files[j]*".pdf"))

            figure(21) # finite size dependence
            plot(omega/E_excitations_NN_pbc[2], susceptibilities/maximum(real(susceptibilities)), c="C1", ls="-.", label="\$N = $N\$", zorder=10-j)

            figure(22) # longitudinal field dependence
            plot(omega/E_excitations_NN_pbc[2], susceptibilities, c="C1", ls="--", label="\$g = $g\$", zorder=10-j)
        end
        if j==3
            figure(21) # finite size dependence
            plot(omega/E_excitations_NN_pbc[2], susceptibilities/maximum(real(susceptibilities)), c="C2", label="\$N = $N\$", zorder=10-j)
        end
        if j==1
            figure(22) # longitudinal field dependence
            plot(omega/E_excitations_NN_pbc[2], susceptibilities, c="C0", ls="-", label="\$g = $g\$", zorder=10-j)
        end
        if j>=4
            figure(22) # longitudinal field dependence
            lpatterns = ["-.",":"]
            plot(omega/E_excitations_NN_pbc[2], susceptibilities, c="C"*string(j-2), ls=lpatterns[j-3], label="\$g = $g\$", zorder=10-j)
        end
    end
end


### ------------------------------------------   LR data   ------------------------------------------------
for j = 1:length(LR_files)
    LR_data =  BSON.load(string(@__DIR__,"/data/LR_data_"*LR_files[j]*".bson"))

    println("LR_data: ",LR_data[:info])
    N = LR_data[:N]; d = LR_data[:d]; num_states = LR_data[:num_states]; J = LR_data[:J]; h = LR_data[:h]; g = LR_data[:g]; alpha = LR_data[:alpha]; Γ = LR_data[:Gamma]
    E_excitations_LR_obc = LR_data[:E_excitations_LR_obc]; E_excitations_LR_pbc = LR_data[:E_excitations_LR_pbc]
    E_ratios_LR_obc = LR_data[:E_ratios_LR_obc]; E_ratios_LR_pbc = LR_data[:E_ratios_LR_pbc]
    omega = LR_data[:omega]; susceptibilities = LR_data[:chi]
    omega_max = omega[end]/E_excitations_LR_pbc[2]
    omega_plot_max_ind = Int(floor(omega_plot_max/omega_max*length(omega)))

    ## energy spectrum:
    if plot_energy_spectra
        figure(200+j)
        for i in 1:8 axhline(E8_masses[i], ls="--", c="grey",zorder=-1) end
        axhline(E8_masses[1]+E8_masses[2], ls=":", c="silver",zorder=-1)
        axhline(E8_masses[1]+E8_masses[3], ls=":", c="silver",zorder=-1)
        axhline(E8_masses[2]+E8_masses[2], ls=":", c="silver",zorder=-1)
        plot(1:num_states-1,E_ratios_LR_obc,ls="",marker="o", label="LR obc")
        plot(1:num_states-1,E_ratios_LR_pbc,ls="",marker="v", label="LR pbc")
        ylim(top=4.5)
        xlabel("\$n\$", labelpad=-2)
        ylabel("\$m_n/m_1\$")
        # legend(loc = "lower right", numpoints=3, frameon=1, facecolor="white", fancybox=1, columnspacing=1, title="\$N = $N, g=$g, \\alpha=$alpha \$")
        text(0,4, "\$N = $N, g=$g, \\alpha=$alpha \$")
        layout.nice_ticks()
        savefig(string(@__DIR__,"/figures/IsingSpectrum_LR_"*LR_files[j]*".pdf"))
    end

    ## absorption spectrum:
    if plot_absorption_spectra
        if 2 <= j <= 5 # alpha dependence
            figure(219)
            lpatterns = [":","-.","--","-"]
            plot(omega/E_excitations_LR_pbc[2], real(susceptibilities), c="C"*string(j-1), ls=lpatterns[j-1], label="\$\\alpha=$alpha\$", zorder=10-j)
            # plot(omega/E_excitations_LR_pbc[2], real(susceptibilities)/maximum(real(susceptibilities)), c="C"*string(j-1), ls=lpatterns[j-1], label="\$\\alpha=$alpha\$", zorder=10-j)
        end
        if j==2
            figure(221) # finite size dependence
            plot(omega/E_excitations_LR_pbc[2], susceptibilities/maximum(real(susceptibilities)), c="C1", ls="-.", label="\$N = $N\$", zorder=10-j)
            figure(222) # longitudinal field dependence
            plot(omega/E_excitations_LR_pbc[2], susceptibilities, c="C1", ls="--", label="\$g = $g\$", zorder=10-j)
        end
        if j==6
            figure(221) # finite size dependence
            plot(omega/E_excitations_LR_pbc[2], susceptibilities/maximum(real(susceptibilities)), c="C2", label="\$N = $N\$", zorder=10-j)
        end
        if j==1 # longitudinal field dependence
            figure(222)
            plot(omega/E_excitations_LR_pbc[2], susceptibilities, c="C0", ls="-", label="\$g = $g\$", zorder=10-j)
        end
        if j>=7 # longitudinal field dependence
            figure(222)
            lpatterns = ["-.",":"]
            plot(omega/E_excitations_LR_pbc[2], susceptibilities, c="C"*string(j-5), ls=lpatterns[j-6], label="\$g = $g\$", zorder=10-j)
        end
    end
end




if plot_absorption_spectra
            ##+++++++++++++++++++++++++++++
            ##--- alpha dependence:
            figure(219)
            for i in 1:8 axvline(E8_masses[i], ls="--", c="grey",zorder=-1) end
            axvline(E8_masses[1]+E8_masses[2], ls=":", c="silver",zorder=-1)
            axvline(E8_masses[1]+E8_masses[3], ls=":", c="silver",zorder=-1)
            axvline(E8_masses[2]+E8_masses[2], ls=":", c="silver",zorder=-1)
            # xlim(0,omega_max)
            xlim(0,omega_plot_max)
            # ylim(0,1.1)
            ylim(0,14000)
            # ylim(bottom=0)
            xlabel("\$\\omega / m_1\$")
            ylabel("\$\\chi^{\\prime\\prime}(\\omega,k=0)\$")
            # ylabel("\$\\chi^{\\prime\\prime}(\\omega,k=0)\$ (scaled)")
            legend(loc = "upper right", numpoints=3, frameon=1, facecolor="white", fancybox=1, columnspacing=1, title="\$N = 18, g=3.0, \\Gamma/J=0.1\$\n LR pbc")
            layout.nice_ticks()
            savefig(string(@__DIR__,"/figures/absorptionrate_LR_alpha_dependence.pdf"))


            ##--- finite size dependence:
            ## NN
            figure(21)
            for i in 1:8 axvline(E8_masses[i], ls="--", c="grey",zorder=-1) end
            axvline(E8_masses[1]+E8_masses[2], ls=":", c="silver",zorder=-1)
            axvline(E8_masses[1]+E8_masses[3], ls=":", c="silver",zorder=-1)
            axvline(E8_masses[2]+E8_masses[2], ls=":", c="silver",zorder=-1)
            # xlim(0,omega_max)
            xlim(0,omega_plot_max)
            ylim(0,1.1)
            # ylim(bottom=0)
            xlabel("\$\\omega / m_1\$")
            ylabel("\$\\chi^{\\prime\\prime}(\\omega,k=0)\$ (scaled)")
            legend(loc = "upper right", numpoints=3, frameon=1, facecolor="white", fancybox=1, columnspacing=1, title="\$g=3.0, \\Gamma/J=0.1\$\n NN pbc")
            layout.nice_ticks()
            savefig(string(@__DIR__,"/figures/absorptionrate_NN_finite_size_dependence.pdf"))

            ## LR
            figure(221)
            for i in 1:8 axvline(E8_masses[i], ls="--", c="grey",zorder=-1) end
            axvline(E8_masses[1]+E8_masses[2], ls=":", c="silver",zorder=-1)
            axvline(E8_masses[1]+E8_masses[3], ls=":", c="silver",zorder=-1)
            axvline(E8_masses[2]+E8_masses[2], ls=":", c="silver",zorder=-1)
            # xlim(0,omega_max)
            xlim(0,omega_plot_max)
            ylim(0,1.1)
            # ylim(bottom=0)
            xlabel("\$\\omega / m_1\$")
            ylabel("\$\\chi^{\\prime\\prime}(\\omega,k=0)\$ (scaled)")
            legend(loc = "upper right", numpoints=3, frameon=1, facecolor="white", fancybox=1, columnspacing=1, title="\$g=3.0, \\alpha=3.0, \\Gamma/J=0.1\$\n LR pbc")
            layout.nice_ticks()
            savefig(string(@__DIR__,"/figures/absorptionrate_LR_finite_size_dependence.pdf"))


            ##--- longitudinal field dependence:
            ## NN
            figure(22)
            for i in 1:8 axvline(E8_masses[i], ls="--", c="grey",zorder=-1) end
            axvline(E8_masses[1]+E8_masses[2], ls=":", c="silver",zorder=-1)
            axvline(E8_masses[1]+E8_masses[3], ls=":", c="silver",zorder=-1)
            axvline(E8_masses[2]+E8_masses[2], ls=":", c="silver",zorder=-1)
            # xlim(0,omega_max)
            xlim(0.8,omega_plot_max)
            ylim(0,14000)
            # ylim(bottom=0)
            xlabel("\$\\omega / m_1\$")
            ylabel("\$\\chi^{\\prime\\prime}(\\omega,k=0)\$")
            legend(loc = "upper right", numpoints=3, frameon=1, facecolor="white", fancybox=1, columnspacing=1, title="\$N=18, \\Gamma/J=0.1\$\n NN pbc")
            layout.nice_ticks()
            savefig(string(@__DIR__,"/figures/absorptionrate_NN_field_dependence.pdf"))

            ## LR
            figure(222)
            for i in 1:8 axvline(E8_masses[i], ls="--", c="grey",zorder=-1) end
            axvline(E8_masses[1]+E8_masses[2], ls=":", c="silver",zorder=-1)
            axvline(E8_masses[1]+E8_masses[3], ls=":", c="silver",zorder=-1)
            axvline(E8_masses[2]+E8_masses[2], ls=":", c="silver",zorder=-1)
            # xlim(0,omega_max)
            xlim(0.8,omega_plot_max)
            ylim(0,14000)
            # ylim(bottom=0)
            xlabel("\$\\omega / m_1\$")
            ylabel("\$\\chi^{\\prime\\prime}(\\omega,k=0)\$")
            legend(loc = "upper right", numpoints=3, frameon=1, facecolor="white", fancybox=1, columnspacing=1, title="\$N=18, \\alpha=3.0, \\Gamma/J=0.1\$\n LR pbc")
            layout.nice_ticks()
            savefig(string(@__DIR__,"/figures/absorptionrate_LR_field_dependence.pdf"))
            ##+++++++++++++++++++++++++++++
end # of plot_absorption_spectra condition





##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if plot_adiabatic_comparison
    LR_data =  BSON.load(string(@__DIR__,"/data/LR_data_"*LR_files[6]*".bson"))
    println("LR_data: ",LR_data[:info])
    N = LR_data[:N]; d = LR_data[:d]; num_states = LR_data[:num_states]; J = LR_data[:J]; h = LR_data[:h]; g = LR_data[:g]; alpha = LR_data[:alpha]; Γ = LR_data[:Gamma]
    E_excitations_LR_obc = LR_data[:E_excitations_LR_obc]; E_excitations_LR_pbc = LR_data[:E_excitations_LR_pbc]
    E_ratios_LR_obc = LR_data[:E_ratios_LR_obc]; E_ratios_LR_pbc = LR_data[:E_ratios_LR_pbc]
    omega = LR_data[:omega]; susceptibilities = LR_data[:chi]
    omega_max = omega[end]/E_excitations_LR_pbc[2]
    omega_plot_max_ind = Int(floor(omega_plot_max/omega_max*length(omega)))

    figure(70)
    plot(omega/E_excitations_LR_pbc[2], susceptibilities, c="k", ls="-", label="exact gs")

    ## adiabatic states:
    polar_data =  BSON.load(string(@__DIR__,"/data/LR_data_g3_alpha3_adiabatic_tramp1.bson")); susceptibilities = polar_data[:chi]
    plot(omega/E_excitations_LR_pbc[2], susceptibilities, c="C0", ls="-.", label="\$t_{ramp}J = 1.0\$",zorder=0)
    # polar_data =  BSON.load(string(@__DIR__,"/data/LR_data_g3_alpha3_adiabatic_tramp2.bson")); susceptibilities = polar_data[:chi]
    # plot(omega/E_excitations_LR_pbc[2], susceptibilities, c="C1", ls="-.", label="\$t_{ramp}J = 2.0\$")
    polar_data =  BSON.load(string(@__DIR__,"/data/LR_data_g3_alpha3_adiabatic_tramp4.bson")); susceptibilities = polar_data[:chi]
    plot(omega/E_excitations_LR_pbc[2], susceptibilities, c="C1", ls="--", label="\$t_{ramp}J = 4.0\$")
    polar_data =  BSON.load(string(@__DIR__,"/data/LR_data_g3_alpha3_adiabatic_tramp8.bson")); susceptibilities = polar_data[:chi]
    plot(omega/E_excitations_LR_pbc[2], susceptibilities, c="C2", ls=":", label="\$t_{ramp}J = 8.0\$")

    ## plot props:
    for i in 1:8 axvline(E8_masses[i], ls="--", c="grey",zorder=-1) end
    axvline(E8_masses[1]+E8_masses[2], ls=":", c="silver",zorder=-1)
    axvline(E8_masses[1]+E8_masses[3], ls=":", c="silver",zorder=-1)
    axvline(E8_masses[2]+E8_masses[2], ls=":", c="silver",zorder=-1)
    xlim(0.8,omega_plot_max)
    ylim(bottom=0)
    # ylim(0,14000)
    xlabel("\$\\omega / m_1\$")
    ylabel("\$\\chi^{\\prime\\prime}(\\omega,k=0)\$")
    legend(loc = "upper right", numpoints=3, frameon=1, facecolor="white", fancybox=1, columnspacing=1, title="\$N=12, g=3.0, \\alpha=3.0, \\Gamma/J=0.1\$\n LR pbc")
    layout.nice_ticks()
    savefig(string(@__DIR__,"/figures/absorptionrate_adiabatic_comparison.pdf"))
end
