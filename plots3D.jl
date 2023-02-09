include("layout.jl")
using layout
using BSON
using PyPlot
using PyCall
@pyimport matplotlib.transforms as mpltrafo

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

### dummy
figure(1)
x=linspace(0,10)
y=x.^2
z=sin.(x)
plot3D(x,y,z)


figure(23)
ax23=subplot(111, projection="3d")
figure(220)
ax220=subplot(111, projection="3d")
figure(223)
ax223=subplot(111, projection="3d")

### ------------------------------------------   NN data   -----------------------
for j = 1:length(NN_files)
    NN_data =  BSON.load(string(@__DIR__,"/data/NN_data_"*NN_files[j]*".bson"))

    println("NN_data: ",NN_data[:info])
    N = NN_data[:N]; d = NN_data[:d]; num_states = NN_data[:num_states]; J = NN_data[:J]; h = NN_data[:h]; g = NN_data[:g]; Γ = NN_data[:Gamma]
    E_excitations_NN_obc = NN_data[:E_excitations_NN_obc]; E_excitations_NN_pbc = NN_data[:E_excitations_NN_pbc]
    E_ratios_NN_obc = NN_data[:E_ratios_NN_obc]; E_ratios_NN_pbc = NN_data[:E_ratios_NN_pbc]
    omega = NN_data[:omega]; susceptibilities = NN_data[:chi]
    omega_max = omega[end]/E_excitations_NN_pbc[2]
    omega_plot_max_ind = Int(floor(omega_plot_max/omega_max*length(omega)))

    ## absorption spectrum:
    if j==2
        figure(23)
        plot3D(omega[1:omega_plot_max_ind]/E_excitations_NN_pbc[2], g*ones(omega_plot_max_ind), real(susceptibilities)[1:omega_plot_max_ind]/maximum(real(susceptibilities)[1:omega_plot_max_ind]), c="C1", zorder=10-j)
    end
    if j==1
        figure(23)
        # ax23=subplot(111, projection="3d")
        # global ax23=subplot(111, projection="3d")
        plot3D(omega[1:omega_plot_max_ind]/E_excitations_NN_pbc[2], g*ones(omega_plot_max_ind), real(susceptibilities)[1:omega_plot_max_ind]/maximum(real(susceptibilities)[1:omega_plot_max_ind]), c="C0", zorder=10-j)
    end
    if j>=4
        figure(23)
        plot3D(omega[1:omega_plot_max_ind]/E_excitations_NN_pbc[2], g*ones(omega_plot_max_ind), real(susceptibilities)[1:omega_plot_max_ind]/maximum(real(susceptibilities)[1:omega_plot_max_ind]), c="C"*string(j-2), zorder=10-j)
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
    omega_plot_max_ind = Int(floor(omega_plot_max/omega_max*length(omega)))

    ## absorption spectrum:
    if 2 <= j <= 5 # alpha dependence
        figure(220)
        # ax220=subplot(111, projection="3d")
        # global ax220=subplot(111, projection="3d")
        # plot3D(omega[1:omega_plot_max_ind]/E_excitations_LR_pbc[2], alpha*ones(omega_plot_max_ind), real(susceptibilities)[1:omega_plot_max_ind], c="C"*string(j-1), zorder=10-j)
        plot3D(omega[1:omega_plot_max_ind]/E_excitations_LR_pbc[2], alpha*ones(omega_plot_max_ind), real(susceptibilities)[1:omega_plot_max_ind]/maximum(real(susceptibilities)[1:omega_plot_max_ind]), c="C"*string(j-1), zorder=10-j)
    end
    if j==2
        figure(223) # longitudinal field dependence
        plot3D(omega[1:omega_plot_max_ind]/E_excitations_LR_pbc[2], g*ones(omega_plot_max_ind), real(susceptibilities)[1:omega_plot_max_ind]/maximum(real(susceptibilities)[1:omega_plot_max_ind]), c="C1", zorder=10-j)
    end
    if j==1 # longitudinal field dependence
        figure(223)
        # ax223=subplot(111, projection="3d")
        # global ax223=subplot(111, projection="3d")
        plot3D(omega[1:omega_plot_max_ind]/E_excitations_LR_pbc[2], g*ones(omega_plot_max_ind), real(susceptibilities)[1:omega_plot_max_ind]/maximum(real(susceptibilities)[1:omega_plot_max_ind]), c="C0", zorder=10-j)
    end
    if j>=7 # longitudinal field dependence
        figure(223)
        plot3D(omega[1:omega_plot_max_ind]/E_excitations_LR_pbc[2], g*ones(omega_plot_max_ind), real(susceptibilities)[1:omega_plot_max_ind]/maximum(real(susceptibilities)[1:omega_plot_max_ind]), c="C"*string(j-5), zorder=10-j)
    end
end



##--- alpha dependence 3D:
figure(220)
for i in 1:5   plot3D(E8_masses[i]*ones(100),linspace(-0.4,3.4,100),zeros(100), ls="--", c="k",zorder=-1) end
plot3D((E8_masses[1]+E8_masses[2])*ones(100),linspace(-0.4,3.4,100),zeros(100), ls=":", c="grey",zorder=-1)
plot3D((E8_masses[1]+E8_masses[3])*ones(100),linspace(-0.4,3.4,100),zeros(100), ls=":", c="grey",zorder=-1)
xlim(0,omega_plot_max)
ylim(3.4,-0.4)
zlim(0,1.1)
# zlim(0,14000)
xticks([0.0,1.0,2.0,3.0])
yticks([3.0,2.0,1.0,0.0])
zticks([0,0.25,0.5,0.75,1])
xlabel("\$\\omega / m_1\$", labelpad=14)
ylabel("\$\\alpha\$", labelpad=12)
zlabel("\$\\chi^{\\prime\\prime}(\\omega,k=0)\$ (scaled)", labelpad=25)
ax220[:tick_params](axis="z", pad=13)
ax220[:set_position](mpltrafo.Bbox([[-0.07, 0.02], [0.95, 1.05]]))
savefig(string(@__DIR__,"/figures/absorptionrate_LR_alpha_dependence_3D.pdf"))


##--- longitudinal field dependence 3D:
## NN
figure(23)
for i in 1:5   plot3D(E8_masses[i]*ones(100),linspace(0.6,4.4,100),zeros(100), ls="--", c="k",zorder=-1) end
plot3D((E8_masses[1]+E8_masses[2])*ones(100),linspace(0.6,4.4,100),zeros(100), ls=":", c="grey",zorder=-1)
plot3D((E8_masses[1]+E8_masses[3])*ones(100),linspace(0.6,4.4,100),zeros(100), ls=":", c="grey",zorder=-1)
xlim(0,omega_plot_max)
ylim(4.4,0.6)
zlim(0,1.1)
xticks([0.0,1.0,2.0,3.0])
yticks([4.0,3.0,2.0,1.0])
zticks([0,0.25,0.5,0.75,1])
xlabel("\$\\omega / m_1\$", labelpad=14)
ylabel("\$g\$", labelpad=12)
zlabel("\$\\chi^{\\prime\\prime}(\\omega,k=0)\$ (scaled)", labelpad=25)
ax23[:tick_params](axis="z", pad=13)
ax23[:set_position](mpltrafo.Bbox([[-0.07, 0.02], [0.95, 1.05]]))
savefig(string(@__DIR__,"/figures/absorptionrate_NN_field_dependence_3D.pdf"))

## LR
figure(223)
# ax223=subplot(111, projection="3d")
for i in 1:5   plot3D(E8_masses[i]*ones(100),linspace(0.6,4.4,100),zeros(100), ls="--", c="k",zorder=-1) end
plot3D((E8_masses[1]+E8_masses[2])*ones(100),linspace(0.6,4.4,100),zeros(100), ls=":", c="grey",zorder=-1)
plot3D((E8_masses[1]+E8_masses[3])*ones(100),linspace(0.6,4.4,100),zeros(100), ls=":", c="grey",zorder=-1)
xlim(0,omega_plot_max)
ylim(4.4,0.6)
zlim(0,1.1)
# zlim(0,14000)
xticks([0.0,1.0,2.0,3.0])
yticks([4.0,3.0,2.0,1.0])
zticks([0,0.25,0.5,0.75,1])
# zticks([0,4000,8000,12000])
xlabel("\$\\omega / m_1\$", labelpad=14)
ylabel("\$g\$", labelpad=12)
zlabel("\$\\chi^{\\prime\\prime}(\\omega,k=0)\$ (scaled)", labelpad=25)
ax223[:tick_params](axis="z", pad=13)
ax223[:set_position](mpltrafo.Bbox([[-0.07, 0.02], [0.95, 1.05]]))
savefig(string(@__DIR__,"/figures/absorptionrate_LR_field_dependence_3D.pdf"))
