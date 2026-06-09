path_to_adaptivesampling = "/adaptive_sampling"

import os
import sys
sys.path.append(path_to_adaptivesampling)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adaptive_sampling.processing_tools import mbar
from adaptive_sampling import units
from adaptive_sampling.processing_tools import utils

# Parameters
T = 450.0  # Temperature (K)
beta=1/(T*8.314e-3) #in kJ/mol 
ev_kjmol = 96.4853
dval = 0.25
min_val =-6.611871400257606 #from umbrella windows
max_val = -min_val
targets = np.arange(min_val, max_val + dval, dval)
device='cpu'#0

#*************Load directory of npz files with energy values of frames*************************

#data_directory='/home/gridsan/smajumdar/jobs/completed/april9_umb_0.8ns_55h2o_1li/april9_10000_downsampled_files/energy_frames_npz_uma-s-1/'
#data_directory='/home/gridsan/smajumdar/jobs/completed/april9_umb_0.8ns_55h2o_1li/april9_10000_downsampled_files/energy_frames_npz_mace_Li_finetuned'
#data_directory='/home/gridsan/smajumdar/jobs/completed/april9_umb_0.8ns_55h2o_1li/april9_10000_downsampled_files/energy_frames_npz_orb_v3_conservative_inf_omat/'
#data_directory='/home/gridsan/smajumdar/jobs/completed/april9_umb_0.8ns_55h2o_1li/april9_10000_downsampled_files/energy_frames_npz_mace-mpa-0/'
data_directory="/home/gridsan/smajumdar/jobs/completed/april9_umb_0.8ns_55h2o_1li/april9_10000_downsampled_files/energy_frames_npz_mace-matpes/"
#data_directory="/home/gridsan/smajumdar/jobs/completed/april9_umb_0.8ns_55h2o_1li/april9_10000_downsampled_files/energy_frames_npz_mace-mp0-d3/"
#data_directory="/home/gridsan/smajumdar/jobs/completed/april9_umb_0.8ns_55h2o_1li/pet-data/energy_frames_npz_pet-mad_downsampled"


# Containers
trajs = []
pot_eners_A = []  # Model A energies, MACE-mp0
pot_eners_B = []  # Model B energies
diff_AB = [] #Model A energy - Model B energy for each frame
valid_targets=[]
com_ext=[]
# Load data for each umbrella window
for target in targets:
    try:
        # Load Model B data from .npz
        npz = np.load(f"{data_directory}/selected_frames_window_{target}.npz")
        frame_indices_all = npz['frame_indices']           # shape [N_k]
        traj_filtered_all = npz['traj_filtered']           # CV values, shape [N_k]
        energies_B_all = npz['pot_energy_B_filtered']      # Model B energies, shape [N_k]

        energies_A_all=npz['pot_energy_A_filtered']
        N_k = len(frame_indices_all)
        n_sample = min(10000, N_k)
        chosen = np.random.choice(N_k, size=n_sample, replace=False)

        # Subset the arrays
        frame_indices = frame_indices_all[chosen]
        traj_filtered = traj_filtered_all[chosen]
        energies_B_kjmol = energies_B_all[chosen]*ev_kjmol
        energies_A_kjmol = energies_A_all[chosen]*ev_kjmol
        

        # Append to lists
        trajs.append(traj_filtered)
        pot_eners_A.append(energies_A_kjmol)
        pot_eners_B.append(energies_B_kjmol)
        valid_targets.append(target)
     
    except Exception as e:
        print(e)

print("Data reading over")

all_frames = trajs[0]
for traj in trajs[1:]:
    all_frames = np.concatenate((all_frames, traj))

# Build metadata for bias potentials
n_states = len(valid_targets)
meta_f = np.zeros((n_states, 3))
meta_f[:,1] = valid_targets
# spring constant k = (k_B T)/(sigma^2); assume sigma = dval/1.5
meta_f[:,2] = (8.314 * T / 1000.0) / ((dval/1.5)**2)



fig, ax = plt.subplots(1, sharex=True)

for traj, target in zip(trajs, valid_targets):
    ax.hist(traj, alpha=0.5, bins=50)#,range=(target-3*dval, target+3*dval))

ax.set_xlim(-7.5, 7.5)
ax.set_ylim(0, 3000)
xticks = np.arange(-7.5, 7.5 + 1e-9, 2.5)
ax.set_xticks(xticks)
yticks=np.arange(0,3000+1e-9,500)
ax.set_yticks(yticks)
ax.spines['bottom'].set_linewidth(3)
ax.spines['top'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(3)
ax.tick_params(axis='y', length=6, width=3,
               labelsize=20, pad=10, direction='in')
ax.tick_params(axis='x', length=6, width=3,
               labelsize=20, pad=10, direction='in')
ax.set_ylabel(r'$\mathrm{Density}$', fontsize=20)#ax.set_xlabel(r'CV', fontsize=20)
ax.set_xlabel(r'$\mathrm{z}\ (\mathrm{\AA})$', fontsize=20)

fig.tight_layout()
#plt.savefig('/home/gridsan/smajumdar/jobs/completed/april9_umb_0.8ns_55h2o_1li/april9_10000_downsampled_files/energy_frames_npz_mace-matpes/plots_mace-matpes/reweighted_umbrella_density_mace-matpes.pdf', dpi=600)
#plt.savefig('/home/gridsan/smajumdar/jobs/completed/april9_umb_0.8ns_55h2o_1li/pet-data/reweighted_umbrella_density_pet.pdf', dpi=600)


##Computing the weights corresponding to potential B
all_UA = np.concatenate(pot_eners_A)
all_UB = np.concatenate(pot_eners_B)

diff_AB=all_UB-all_UA #UB-UA
rel_diff_AB=diff_AB-diff_AB.min()#to account for offset
print(f"Energy Diff Stats (kJ/mol): Min={rel_diff_AB.min():.2f}, Max={rel_diff_AB.max():.2f}, Mean={rel_diff_AB.mean():.2f}")
print(rel_diff_AB)


exp_U_B, frames_per_traj = mbar.build_boltzmann(
    traj_list=trajs,
    meta_f=meta_f,
    dU_list=None,
    equil_temp=450.0
)

print("Any NaNs in exp_U?", np.isnan(exp_U_B).any())
print("Any Infs in exp_U?", np.isinf(exp_U_B).any())

# Unbiased weights from MBAR
weights = mbar.run_mbar(
    exp_U_B,
    frames_per_traj,
    outfreq=100,
    conv=1e-6,
    conv_errvec=None,
    max_iter=int(1e6),
    device='cpu',  # or GPU index
)

weights_B=weights*np.exp(-beta*rel_diff_AB) #refer to derivation in article


minimum = all_frames.min()
maximum = all_frames.max()
bin_width = dval
grid = np.arange(minimum, maximum, bin_width)


print(all_frames.min())

pmf, rho = mbar.pmf_from_weights(grid, all_frames, weights_B, equil_temp=450.0) 
pmf -= pmf.min()

colors = ['#0000a5', '#ac5775', '#ffaf1e','#228B22', '#1f77b4','#808080', '#FF5349'] 
# PMF: free energy profile along CV
fig, axs = plt.subplots(1, figsize=(8, 6))


axs.plot(grid, pmf, linewidth=4,color=colors[0])

axs.tick_params(axis='y', length=8, width=4,
                labelsize=20, pad=10, direction='in')
axs.tick_params(axis='x', length=8, width=4,
                labelsize=20, pad=10, direction='in')

axs.set_xlabel(r'$\mathrm{z}\ (\mathrm{\AA})$', fontsize=20)
axs.set_ylabel(r'$\mathrm{A(z)}\ (\mathrm{kJ}\,\mathrm{mol}^{-1})$',fontsize=20)
axs.spines['bottom'].set_linewidth(3)
axs.spines['top'].set_linewidth(3)
axs.spines['left'].set_linewidth(3)
axs.spines['right'].set_linewidth(3)

#plt.savefig('/home/gridsan/smajumdar/jobs/completed/april9_umb_0.8ns_55h2o_1li/april9_10000_downsampled_files/energy_frames_npz_mace-matpes/plots_mace-matpes/pmf_mace-matpes_dec2025_raw.png')
np.savez_compressed(
               "/home/gridsan/smajumdar/jobs/completed/april9_umb_0.8ns_55h2o_1li/april9_10000_downsampled_files/energy_frames_npz_mace-matpes/pmf_mace-matpes_dec2025.npz",
               pmf=np.array(pmf),
               grid=np.array(grid)
            )
