"""
Identify the n20, Eb operating point that optimizes Rev/Vol
above a provided NWL and below a provided Pnbi for several Rm
at fixed Bm and injection angle
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as pe
import pandas as pd
from pathlib import Path

# Import from our modular files
import POPCON.utils.equations as eqn
from POPCON.popcon import Popcon
from POPCON.params import Params


def get_idx_max_rev_per_vol(popcon, max_pnbi=Params.max_nbi_power_ftop, min_nwl=Params.min_NWL, min_rm_vac=4)-> tuple:
        """
        Returns the indices of the valid operating point in the POPCON
        that has the highest revenue per volume. 
        Can additionally specify a maximum Pnbi and minimum NWL for the selected point
        """
        # Find max Rev per volume over the valid region by making invalid points -inf
        mask_high_pnbi = popcon.P_nbi > max_pnbi
        mask_low_nwl = popcon.NWL < min_nwl
        mask_valid = ~ (mask_high_pnbi | mask_low_nwl | popcon.invalid)
        rev_per_vol_valid = np.where(mask_valid, popcon.rev_per_vol, -np.inf)
        if popcon.R_M_vac < min_rm_vac or np.max(rev_per_vol_valid) < 0:
            return np.nan, np.nan
        i, j = np.unravel_index(np.argmax(rev_per_vol_valid), rev_per_vol_valid.shape)
        return i, j

def popcon_scan(B_0_scan, B_max, bypass=False):
    fn = f'Rm_optimizaiton_Bm_{B_max:.0f}.csv'
    print(fn)
    if os.path.exists(fn) and not bypass:
        print(f"POPCON scan already exists.\nReading results from {fn}")
        df = pd.read_csv(fn)
        return df
    print(f"POPCON scan for $B_m = ${B_max:0f} T")
    dfs = []
    for b0 in B_0_scan:
        print(f'B_max: {B_max:.1f}, B_0_vac: {b0:0.2f}')
        popcon = Popcon(B_0_vac=b0, B_m=B_max)
        popcon.create_popcon()
        j, k = get_idx_max_rev_per_vol(popcon, max_pnbi=15., min_nwl=0.5)
        if np.isnan(j):
            df = pd.DataFrame({
                'B_0_vac': [popcon.B_0_vac],
                'B_m': [popcon.B_m],
                'N_rho': [popcon.N_rho],
                'Rev_per_vol_opt': [np.nan],
                'n_20_opt': [np.nan],
                'E_b100_opt': [np.nan],
                'NWL_at_opt_Rev_per_vol': [np.nan],
                'a_0_opt': [np.nan],
                'a_0_limit_opt': [np.nan],
            })
        else:
            df = pd.DataFrame({
                'B_0_vac': [b0],
                'B_m': [popcon.B_m],
                'N_rho': [popcon.N_rho],
                'Rev_per_vol_opt': [popcon.rev_per_vol[j,k]],
                'n_20_opt': [popcon.n_20_grid[j,k]],
                'E_b100_opt': [popcon.E_b100_grid[j,k]],
                'NWL_at_opt_Rev_per_vol': [popcon.NWL[j,k]],
                'a_0_opt': [popcon.a_0_min[j,k]],
                'a_0_limit_opt': [popcon.a_0_min_limit[j,k]],
            })
        dfs.append(df)
    result_df = pd.concat(dfs) 
    result_df.to_csv(fn, index=False)
    return result_df


if __name__=="__main__":
    plt.rcParams['font.size'] = 11
    B_0_scan = np.arange(2.5, 7.25, 0.25)
    df_22 = popcon_scan(B_0_scan, B_max=22, bypass=True)
    df_25 = popcon_scan(B_0_scan, B_max=25, bypass=True)
    df_28 = popcon_scan(B_0_scan, B_max=28, bypass=True)
    dfs = [df_22, df_25, df_28]
    labels = ['$B_m = 22$ T', '$B_m = 25$ T', '$B_m = 28$ T']
    cmap = plt.get_cmap('Reds')
    Bm_colors = [cmap(x) for x in np.linspace(0.3, 0.9, 3)]
    for df, l, c in zip(dfs, labels, Bm_colors):
        plt.plot(df['B_0_vac'], df['Rev_per_vol_opt']/1e6, marker='o', label=l, c=c)
    plt.xlabel(r'$B_{0,vac}$ [T]', fontsize=14)
    plt.ylabel(r'$R/V_p$ [\$M/yr/m$^{3}$]', fontsize=14)
    plt.ylim(0, 6000)
    plt.xticks(np.arange(2.5, 7.5, 0.5))
    plt.xlim(2.5, 7)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('rm_optimization.png')
    plt.show()
