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


def popcon_scan(B_0_scan, B_max, bypass=False):
    fn = f'Rm_optimizaiton_Bm_{B_max:.0f}.csv'
    print(fn)
    if os.path.exists(fn) and not bypass:
        print(f"POPCON scan already exists.\nReading results from {fn}")
        df = pd.read_csv(fn)
        return df
    print(f"POPCON scan for $B_m = ${B_max:0f} T")
    max_rev_scan = np.zeros_like(B_0_scan)
    n_20_opt_scan = np.zeros_like(B_0_scan)
    E_b100_opt_scan = np.zeros_like(B_0_scan)
    nwl_opt_scan = np.zeros_like(B_0_scan)
    a_0_opt_scan = np.zeros_like(B_0_scan)
    a_0_limit_opt_scan = []
    for i, b0 in enumerate(B_0_scan):
        print(f'B_max: {B_max:.1f}, B_0_vac: {b0:0.2f}')
        popcon = Popcon(B_0_vac=b0, B_m=B_max)
        popcon.create_popcon()
        # TODO: This needs to return NaNs when there's no point!
        j, k = popcon.get_idx_max_rev_per_vol(max_pnbi=15., min_nwl=0.5)
        max_rev_scan[i] = popcon.rev_per_vol[j,k]
        n_20_opt_scan[i] = popcon.n_20_grid[j,k]
        E_b100_opt_scan[i] = popcon.E_b100_grid[j,k]
        nwl_opt_scan[i] = popcon.NWL[j,k]
        a_0_opt_scan[i] = popcon.a_0_min[j,k]
        a_0_limit_opt_scan.append(popcon.a_0_min_limit[j,k])
    df = pd.DataFrame({
        'B_0': B_0_scan,
        'Rev_per_vol_opt': max_rev_scan,
        'n_20_opt': n_20_opt_scan,
        'E_b100_opt': E_b100_opt_scan,
        'NWL_at_opt_Rev_per_vol': nwl_opt_scan,
        'a_0_opt': a_0_opt_scan,
        'a_0_limit_opt': np.array(a_0_limit_opt_scan, dtype=str),
    })
    df.to_csv(fn, index=False)
    return df


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
        plt.plot(df['B_0'], df['Rev_per_vol_opt']/1e6, marker='o', label=l, c=c)
    plt.xlabel(r'$B_0$ [T]', fontsize=14)
    plt.ylabel(r'$R/V_p$ [\$M/yr/m$^{3}$]', fontsize=14)
    plt.ylim(0, 6000)
    plt.xticks(np.arange(2.5, 7.5, 0.5))
    plt.xlim(2.5, 7)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('rm_optimization.png')
    plt.show()
