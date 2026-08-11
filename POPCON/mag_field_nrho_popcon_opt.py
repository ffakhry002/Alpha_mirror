"""
Identify the n20, Eb operating point that optimizes Rev/Vol
above a provided NWL and below a provided Pnbi for several Rm
at fixed Bm and injection angle
"""

import os
from multiprocessing import Pool

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


def get_idx_max_rev_per_vol(popcon, max_pnbi=Params.max_nbi_power_ftop, min_nwl=Params.min_NWL)-> tuple:
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
        if np.max(rev_per_vol_valid) < 0:
            return np.nan, np.nan
        i, j = np.unravel_index(np.argmax(rev_per_vol_valid), rev_per_vol_valid.shape)
        return i, j

def get_popcon_df(B_0, B_m, N_rho, min_rm_vac=4):
    """
    Returns dictionary of params of interest associated with popcon
    If no valid solution exists, dict contains NaN values
    """
    print(f"Getting POPCON for B_0={B_0:.2f}, B_m={B_m:.0f}, N_rho={N_rho:.0f}")
    nan_df = pd.DataFrame({
        'B_0_vac': [B_0],
        'B_m': [B_m],
        'N_rho': [N_rho],
        'Rev_per_vol_opt': [np.nan],
        'n_20_opt': [np.nan],
        'E_b100_opt': [np.nan],
        'P_nbi_opt': [np.nan],
        'NWL_at_opt_Rev_per_vol': [np.nan],
        'a_0_opt': [np.nan],
        'a_0_limit_opt': [np.nan],
        'L_mirror_opt': [np.nan],
        'V_plasma_opt': [np.nan],
        'a_w_opt': [np.nan],
        'B_w_opt': [np.nan],
        'q_w_opt': [np.nan]
    })
    if B_m / B_0 < min_rm_vac:
        return nan_df
    popcon = Popcon(B_0_vac=B_0, B_m=B_m, N_rho=N_rho)
    popcon.create_popcon()
    j, k = get_idx_max_rev_per_vol(popcon, max_pnbi=15., min_nwl=0.5)
    if np.isnan(j):
        return nan_df
    return pd.DataFrame({
        'B_0_vac': [popcon.B_0_vac],
        'B_m': [popcon.B_m],
        'N_rho': [popcon.N_rho],
        'Rev_per_vol_opt': [popcon.rev_per_vol[j,k]],
        'n_20_opt': [popcon.n_20_grid[j,k]],
        'E_b100_opt': [popcon.E_b100_grid[j,k]],
        'P_nbi_opt': [popcon.P_nbi[j,k]],
        'NWL_at_opt_Rev_per_vol': [popcon.NWL[j,k]],
        'a_0_opt': [popcon.a_0_min[j,k]],
        'a_0_limit_opt': [popcon.a_0_min_limit[j,k]],
        'L_mirror_opt': [popcon.L_mirror[j,k]],
        'V_plasma_opt': [popcon.V_plasma[j,k]],
        'a_w_opt': [popcon.a_w[j,k]],
        'B_w_opt': [popcon.B_w[j,k]],
        'q_w_opt': [popcon.q_w[j,k]]
    })


def popcon_scan(B_0_scan, B_m_scan, N_rho_scan, bypass=False):
    fn = f'mag_field_nrho_popcon_optimization.csv'
    print(fn)
    if os.path.exists(fn) and not bypass:
        print(f"POPCON scan already exists.\nReading results from {fn}")
        df = pd.read_csv(fn)
        return df
    scan = [
        (b0, bm, nrho)
        for nrho in N_rho_scan
        for bm in B_m_scan
        for b0 in B_0_scan
    ]
    with Pool() as pool:
        dfs = pool.starmap(get_popcon_df, scan)
    result_df = pd.concat(dfs)
    result_df = result_df.sort_values(by=['N_rho', 'B_m', 'B_0_vac'])
    result_df.to_csv(fn, index=False)
    return result_df

def bin_edges(centers):
    """Compute bin edges from bin centers."""
    mids = (centers[:-1] + centers[1:]) / 2
    left  = centers[0]  - (centers[1]  - centers[0])  / 2
    right = centers[-1] + (centers[-1] - centers[-2]) / 2
    return np.concatenate([[left], mids, [right]])

def make_histograms(df):
    plt.rcParams['font.size'] = 12
    nrhos = np.unique(df['N_rho'])
    df['Rev_per_vol_opt'] /= 1e6  # [$/yr/m^3] -> [$M/yr/m^3]
    fig, axs = plt.subplots(ncols=len(nrhos), nrows=1, figsize=(7, 6), sharex=True, sharey=True)
    B_m_scan     = np.unique(df['B_m'])
    B_0_vac_scan = np.unique(df['B_0_vac'])

    vmin = 0
    vmax = Params.max_rev_per_vol / 1e6
    norm = plt.Normalize(vmin=4000, vmax=vmax)
    cmap = plt.cm.viridis

    axs[0].set_yticks(B_0_vac_scan[0::2])
    axs[0].set_xlim(B_m_scan[0] - 1, B_m_scan[-1] + 1)
    fig.supxlabel(r'$B_m$ [T]', fontsize=14)

    for i, (nrho, ax) in enumerate(zip(nrhos, axs)):
        df_nrho = df[df['N_rho'] == nrho]
        pivot   = df_nrho.pivot(index='B_0_vac', columns='B_m', values='Rev_per_vol_opt')

        B_m_grid, B_0_vac_grid = np.meshgrid(pivot.columns, pivot.index)
        valid = ~np.isnan(pivot.values)
        missing = np.isnan(pivot.values)

        # Grid to show simulated values
        for edge in B_m_scan:
            ax.axvline(edge, color='grey', linewidth=0.8, linestyle='--')
        for edge in B_0_vac_scan:
            ax.axhline(edge, color='grey', linewidth=0.8, linestyle='--')

        # Colored scatter for valid points
        ax.scatter(
            B_m_grid[valid], B_0_vac_grid[valid],
            c=pivot.values[valid],
            cmap=cmap, norm=norm,
            s=160, marker='s', zorder=2,
        )

        # 'x' markers for missing runs
        ax.scatter(
            B_m_grid[missing], B_0_vac_grid[missing],
            marker='x', color='k', s=60, zorder=2, label='N/A',
        )

        # Show chosen point
        if nrho == 12:
            ax.scatter(22, 5.25,
                    facecolors='none',
                    edgecolors='orange',
                    s=240,           # slightly larger than data markers
                    linewidths=2.5,
                    marker='s',
                    zorder=3,
                    label='Alternate Design')
        if nrho == 15:
            ax.scatter(25, 6.0,
                    facecolors='none',
                    edgecolors='magenta',
                    s=240,           # slightly larger than data markers
                    linewidths=2.5,
                    marker='s',
                    zorder=3,
                    label='Chosen Design')

        #ax.set_xlabel(r'$B_m$ [T]', fontsize=14)
        if i == 0:                              # change 1: ylabel on leftmost panel only
            ax.set_ylabel(r'$B_{0,vac}$ [T]', fontsize=14)
        ax.set_title(r'$N_{\rho} =$ ' + f'{nrho}', fontsize=14)
        ax.set_xticks(B_m_scan)

    # ScalarMappable needed for colorbar since we no longer have a pcolormesh handle
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs[-1], label=r'$R/V_p\ [\$\mathrm{M/yr/m^{3}}]$')
    cbar.set_ticks(np.arange(0, 6e3, 500))

    plt.tight_layout()
    plt.show()
    return fig

if __name__=="__main__":
    plt.rcParams['font.size'] = 11
    B_0_scan = np.arange(2.5, 7.25, 0.25)
    B_m_scan = np.arange(22, 31, 3)
    N_rho_scan = np.array([10, 12, 15, 18])
    df = popcon_scan(B_0_scan, B_m_scan, N_rho_scan, bypass=False)
    fig = make_histograms(df)
    fig.savefig('mag_field_nrho_popcon_optimization.png')
