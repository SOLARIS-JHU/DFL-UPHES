import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import dill as pickle
from pathlib import Path

script_dir = Path(__file__).parent
df = pd.read_csv(script_dir / 'Mod_Francis_joint.csv', index_col=0)
power_all = df.columns.astype(float).values
head_all  = df.index.astype(float).values
flow_all  = df.values.astype(float)

mask_pump    = power_all <  0
mask_turbine = power_all >  0
power_pump    = power_all[mask_pump]
flow_pump     = flow_all[:, mask_pump]
power_turbine = power_all[mask_turbine]
flow_turbine  = flow_all[:, mask_turbine]

P_p, H_p = np.meshgrid(power_pump,    head_all)
P_t, H_t = np.meshgrid(power_turbine, head_all)

vmin, vmax = np.nanmin(flow_all), np.nanmax(flow_all)
cmap = 'nipy_spectral'

def find_corners(flow, P, H, mode_name):
    corners = []
    for i in [0, -1]:
        row = flow[i, :]
        valid = ~np.isnan(row)
        j_min = np.argmax(valid)
        j_max = len(valid) - 1 - np.argmax(valid[::-1])
        corners.extend([(i, j_min), (i, j_max)])
    corners = list(dict.fromkeys(corners))
    print(f"{mode_name} corners:")
    for i, j in corners:
        print(f"  index ({i},{j}) -> Power={P[i,j]:.2f} MW, Head={H[i,j]:.2f} m, Flow={flow[i,j]:.4f} m3/s")
    return corners

pump_corners  = find_corners(flow_pump,    P_p, H_p, "Pump mode")
turb_corners = find_corners(flow_turbine, P_t, H_t, "Turbine mode")

i_mid, j_mid = len(head_all)//2, len(power_pump)//2
x_mp, y_mp, z_mp = P_p[i_mid,j_mid], H_p[i_mid,j_mid], flow_pump[i_mid,j_mid]
i_mid_t, j_mid_t = len(head_all)//2, len(power_turbine)//2
x_mt, y_mt, z_mt = P_t[i_mid_t,j_mid_t], H_t[i_mid_t,j_mid_t], flow_turbine[i_mid_t,j_mid_t]
i_mid_i = len(head_all)//2
x_mi, y_mi, z_mi = 0.0, head_all[i_mid_i], 0.0

fig = plt.figure(figsize=(8,6), dpi=300)
ax  = fig.add_subplot(111, projection='3d')

dp = np.ptp(power_all)
dq = vmax - vmin
ax.set_box_aspect((dp, dp, dq))

scale = 0.9 
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, scale, 1]))

ax.plot_surface(P_p, H_p, flow_pump,    rcount=100, ccount=100,
                cmap=cmap, vmin=vmin, vmax=vmax, antialiased=False, alpha=0.9)
ax.plot_surface(P_t, H_t, flow_turbine, rcount=100, ccount=100,
                cmap=cmap, vmin=vmin, vmax=vmax, antialiased=False, alpha=0.9)

idle_color = plt.cm.get_cmap(cmap)((0 - vmin)/(vmax - vmin))
ax.plot(np.zeros_like(head_all), head_all, np.zeros_like(head_all),
        color=idle_color, alpha=0.5, lw=2)

levels3d = np.linspace(vmin, vmax, 200)
ax.contourf(P_p, H_p, flow_pump,    zdir='z', offset=vmin, levels=levels3d,
            cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.6)
ax.contourf(P_t, H_t, flow_turbine, zdir='z', offset=vmin, levels=levels3d,
            cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.6)

for i,j in pump_corners:
    ax.plot([P_p[i,j]]*2, [H_p[i,j]]*2, [flow_pump[i,j], vmin],
            color='grey', linestyle='--', linewidth=1)
for i,j in turb_corners:
    ax.plot([P_t[i,j]]*2, [H_t[i,j]]*2, [flow_turbine[i,j], vmin],
            color='grey', linestyle='--', linewidth=1)

offset=0.15
lx, ly, lz = x_mp-dp*offset, y_mp, z_mp+dq*offset
ax.plot([x_mp,lx],[y_mp,ly],[z_mp,lz],'k',zorder=10)
ax.text(lx,ly,lz,'Pump Mode',ha='center',zorder=10)

lx, ly, lz = x_mt+dp*offset, y_mt, z_mt+dq*offset
ax.plot([x_mt,lx],[y_mt,ly],[z_mt,lz],'k',zorder=10)
ax.text(lx,ly,lz,'Turbine Mode',ha='center',zorder=10)

lx, ly, lz = x_mi, y_mi+dp*offset, z_mi+dq*offset
ax.plot([x_mi,lx],[y_mi,ly],[z_mi,lz],'k',zorder=10)
ax.text(lx,ly,lz,'Idle Mode',ha='center',zorder=10)

ax.set_xlabel('Power (MW)')
ax.set_ylabel('Head (m)')
ax.set_zlabel('Flow Rate (m³/s)')
ax.view_init(elev=30, azim=225)

mapp = plt.cm.ScalarMappable(norm=plt.Normalize(vmin,vmax), cmap=cmap)
fig.colorbar(mapp, ax=ax, shrink=0.6, pad=0.1).set_label('Flow Rate (m³/s)')
plt.tight_layout()
plt.savefig(script_dir / 'francis_turbine_3d_visualization.pdf', format='pdf', bbox_inches='tight')
plt.show()

with open(script_dir.parent.parent / 'preprocess.pkl', 'rb') as f:
    (v_low_h_coeffs, h_v_coeffs, v_low_to_h_fitted, v_low_h_poly, h_vlow_coeff_lin, 
     coefs_tur_lin, intercept_tur_lin, coefs_pump_lin, intercept_pump_lin, 
     predict_q_linear_tur, predict_q_linear_pump, h_to_v_low_lin, h_fit, 
     neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, h_v_coeffs, 
     DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly, 
     neg_min, neg_max, pos_min, pos_max, prepare_and_fit_model, 
     get_UPC_bound, LR_UPC_bound) = pickle.load(f)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.01,
    'legend.framealpha': 0.95,
    'legend.edgecolor': 'black',
    'legend.fancybox': False
})

head_all_fine = np.linspace(head_all.min(), head_all.max(), len(head_all) * 5)

P_pump_fine, H_pump_fine = np.meshgrid(power_pump, head_all_fine)
P_turb_fine, H_turb_fine = np.meshgrid(power_turbine, head_all_fine)

p_pump_tensor = torch.tensor(P_pump_fine, dtype=torch.float32)
h_pump_tensor = torch.tensor(H_pump_fine, dtype=torch.float32)
q_pump_fine = predict_q_poly(p_pump_tensor, h_pump_tensor).numpy()

p_turb_tensor = torch.tensor(P_turb_fine, dtype=torch.float32)
h_turb_tensor = torch.tensor(H_turb_fine, dtype=torch.float32)
q_turb_fine = predict_q_poly(p_turb_tensor, h_turb_tensor).numpy()

minp_fine = neg_min(h_pump_tensor).numpy()
maxp_fine = neg_max(h_pump_tensor).numpy()
mask_pump_valid = (P_pump_fine >= minp_fine) & (P_pump_fine <= maxp_fine)
q_pump_fine[~mask_pump_valid] = np.nan

mint_fine = pos_min(h_turb_tensor).numpy()
maxt_fine = pos_max(h_turb_tensor).numpy()
mask_turb_valid = (P_turb_fine >= mint_fine) & (P_turb_fine <= maxt_fine)
q_turb_fine[~mask_turb_valid] = np.nan

fig, (ax_pump, ax_turb) = plt.subplots(1, 2, figsize=(6, 2), dpi=600, sharey=True)

levels2d_fine = np.linspace(vmin, vmax, 300)

cf_pump = ax_pump.contourf(P_pump_fine, H_pump_fine, q_pump_fine, 
                            levels=levels2d_fine, cmap=cmap, vmin=vmin, vmax=vmax)
ct_pump = ax_pump.contour(P_pump_fine, H_pump_fine, q_pump_fine, linestyles='dashed',
                           levels=15, colors='w', linewidths=0.4, alpha=0.6)
ax_pump.clabel(ct_pump, fmt='%1.1f', fontsize=5)

ax_pump.set_xlim([power_pump.min(), -2.5])
ax_pump.set_ylabel('Head (m)')
ax_pump.spines['right'].set_visible(False)

cf_turb = ax_turb.contourf(P_turb_fine, H_turb_fine, q_turb_fine,
                            levels=levels2d_fine, cmap=cmap, vmin=vmin, vmax=vmax)
ct_turb = ax_turb.contour(P_turb_fine, H_turb_fine, q_turb_fine, linestyles='dashed',
                           levels=25, colors='k', linewidths=0.4, alpha=0.6)
ax_turb.clabel(ct_turb, fmt='%1.1f', fontsize=5)

ax_turb.set_xlim([1.5, power_turbine.max()])
ax_turb.spines['left'].set_visible(False)
ax_turb.tick_params(left=False)

ax_pump.set_xticks(np.arange(np.floor(power_pump.min()), -2, 1.0))
ax_turb.set_xticks(np.arange(1, np.ceil(power_turbine.max())+1, 1.0))

ax_pump.grid(True, which='major', alpha=0.4, linestyle='-', linewidth=0.6, color='gray')
ax_turb.grid(True, which='major', alpha=0.4, linestyle='-', linewidth=0.6, color='gray')

d = 0.015
kwargs = dict(transform=ax_pump.transAxes, color='k', clip_on=False, linewidth=0.8)
ax_pump.plot((1-d, 1+d), (-d, +d), **kwargs)
ax_pump.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

kwargs.update(transform=ax_turb.transAxes)
ax_turb.plot((-d, +d), (-d, +d), **kwargs)
ax_turb.plot((-d, +d), (1-d, 1+d), **kwargs)

plt.subplots_adjust(wspace=0.05)

fig.text(0.45, -0.03, 'Power (MW)', ha='center', fontsize=8) 

cbar = fig.colorbar(cf_turb, ax=[ax_pump, ax_turb], label='Flow Rate (m³/s)', 
                    pad=0.02, aspect=30, format='%.1f')

plt.savefig(script_dir / 'francis_turbine_polynomial_combined_contours.pdf',
            format='pdf', bbox_inches='tight', pad_inches=0.01)
plt.savefig(script_dir / 'francis_turbine_polynomial_combined_contours.png',
            format='png', bbox_inches='tight', pad_inches=0.01)
print("Saved: francis_turbine_polynomial_combined_contours.pdf/.png")
plt.show()
