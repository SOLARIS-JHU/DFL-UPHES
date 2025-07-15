#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D + 2D Contour Visualization of Reversible Francis Turbine UPC
– Power (–10→10 MW) & Head (50→100 m) axes equal length.
– Contours projected onto the power–head plane at min(flow).
– Dashed projection lines from the 4 true corners of both pump & turbine surfaces.
– Prints out those corner coordinates for verification.
– 2D contour heatmaps with isolines.
– Uses "nipy_spectral" for a vivid palette.
– Adds 3D labels for Pump Mode, Turbine Mode, Idle Mode with leader lines from surface midpoints.
– Higher-resolution color gradients: increased mesh sampling & more contour levels.
– Saves all figures as SVG files.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import torch
import dill as pickle

# === 1) Load data ===
df = pd.read_csv('Mod_Francis_joint.csv', index_col=0)
power_all = df.columns.astype(float).values
head_all  = df.index.astype(float).values
flow_all  = df.values.astype(float)

# === 2) Split pump vs turbine ===
mask_pump    = power_all <  0
mask_turbine = power_all >  0
power_pump    = power_all[mask_pump]
flow_pump     = flow_all[:, mask_pump]
power_turbine = power_all[mask_turbine]
flow_turbine  = flow_all[:, mask_turbine]

# === 3) Build meshgrids ===
P_p, H_p = np.meshgrid(power_pump,    head_all)
P_t, H_t = np.meshgrid(power_turbine, head_all)

# === 4) Colormap & common limits ===
vmin, vmax = np.nanmin(flow_all), np.nanmax(flow_all)
cmap = 'nipy_spectral'

# === 5) Determine true corners of valid surface ===
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

# === 6) Compute midpoints for 3D labels ===
i_mid, j_mid = len(head_all)//2, len(power_pump)//2
x_mp, y_mp, z_mp = P_p[i_mid,j_mid], H_p[i_mid,j_mid], flow_pump[i_mid,j_mid]
i_mid_t, j_mid_t = len(head_all)//2, len(power_turbine)//2
x_mt, y_mt, z_mt = P_t[i_mid_t,j_mid_t], H_t[i_mid_t,j_mid_t], flow_turbine[i_mid_t,j_mid_t]
i_mid_i = len(head_all)//2
x_mi, y_mi, z_mi = 0.0, head_all[i_mid_i], 0.0

# === 7) Figure 1: 3D surfaces + projections + labels ===
fig = plt.figure(figsize=(8,6), dpi=300)
ax  = fig.add_subplot(111, projection='3d')

# Equal-aspect for Power & Head
dp = power_all.ptp()
dq = vmax - vmin
ax.set_box_aspect((dp, dp, dq))

# Adjust z-axis scale for better visibility
scale = 0.9 
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, scale, 1]))

# High-res surface sampling
ax.plot_surface(P_p, H_p, flow_pump,    rcount=100, ccount=100,
                cmap=cmap, vmin=vmin, vmax=vmax, antialiased=False, alpha=0.9)
ax.plot_surface(P_t, H_t, flow_turbine, rcount=100, ccount=100,
                cmap=cmap, vmin=vmin, vmax=vmax, antialiased=False, alpha=0.9)

# Idle line
idle_color = plt.cm.get_cmap(cmap)((0 - vmin)/(vmax - vmin))
ax.plot(np.zeros_like(head_all), head_all, np.zeros_like(head_all),
        color=idle_color, alpha=0.5, lw=2)

# Contour projections
levels3d = np.linspace(vmin, vmax, 200)
ax.contourf(P_p, H_p, flow_pump,    zdir='z', offset=vmin, levels=levels3d,
            cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.6)
ax.contourf(P_t, H_t, flow_turbine, zdir='z', offset=vmin, levels=levels3d,
            cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.6)

# Projection lines
for i,j in pump_corners:
    ax.plot([P_p[i,j]]*2, [H_p[i,j]]*2, [flow_pump[i,j], vmin],
            color='grey', linestyle='--', linewidth=1)
for i,j in turb_corners:
    ax.plot([P_t[i,j]]*2, [H_t[i,j]]*2, [flow_turbine[i,j], vmin],
            color='grey', linestyle='--', linewidth=1)

# Labels with leader lines
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

# Axis labels & view
ax.set_xlabel('Power (MW)')
ax.set_ylabel('Head (m)')
ax.set_zlabel('Flow Rate (m³/s)')
ax.view_init(elev=30, azim=225)

# Colorbar
mapp = plt.cm.ScalarMappable(norm=plt.Normalize(vmin,vmax), cmap=cmap)
fig.colorbar(mapp, ax=ax, shrink=0.6, pad=0.1).set_label('Flow Rate (m³/s)')
plt.tight_layout()
plt.savefig('francis_turbine_3d_visualization.svg', format='svg', bbox_inches='tight')
plt.show()

# === 8) 2D Contour Plots ===
levels2d = np.linspace(vmin, vmax, 200)
for Pg, Hg, Fa, title in [(P_p,H_p,flow_pump,'Pump Mode'), (P_t,H_t,flow_turbine,'Turbine Mode')]:
    fig, ax = plt.subplots(figsize=(6,5), dpi=300)
    cf = ax.contourf(Pg, Hg, Fa, levels=levels2d, cmap=cmap, vmin=vmin, vmax=vmax)
    ct = ax.contour(Pg, Hg, Fa, levels=20, colors='k', linewidths=0.8)
    ax.clabel(ct, fmt='%1.1f', fontsize=8)
    ax.set_title(f"{title} Flow Rate Contours")
    ax.set_xlabel('Power (MW)')
    ax.set_ylabel('Head (m)')
    fig.colorbar(cf, ax=ax, label='Flow Rate (m³/s)')
    plt.tight_layout()
    
    filename = f"francis_turbine_{title.lower().replace(' ', '_')}_contours.svg"
    plt.savefig(filename, format='svg', bbox_inches='tight')
    plt.show()

#%% Global Linear Approximation 3D Visualization
"""
Visualize the global linear approximation surfaces alongside original UPC data.
- Loads linear models from preprocess.pkl
- Plots original surfaces in light grey
- Overlays linearized surfaces with same "nipy_spectral" colormap
- No projection contours for clarity
"""
device = torch.device("cpu")
# Load preprocessed linear models & data
with open('preprocess.pkl', 'rb') as f:
    v_low_h_coeffs, h_v_coeffs, v_low_to_h_fitted, v_low_h_poly, h_vlow_coeff_lin, coefs_tur_lin, intercept_tur_lin, coefs_pump_lin, intercept_pump_lin, predict_q_linear_tur,predict_q_linear_pump, h_to_v_low_lin, h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, h_v_coeffs, DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly, neg_min, neg_max, pos_min, pos_max, prepare_and_fit_model, get_UPC_bound, LR_UPC_bound = pickle.load(f)

h_min, h_max = np.min(h_fit), np.max(h_fit)
head_lin = head_all[(head_all >= h_min) & (head_all <= h_max)]
P_p_lin, H_p_lin = np.meshgrid(power_pump, head_lin)
P_t_lin, H_t_lin = np.meshgrid(power_turbine, head_lin)

p_p_lin_tensor = torch.tensor(P_p_lin, dtype=torch.float32)
h_p_lin_tensor = torch.tensor(H_p_lin, dtype=torch.float32)
q_lin_pump = predict_q_linear_pump(p_p_lin_tensor, h_p_lin_tensor).numpy()

p_t_lin_tensor = torch.tensor(P_t_lin, dtype=torch.float32)
h_t_lin_tensor = torch.tensor(H_t_lin, dtype=torch.float32)
q_lin_turbine = predict_q_linear_tur(p_t_lin_tensor, h_t_lin_tensor).numpy()

neg_min_line = np.polyval(neg_min_fit, H_p_lin)
neg_max_line = np.polyval(neg_max_fit, H_p_lin)
mask_pump_lin = (P_p_lin >= neg_min_line) & (P_p_lin <= neg_max_line)
q_lin_pump[~mask_pump_lin] = np.nan

pos_min_line = np.polyval(pos_min_fit, H_t_lin)
pos_max_line = np.polyval(pos_max_fit, H_t_lin)
mask_turb_lin = (P_t_lin >= pos_min_line) & (P_t_lin <= pos_max_line)
q_lin_turbine[~mask_turb_lin] = np.nan

fig_lin = plt.figure(figsize=(8,6), dpi=300)
ax_lin = fig_lin.add_subplot(111, projection='3d')

dp = power_all.ptp()
dz = vmax - vmin
ax_lin.set_box_aspect((dp, dp, dz))
ax_lin.get_proj = lambda: np.dot(Axes3D.get_proj(ax_lin), np.diag([1,1,0.9,1]))

# Original surfaces
taf_kw = dict(color='lightgrey', alpha=0.5, rcount=100, ccount=100, antialiased=False)
ax_lin.plot_surface(P_p, H_p, flow_pump,    **taf_kw)
ax_lin.plot_surface(P_t, H_t, flow_turbine, **taf_kw)

# Linear surfaces
ax_lin.plot_surface(P_p_lin, H_p_lin, q_lin_pump,    rcount=100, ccount=100,
                    cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.9)
ax_lin.plot_surface(P_t_lin, H_t_lin, q_lin_turbine, rcount=100, ccount=100,
                    cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.9)

# Idle line
idle_color_lin = plt.cm.get_cmap(cmap)((0 - vmin)/(vmax - vmin))
ax_lin.plot(np.zeros_like(head_lin), head_lin, np.zeros_like(head_lin),
            color=idle_color_lin, alpha=0.5, lw=2)

# Axes & colorbar
ax_lin.set_xlabel('Power (MW)')
ax_lin.set_ylabel('Head (m)')
ax_lin.set_zlabel('Flow Rate (m³/s)')
ax_lin.view_init(elev=30, azim=225)

mapp_lin = plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=cmap)
fig_lin.colorbar(mapp_lin, ax=ax_lin, shrink=0.6, pad=0.1).set_label('Flow Rate (m³/s)')

plt.tight_layout()
plt.savefig('francis_turbine_linear_3d_visualization.svg', format='svg', bbox_inches='tight')
plt.show()

#%% Polynomial Approximation 3D Visualization
"""
Visualize original UPC and polynomial-fit surfaces with proper depth sorting
so that whichever surface is closer to the camera at each patch is rendered on top.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# assume power_all, power_pump, power_turbine, head_all,
# flow_pump, flow_turbine, predict_q_poly,
# neg_min, neg_max, pos_min, pos_max, cmap, vmin, vmax
# have all been defined/imported already in this session

# -- 1) build meshes for pump & turbine polynomial predictions --
P_p_poly, H_p_poly = np.meshgrid(power_pump, head_all)
P_t_poly, H_t_poly = np.meshgrid(power_turbine, head_all)

pp = torch.tensor(P_p_poly, dtype=torch.float32)
hp = torch.tensor(H_p_poly, dtype=torch.float32)
qt_pump = predict_q_poly(pp, hp).numpy()

pt = torch.tensor(P_t_poly, dtype=torch.float32)
ht = torch.tensor(H_t_poly, dtype=torch.float32)
qt_turb = predict_q_poly(pt, ht).numpy()

# -- 2) mask outside fitted boundaries --
minp = neg_min(hp).numpy(); maxp = neg_max(hp).numpy()
mask_p = (P_p_poly >= minp) & (P_p_poly <= maxp)
qt_pump[~mask_p] = np.nan

mint = pos_min(ht).numpy(); maxt = pos_max(ht).numpy()
mask_t = (P_t_poly >= mint) & (P_t_poly <= maxt)
qt_turb[~mask_t] = np.nan

# -- 3) plot both surfaces with depth sorting enabled --
fig = plt.figure(figsize=(8, 6), dpi=300)
ax = fig.add_subplot(111, projection='3d')

# equal-aspect in x/y, adjust z-scale
dp = power_all.ptp(); dz = vmax - vmin
ax.set_box_aspect((dp, dp, dz))
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1,1,0.9,1]))

# (a) original UPC surfaces in light grey
surf_kwargs = dict(shade=False, zsort='average', rcount=100, ccount=100,
                   antialiased=False)
ax.plot_surface(*np.meshgrid(power_pump, head_all), flow_pump,
                color='lightgrey', alpha=0.5, **surf_kwargs)
ax.plot_surface(*np.meshgrid(power_turbine, head_all), flow_turbine,
                color='lightgrey', alpha=0.5, **surf_kwargs)

# (b) polynomial-fit surfaces with same colormap
poly_kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax,
                   shade=False, zsort='average', rcount=100, ccount=100)
ax.plot_surface(P_p_poly, H_p_poly, qt_pump, alpha=0.9, **poly_kwargs)
ax.plot_surface(P_t_poly, H_t_poly, qt_turb, alpha=0.9, **poly_kwargs)

# idle line
idle_col = plt.cm.get_cmap(cmap)((0 - vmin) / (vmax - vmin))
ax.plot(np.zeros_like(head_all), head_all, np.zeros_like(head_all),
        color=idle_col, alpha=0.5, lw=2)

# labels & camera
ax.set_xlabel('Power (MW)'); ax.set_ylabel('Head (m)')
ax.set_zlabel('Flow Rate (m³/s)')
ax.view_init(elev=30, azim=225)

# colorbar
mappable = plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=cmap)
cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label('Flow Rate (m³/s)')

plt.tight_layout()
plt.savefig('polynomial_vs_linear_3d_depthsorted.svg', bbox_inches='tight')
plt.show()

