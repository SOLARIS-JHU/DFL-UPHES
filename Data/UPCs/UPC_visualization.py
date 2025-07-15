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

# High-res surface sampling using rcount/ccount
ax.plot_surface(P_p, H_p, flow_pump,    rcount=100, ccount=100,
                cmap=cmap, vmin=vmin, vmax=vmax, antialiased=False, alpha=0.9)
ax.plot_surface(P_t, H_t, flow_turbine, rcount=100, ccount=100,
                cmap=cmap, vmin=vmin, vmax=vmax, antialiased=False, alpha=0.9)

# Idle line
idle_color = plt.cm.get_cmap(cmap)((0 - vmin)/(vmax - vmin))
ax.plot(np.zeros_like(head_all), head_all, np.zeros_like(head_all),
        color=idle_color, alpha=0.5, lw=2)

# Contour projections at z=vmin with finer levels
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

# Labels with leader lines on top layer
offset=0.15
# Pump label
lx, ly, lz = x_mp-dp*offset, y_mp, z_mp+dq*offset
ax.plot([x_mp,lx],[y_mp,ly],[z_mp,lz],'k',zorder=10)
ax.text(lx,ly,lz,'Pump Mode',ha='center',zorder=10)
# Turbine label
lx, ly, lz = x_mt+dp*offset, y_mt, z_mt+dq*offset
ax.plot([x_mt,lx],[y_mt,ly],[z_mt,lz],'k',zorder=10)
ax.text(lx,ly,lz,'Turbine Mode',ha='center',zorder=10)
# Idle label
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

# Save the 3D plot as SVG
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
    
    # Save each 2D contour plot as SVG
    filename = f"francis_turbine_{title.lower().replace(' ', '_')}_contours.svg"
    plt.savefig(filename, format='svg', bbox_inches='tight')
    plt.show()

