#%%
import matplotlib.pyplot as plt
import matplotlib.colorbar as colorbar
import matplotlib.colors as colors

# Set up the figure for a clean, isolated colorbar
fig, ax = plt.subplots(figsize=(4, 1.2), dpi=300) # 300 DPI for high resolution

# Create a normalized scale from 0.0 to 1.0
norm = colors.Normalize(vmin=0.0, vmax=1.0)

# Use 'RdYlBu_r' (reversed) to put Blue/High on the left and Red/Low on the right
cmap = plt.get_cmap('RdYlGn_r')

# Draw the colorbar horizontally
cb = colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')

# Configure the title and position it above the colorbar
ax.set_title('Reliability Score', fontsize=16, pad=10, fontweight='medium')

# Define specific ticks and custom labels underneath
ax.set_xticks([0.0, 0.5, 1.0])
ax.set_xticklabels(['1.0\nHigh', '0.5', '0.0\nLow'], fontsize=10)

# Clean up layout and save as a high-res PNG or PDF
plt.tight_layout()
plt.savefig('reliability_score_colorbar.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()