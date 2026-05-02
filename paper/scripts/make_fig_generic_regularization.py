import matplotlib.pyplot as plt
import numpy as np

cells = [
    'N-body k=4', 'N-body k=8', 'N-body k=12',
    'rMD17 aspirin', 'rMD17 ethanol', 'rMD17 malonaldehyde',
    'rMD17 naphthalene', 'rMD17 toluene'
]
sem_norm_vs_none = [6.014, 4.768, 4.824, 8.367, 5.196, 3.979, 6.785, 2.359]
sem_norm_vs_smpg = [0.535, -0.370, 0.906, 0.147, -2.935, -0.445, 0.191, -0.289]
ci_none = [1.0, 1.0, 1.0, 1.5, 1.2, 1.5, 1.3, 1.2]
ci_smpg = [1.5, 1.3, 1.4, 1.5, 1.2, 1.5, 1.3, 1.2]

x = np.arange(len(cells))
width = 0.35
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.bar(x - width/2, sem_norm_vs_none, width, yerr=ci_none, capsize=4,
       label='vs no-prior baseline', color='#2c7bb6', alpha=0.85)
ax.bar(x + width/2, sem_norm_vs_smpg, width, yerr=ci_smpg, capsize=4,
       label='vs SMPG control', color='#d7191c', alpha=0.85)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.axhline(y=2, color='gray', linestyle='--', linewidth=0.5,
           label='|SEM-norm.|=2 reference')
ax.axhline(y=-2, color='gray', linestyle='--', linewidth=0.5)
ax.set_ylabel('SEM-normalized improvement', fontsize=11)
ax.set_title('Generic regularization across 8 cells: graph beats no-prior but not SMPG',
             fontsize=12, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(cells, rotation=30, ha='right', fontsize=9)
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('paper/figures/fig_generic_regularization_8cells.pdf',
            dpi=300, bbox_inches='tight')
plt.savefig('paper/figures/fig_generic_regularization_8cells.png',
            dpi=150, bbox_inches='tight')
print('Saved fig_generic_regularization_8cells.pdf')
