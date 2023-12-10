from typing import Callable, List
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

def plot_geometry(X: np.ndarray, E: float, B: float):
    
    if len(X.shape) == 1:
        X = X[None,:]
        
    coils = [Rectangle((X[0,0]-X[0,4]/2, -X[0,2]), X[0,4], 2*X[0,2]), 
             Rectangle((X[0,1]-X[0,5]/2, -X[0,3]), X[0,5], 2*X[0,3])]

    pc = PatchCollection(coils, facecolor=['tab:red', 'tab:blue'], alpha=0.5)#, edgecolor='None')

    fig, axs = plt.subplots(1, 4, figsize=(10,5), 
                            gridspec_kw={'width_ratios': [5, 1,1,1]})
    axs[0].add_collection(pc)
    axs[0].plot([0,6], [0,0], 'k-.')
    axs[0].set_xlim([0, X[0,1]+X[0,5]/2*1.5])
    axs[0].set_ylim([-2, 2])
    axs[0].set_xlabel('r in m')
    axs[0].set_ylabel('z in m')

    rect = [Rectangle((-1, 0), 2, np.abs(E))]
    pc = PatchCollection(rect, facecolor='tab:green', alpha=0.7)#, edgecolor='None')
    axs[1].add_collection(pc)
    axs[1].plot([-1,1], [180e6,180e6], 'k:', lw=2.5)
    axs[1].set_xlim([-1.5,1.5])
    axs[1].set_ylim([0,10e8])
    axs[1].set_ylabel('Energy E')
    axs[1].set_title('Obj 1.')
    axs[1].set_xticklabels([])

    rect = [Rectangle((-1, 0), 2, B/(200e-6)**2)]
    pc = PatchCollection(rect, facecolor='tab:green', alpha=0.7)#, edgecolor='None')
    axs[2].add_collection(pc)
    axs[2].plot([-1,1], [1,1], 'k:', lw=2.5)
    axs[2].set_xlim([-1.5,1.5])
    axs[2].set_ylim([0,1e-3])
    axs[2].set_ylabel(r'Stray field $B_{stray}$')
    axs[2].set_title('Obj 2.')
    axs[2].set_xticklabels([])
    
    rect = [Rectangle((-1, 0), 2, B/(200e-6)**2)]
    pc = PatchCollection(rect, facecolor='tab:green', alpha=0.7)#, edgecolor='None')
    axs[3].add_collection(pc)
    axs[3].plot([-1,1], [1,1], 'k:', lw=2.5)
    axs[3].set_xlim([-1.5,1.5])
    axs[3].set_ylim([0,1e-7])
    axs[3].set_ylabel(r'Stray field $B_{stray}$')
    axs[3].set_title('Obj 2. zoomed')
    axs[3].set_xticklabels([])

    plt.tight_layout()
    plt.show()