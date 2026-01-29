import matplotlib.pyplot as plt
from pathlib import Path
'''
Centralise management of saving figures
'''

# save figures to disk
def save_figure(figure, name):
    path = Path(f"figures/{name}.png")
    path.parent.mkdir(parents=True, exist_ok=True) # creates missing intermediate files
    figure.savefig(path)
    plt.close(figure)
    return 

