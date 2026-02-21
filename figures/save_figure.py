import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
'''
Centralise management of saving figures
'''


def save_figure(figure: Figure, name: str) -> None:
    '''
    Save figures to disk
    '''
    path = Path(f"figures/{name}.png")
    path.parent.mkdir(parents=True, exist_ok=True) # creates missing intermediate files
    figure.savefig(path)
    plt.close(figure)
    return 

