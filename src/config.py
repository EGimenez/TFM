import platform
from pathlib import Path

if platform.system() == 'Windows':
    data_path = Path('E:/NOT_BACKUP/TFM/data/celeba_wild')
    models_path = Path('C:/Users/EGimenez/ME/projects/BGSE/TFM/projects/TFM/src/models')
    results_path = Path('E:/NOT_BACKUP/TFM/results/')
else:
    data_path = Path('/home/egimenez/data/celeba_wild')
    models_path = Path('C:/Users/EGimenez/ME/projects/BGSE/TFM/projects/TFM/src/models')
    results_path = Path('/home/egimenez/results')