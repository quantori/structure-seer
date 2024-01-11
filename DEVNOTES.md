## Installation
``` 
pip install -r requirements.txt
 ```
The SCF calculation were performed using [ORCA](https://orcaforum.kofo.mpg.de/app.php/portal).
To use quantum-chemistry calculations, install ORCA and specify its global path
in a corresponding environment variable in ```./data_preparation/parallel_dft_calculation.py```.

## Repository structure

- Source code for models is located in ``` ./models ```.
- Model weights trained on QM9 and PubChem Datasets are stored in ```./weights```.
- Utility classes and functions are provided in ```./utils```.
- Scripts for data preparation and SCF calculations can be found in ```./data_preparation```.
- Parallel jobs for ORCA calculations can be run from ```./data_preparation/parallel_dft_calculation.py```
