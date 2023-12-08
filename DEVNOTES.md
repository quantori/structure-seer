## Installation
``` 
pip install -r requirements
 ```
The SCF calculation were performed using [ORCA](https://orcaforum.kofo.mpg.de/app.php/portal).
To use calculations, install ORCA and specify its global path
in a corresponding environment variable in ```./utils/config.py```.

## Repository structure

- Source code for models is located in ``` ./models ```.
- Model weights trained on QM9 and PubChem Datasets are stored in ```./weights```.
- Utility classes and functions are provided in ```./utils```.
- Scripts for data preparation and SCF calculations can be found in ```./data_preparation```.
