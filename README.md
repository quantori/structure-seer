[![DOI:10.1039/D3DD00178D](http://img.shields.io/badge/DOI-10.1039/D3DD00178D-ebe534.svg)](https://doi.org/10.1039/D3DD00178D)

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
# Structure Seer 

The implementation training and evaluation of a Structure Seer model designed for
reconstruction of adjacency of a molecular graph from the labelling of its nodes.
The detailed characterisation and disclosure of the model architecture is provided in:
[Structure Seer - a machine learning model for chemical structure elucidation
from a node labelling of a molecular graph, Digital discovery, 2023](https://doi.org/10.1039/D3DD00178D)

## Datasets

The repository does not contain initial datasets used for training. 
- Small example datasets for detailed model evaluation are provided in ```./example_dataset```
- Model weights trained on QM9 and PubChem Datasets are stored in ```./weights```

## Abstract

The repository contains the implementation for a novel graph convolution based machine-learning model which
is designed to provide a quantitative probabilistic prediction on the connectivity of the atoms based on the
information on the elemental composition of the molecule along with a list of atom-attributed isotropic shielding
constants. The suggested approach holds significant potential for scalability, as it can harness vast amounts
of information on known chemical structures for the model's learning process. The model architecture allows for 
direct structure reconstruction through prediction of molecular graph adjacency based solely on the
labelling of its nodes, which potentially allows dealing with molecules of any size and composition
(given an appropriate training dataset is available) without significant increase in computational resources required. 					
				
## Key approaches

### Unification of adjacency matrix representation

The primary challenge in generating the adjacency matrix is that it is not an invariant for a given graph.
For a given graph with G nodes, there are G! adjacency matrices that can describe its connectivity.
To tackle this issue, the adjacency matrix representation needs to be unified. Typically, in the machine- readable
representation of a molecule, its atoms are stored in the first-depth-tree traversal order. 
While this order contains information about the stored structure, it cannot be easily reconstructed when only
the elemental composition of the molecule and the isotropic shielding constant for each atom are known. 
Since the shielding constant provides a unique characterization of an atom's chemical environment, it can be
employed to standardize the representation of the adjacency matrix in conjunction with element information.

### Generic adjacency matrix

The architecture of the Structure Seer model bears similarities to other GCN-based models used for diverse tasks
involving molecular graphs. However, its distinctive design is centred around encoding the molecule
solely based on node labelling, which allows for the generation of the complete adjacency matrix.
This feature makes the considered architecture applicable to a broad range of atom adjacency reconstruction tasks.

## Training

Refer to the training procedure in the Jupyter notebook ```./training.ipynb``` . 
Customize the procedure by adjusting the global variables in the second code cell.
The main training function source code is in ```./training/train_model.py```.

In order to train the model using Google Colab - extract the repository to the GDrive into ```./MyDrive```.

## Evaluation

For model evaluation, utilize ```./model_evaluation.ipynb``` with the pretrained model weights.
Small example datasets for detailed model evaluation are provided in ```./example_dataset```.

## Code examples

Explore model usage and functionality in ```./structure_seer_code_examples.ipynb```,
which includes illustrative examples.



