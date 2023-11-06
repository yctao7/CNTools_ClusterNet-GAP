# ClusterNet and GAP

This code adopts the framework of the code of ClusterNet method in the NeurIPS 2019 [paper](https://arxiv.org/abs/1905.13732) "End to End Learning and Optimization on Graphs" and adds the option to use GAP method [paper](https://arxiv.org/abs/1903.00614) "GAP: Generalizable Approximate Graph Partitioning Framework" for cellular neighborhood identification.

```
@inproceedings{wilder2019end,
  title={End to End Learning and Optimization on Graphs},
  author={Wilder, Bryan and Ewing, Eric and Dilkina, Bistra and Tambe, Milind},
  booktitle={Advances in Neural and Information Processing Systems},
  year={2019}
}

@article{nazi2019gap,
  title={Gap: Generalizable approximate graph partitioning framework},
  author={Nazi, Azade and Hang, Will and Goldie, Anna and Ravi, Sujith and Mirhoseini, Azalia},
  journal={arXiv preprint arXiv:1903.00614},
  year={2019}
}
```

# Files

* experiments_singlegraph.py runs the experiments which do link prediction on a given graph. See below for parameters.
* experiments_inductive.py runs the experiments which evaluate training and generalization on a distribution of graphs.
* models.py contains the definitions for both the ClusterNet model (GCNClusterNet) as well as various models used for the baselines.
* modularity.py contains helper functions and baseline optimization algorithms for the community detection task.
* kcenter.py contains helper functions and baseline optimization algorithms for the facility location task.
* loss_functions.py contains definitions of the loss function used to train ClusterNet and GCN-e2e for both tasks.
* utils.py contains helper functions to load/manipulate the datasets.

# Datasets

The datasets used are placed in the mydata folder, which are knn-graphs of different k and cell type information as node features (feat). The hlt data can be downloaded from https://drive.google.com/drive/folders/1OImf7SiSwqVJz7HYCGk71Vz7P5lZozFu?usp=share_link.

# Examples of running the experiments

See run.sh for all commands to reproduce the results of ClusterNet and GAP.

# Dependencies

The Dockerfile in the main directory builds the environment that was used to run the original experiments, with the exception of [pygcn](https://github.com/tkipf/pygcn/tree/master/pygcn), which needs to be downloaded and installed separately. For reference, the individual dependencies are:

* PyTorch (tested on version 1.2)
* networkx (tested on version 2.3)
* igraph (tested on version 0.7.1). This is optional; only used to accelerate pre-processing operations.
* [pygcn](https://github.com/tkipf/pygcn/tree/master/pygcn)
* sklearn (tested on version 0.21.3)
* numpy (tested on version 1.16.5)
