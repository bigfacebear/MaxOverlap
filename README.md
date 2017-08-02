# MaxOverlap

## Introduction

​	To generate a dataset, in which each entry contains two shapes(lock, key), and the max overlap area of the two shapes.

![](./misc/L.png)

![](./misc/K.png)

![](./misc/overlap.png)

​	The dataset comes into being as follows:

1. Select a set of primitive shapes from exiting shape dataset.
2. Preprocess these primitive shapes into a regular format.
3. Calculate and save the max overlap area between each pair of primitive shapes.
4. Generate the dataset from primitive shapes and max overlap area by randomly transforming primitive shapes.

## How to use

```bash
pip install opencv-python, numpy, pickle, scoop
```

```bash
git clone https://github.com/bigfacebear/MaxOverlap.git
cd MaxOverlap

# set generation parameters in gen_dataset_flags.py

python -m scoop gen_dataset.py
```

Then you get a folder containing pairs of shapes and a file `OVERLAP_AREAS`. `OVERLAP_AREAS` contains the max overlap area between each pair, and you can use it as following.

```python
import pickle

with open('OVERLAP_AREAS') as fp:
    overlap_area_list = pickle.load(fp)
```

## Primitive Shapes

​	This dataset contains 775 primitive shapes gathered from several different shape dataset:

1. [MPEG 7 Shape Matching](http://www.dabi.temple.edu/~shape/MPEG7/dataset.html)
2. [Animal Dataset](https://sites.google.com/site/xiangbai/animaldataset)
3. [Kimia](http://vision.lems.brown.edu/content/available-software-and-databases)
4. [Myth, Tools](http://tosca.cs.technion.ac.il/book/resources_data.html)


​	**You can use `filled_primitives` or `hollow_primitives` directly**. 

## Calculate Max Overlap Area

### Evolutionary Algorithm

​	This is the most time-consuming part in the whole process. I use the evolutionary algorithm to gain precise results efficiently. The library I used are [`DEAP`](https://github.com/DEAP/deap) and [`SCOOP`](https://github.com/soravux/scoop/).

​	You can use the Python wrapper `gen_max_areas.py` to generate the max overlap areas file, `filled_max_areas` and `hollow_max_areas`. In order to use `SCOOP` multi-process library, make sure you run the `gen_max_areas.py` in the following way. This is significant for efficiency.

```bash
python -m scoop gen_max_areas.py
```

​	When you need to use the generated `*_max_areas` file, you can use the following script in your own code.

```python
import pickle
import numpy as np
with open('filled_max_areas') as fp:
    areas_mat = np.array(pickle.load(fp), dtype=np.int)
```

​	`areas_mat` is a symmetric matrix containing the max overlap between primitive shape pairs. Here is an example:

```
6450    2240    2264
2240    3120    1650
2264    1650    2893

# The max overlap area between 0.png and 1.png is 2240
# The max overlap area between 0.png and 2.png is 2264
# The max overlap area between 1.png and 2.png is 1650
```

## Generate Dataset

```bash
python -m scoop gen_dataset.py
```

