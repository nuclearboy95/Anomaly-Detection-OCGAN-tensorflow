# Tensorflow Implementation of OCGAN
This repository provides a [Tensorflow](https://www.tensorflow.org/) implementation of the *OCGAN* presented in
CVPR 2019 paper "[OCGAN: One-class Novelty Detection Using GANs with Constrained Latent Representations](http://openaccess.thecvf.com/content_CVPR_2019/papers/Perera_OCGAN_One-Class_Novelty_Detection_Using_GANs_With_Constrained_Latent_Representations_CVPR_2019_paper.pdf)".

The author's implementation of *OCGAN* in MXNet is at [here](https://github.com/PramuPerera/OCGAN).


## Installation
This code is written in `Python 3.5` and tested with `Tensorflow 1.13`.

Install using pip or clone this repository.

1. Installation using pip:
```bash
pip install ocgan
```

and

```python
from ocgan import OCGAN
```

2. Clone this repository:

```bash
git clone https://github.com/nuclearboy95/Anomaly-Detection-OCGAN-tensorflow.git
```

## Result (AUROC)
| **MNIST DIGIT** | **OCGAN w/ <br> Informative-negative <br> mining** | **OCGAN w/o <br> Informative-negative <br> mining** |
|:---------------:|:--------------------------------------------------:|:---------------------------------------------------:|
|        0        |                     **0.9952**                     |                        0.9935                       |
|        1        |                       0.9976                       |                      **0.9985**                     |
|        2        |                     **0.9268**                     |                        0.9133                       |
|        3        |                     **0.9410**                     |                        0.9208                       |
|        4        |                     **0.9636**                     |                        0.9600                       |
|        5        |                     **0.9613**                     |                        0.9145                       |
|        6        |                     **0.9910**                     |                        0.9835                       |
|        7        |                     **0.9658**                     |                        0.9526                       |
|        8        |                     **0.9009**                     |                        0.8758                       |
|        9        |                       0.9584                       |                      **0.9701**                     |

NOTE: *The AUROC values are measured only once for each digit.*