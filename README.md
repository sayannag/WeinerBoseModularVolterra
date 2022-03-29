# WeinerBoseModularVolterra

Python implementation of LYSIS Matlab package (https://bmsr.usc.edu/software/lysis/lysis-7-2-matlab/)

If you use this reporsitory please cite as:

APA format:

```
Nag, S. (2022). WeinerBoseModularVolterra (Version 1.0.0) [Computer software]. https://github.com/sayannag/WeinerBoseModularVolterra
```

Bibtex:

```
@software{Nag_WeinerBoseModularVolterra_2022,
author = {Nag, Sayan},
month = {3},
title = {{WeinerBoseModularVolterra}},
url = {https://github.com/sayannag/WeinerBoseModularVolterra},
version = {1.0.0},
year = {2022}
}
```

Usage (on Colab):

```
!git clone https://github.com/sayannag/WeinerBoseModularVolterra.git

%cd /content/WeinerBoseModularVolterra

import numpy as np

from math import log

import matplotlib.pyplot as plt

from wb_mv_utils import *

WB = WeinerBose(isPlot=True)
MV = ModularVolterra(isPlot=True)

N = 1024

x = np.random.rand(N,1)

DLF = WB.function_generate_laguerre(0.5, 3, 40)
h = -0.5 * DLF[:,0] + 1 * DLF[:,1] - 1.5 * DLF[:,2]

v = np.convolve(x.flatten(), h)
y = v[0:N] + v[0:N]**2

alpha = 0.5

L = 3
Q = 2
Nfig = 1

Cest, Kest, Pred, NMSE = WB.LET_1(x, y, alpha, L, Q, Nfig)

Npdms, PDMs, ANFs, Pred, NMSE = MV.PDM_1(x, y, alpha, L, Nfig, Npdms_input=5)
```
