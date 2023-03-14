# view-point-pushing
This repository contains the accompanying code for the paper "Viewpoint Push Planning for Mapping of Unknown Confined Spaces" by N. Dengler, S. Pan, V. Kalagaturu, Rohit Menon, M. Dawood, and M. Bennewitz submitted for IROS, 2023. you can find the paper at https://arxiv.org/pdf/2303.03126.pdf
The repository is work in progress and will be extended over time.

# installation
To install and use the gym environment for this project we suggest using Anaconda3 or other virtual environments.

use the following command to install the pacage via pip:

``` pip install -e .```

All important packages should be installed automatically.

# Demo
## View point planner
to test the view point planner, execute from the root directory:

```python shelf_gym/evaluation/test_vpp.py```

## View point push planner

for the whole view point push planner pipeline use:

```python shelf_gym/evaluation/test_push_planner.py```
