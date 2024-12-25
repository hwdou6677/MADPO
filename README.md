

This code is the official implementation of [MADPO](https://openreview.net/forum?id=xvYI7TCiU6), "Measuring Mutual Policy Divergence for Multi-Agent Sequential Exploration", NeurIPS 2024.

### Installation

Please follow [HARL](https://github.com/PKU-MARL/HARL) to install algorithms and enviroment dependencies. 


### Training Example

```shell
for seed in $(seq 1 3)
do
	python train.py --algo madpo --env mamujoco --task Walker2d-v2 --agent 6x1 --seed $seed
done
```
