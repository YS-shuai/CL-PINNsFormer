# CL-PINNsFormer

CL-PINNsFormer: A lightweight pure-decoder framework that breaks the low-pass filtering bottleneck in PINNs, achieving O(N) complexity and superior high-frequency PDE solutions.

## Installation

Clone this repo and install dependencies:

```bash
git clone https://github.com/YS-shuai/CL-PINNsFormer.git
cd CL-PINNsFormer
pip install -r requirement.txt
```

## Get Started

```shell
python reaction_CL_PINNsFormer.py --model CL_PINNsFormer --device 'cuda:0'
python wave_CL_PINNsFormer.py --model CL_PINNsFormer --device 'cuda:0' 
python convection_CL_PINNsFormer.py --model CL_PINNsFormer --device 'cuda:0'
```

## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code base:

https://github.com/AdityaLab/pinnsformer

https://github.com/miniHuiHui/PINNMamba

