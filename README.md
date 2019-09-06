# NMA
### Introduction
This is the source code for our paper 'Link Prediction with Multi-Stage Structural Attentions in Social Multiplex Networks'
### Installation
After creating a virtual environment of python 3.6, run pip install -r requirements.txt to install all dependencies
### How to use
The code is currently only tested on GPU.

* **Demo**  

   you can run '**python train.py**' to get the result of our demo. The Result store path is */data/CKM-Physicians-Innovation_Multiplex_Social/results/NMA-I_results.txt*.**Note that this is the result using the NMA-I method.**   
    if you want to test the NMA-N method, please run '**python train.py --use_embedding True**'. The Result store path is */data/CKM-Physicians-Innovation_Multiplex_Social/results/NMA-N_results.txt*.

