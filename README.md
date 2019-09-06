# Link Prediction with Multi-Stage Structural Attentions in Social Multiplex Networks
### Introduction
This is the source code for our paper '**Link Prediction with Multi-Stage Structural Attentions in Social Multiplex Networks**'.
### Installation  
   * Requirements 
     * pytorch>=1.0
     * python==3.6
     After creating a virtual environment of python 3.6, run `pip install -r requirements.txt` to install all dependencies.
### How to use
The code is currently only tested on GPU, but you can run it on CPU. Of course, you have to put up with its inefficiencies.

* **Demo**  

     You can run `python train.py` to get the result of our demo. The Result store path is */data/arXiv/results/NMA-I_results.txt*.  
     
     **Note that this is the result using the NMA-I method.** 
     
     If you want to test the NMA-N method, please run `python train.py --use_embedding True`. The Result store path is */data/arXiv/results/NMA-N_results.txt*.
     
* **Source data sets**  

     If you want some other datasets, please click "https://comunelab.fbk.eu/data.php" to download.

