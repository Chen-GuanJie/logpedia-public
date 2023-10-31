#

## baselines

put the log files in data folder and run the code  
`baseline.py` helps to build the graph and every baseline code is based on it  

### ATLAS

---
run `python atlas.py`, the results are in `baselines/result`

### Prodetector

---
`get_k_path.py` get the top K rarest paths

### ShadeWathcer

---
In `shadewatcher.py` .  
function `prepare_shadewatcher` get the 4 encoding files for each scenario in the folder `baselines\ShadeWatcher\data\encoding`  
To get the nodes' embedding run `python driver.py --dataset i --save_embedding` where is the number of the scenario  
After getting the embedding,run the function `test_shadewatcher` to calcuate the results

### ThreaTrace

---

- Python 3.10.12
- pytorch 2.2.0
- dgl 1.1.2

run `threatrace.py`

### result

results are in the baselines/result folder

