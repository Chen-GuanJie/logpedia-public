import sys

sys.path.append('/data1/hdq/Projects/RTA/exp/Threatrace_ATLAS/')
import read_ALTAS_graph as atlas_utils
import test_my_parse_230320 as my_utils
import numpy as np


folder = 'ATLAS/paper_experiments/S1/output/'

S1_log_file = 'testing_preprocessed_logs_S1-CVE-2015-5122_windows'
S1_mal = ['0xalsaheel.com', '192.168.223.3', 'payload.exe']
G1 = atlas_utils.get_ATLAS_G(folder+S1_log_file)

X, y, elabel2no, nlabel2no, nid2no,\
no2nid, mal_nids, mal_nidnos, no2nlabel, no2elabel = my_utils.parse_graph(G1, S1_mal)


sw_data_pth = '/Users/zhanghangsheng/Documents/my_code/attack-analysis/ShadeWatcher/data/encoding/atlas_s1/'
kg_file = sw_data_pth + '/train2id.txt'
rel_file = sw_data_pth + '/relation2id.txt'
entity_file = sw_data_pth + '/entity2id.txt'
inter_file = sw_data_pth + '/inter2id_0.txt'

with open(rel_file, 'w') as f:
    f.write(str(len(no2elabel)+1)+"\n")
    for i in range(len(no2elabel)):
        f.write(no2elabel[i]+' '+str(i)+'\n')
    f.write('SWBUG '+ str(len(no2elabel))+'\n')
        
with open(entity_file, 'w') as f:
    f.write(str(len(G1.nodes()))+"\n")
    for i in range(len(G1.nodes())):
        f.write(str(i)+' '+str(i)+'\n')

with open(kg_file, 'w') as f:
    f.write(str(len(G1.edges()))+"\n")
    for u, v, attr in G1.edges(data=True):
        line = [nid2no[u], nid2no[v], elabel2no[attr['type']]]
        f.write(str(line[0])+' '+str(line[1])+' '+str(line[2])+'\n')

with open(inter_file, 'w') as f:
    for u in G1.nodes():
        if len(G1.out_edges(u)) == 0:
            continue
        f.write(str(nid2no[u])+' ')
        for _, v in G1.out_edges(u):
            f.write(str(nid2no[v])+' ')
        f.write('\n')
        
print(mal_nidnos)

G1.out_edges('NOPROCESSNAME_1848')