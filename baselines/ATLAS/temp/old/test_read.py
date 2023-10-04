import networkx as nx
from networkx.drawing.nx_pydot import read_dot
import matplotlib.pyplot as plt
from networkx_viewer import Viewer

G = read_dot("ATLAS/paper_experiments/M1/h1/output/attack_graph_testing_preprocessed_logs_M1-CVE-2015-5122_windows_h1.dot")
pos = nx.spring_layout(G)
nx.draw(G, with_labels=False)

# labels = {node: node for node in G.nodes() if node in ["c:/users/aalsahee/payload.exe"]}
# nx.draw_networkx_labels(G, pos, labels, font_size=16)

plt.show()



