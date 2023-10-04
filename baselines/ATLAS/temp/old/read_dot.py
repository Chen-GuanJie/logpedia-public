import graphviz as gv

# Load the .dot file
with open("/Users/zhanghangsheng/others_code/Provenance_graph/ATLAS-main/paper_experiments/S1/output/attack_graph_testing_preprocessed_logs_S1-CVE-2015-5122_windows.dot") as f:
    dot_graph = f.read()

# Create a graphviz graph object
g = gv.Source(dot_graph)

# Render the graph as a PNG image
g.render("graph")
