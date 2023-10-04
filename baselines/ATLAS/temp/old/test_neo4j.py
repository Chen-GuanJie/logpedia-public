from py2neo import Graph, Node, Relationship
import networkx as nx
from networkx.drawing.nx_pydot import read_dot

# Read the graph from a .dot file
G = read_dot("/Users/zhanghangsheng/others_code/Provenance_graph/ATLAS-main/paper_experiments/M1/h1/output/attack_graph_testing_preprocessed_logs_M1-CVE-2015-5122_windows_h1.dot")

# Connect to a Neo4j database
graph = Graph(host="localhost", user="neo4j", password="password")

# Clear the existing data in the database
graph.run("MATCH (n) DETACH DELETE n")

# Create nodes and relationships in the Neo4j database
for node in G.nodes():
    neo4j_node = Node("Node", name=node)
    graph.create(neo4j_node)

for u, v, data in G.edges(data=True):
    neo4j_node1 = graph.nodes.match("Node", name=u).first()
    neo4j_node2 = graph.nodes.match("Node", name=v).first()
    relationship = Relationship(neo4j_node1, "LINKS_TO", neo4j_node2)
    graph.create(relationship)

# Visualize the graph in the Neo4j web browser
print("Graph created in the Neo4j database. Open the Neo4j web browser at http://localhost:7474 to visualize the graph.")
