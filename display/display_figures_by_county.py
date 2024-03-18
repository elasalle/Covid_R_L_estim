import networkx as nx
import matplotlib.pyplot as plt

from include.load_data.load_graph_examples import example_choice, set_Graph_fromMatrix


def display_connect_structure(example):
    """
    Displays the connectivity structure named 'example'.
    :param example: str between 'Line', 'Hub', ... described in include/load_data/load_graph_examples.py
    :return:
    """
    depCont, pos, labels, colorMap = example_choice(example)
    G_graph = set_Graph_fromMatrix(depCont)

    fig5, axG = plt.subplots(figsize=(3.75, 2.45))
    fig5.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    nx.draw(G_graph, ax=axG, labels=labels, node_color=colorMap, pos=pos, node_size=500, font_size=20)
    axG.axis("off")
    fig5.show()
