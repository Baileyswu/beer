import sys
sys.path.append('/home/danliwoo/gplab/beer')
from beer import __init__
import beer

def construct_graph():
    graph = beer.graph.Graph()
    # Initial and final state are non-emitting.
    s0 = graph.add_state()
    s4 = graph.add_state()
    graph.start_state = s0
    graph.end_state = s4

    s1 = graph.add_state(pdf_id=0)
    s2 = graph.add_state(pdf_id=1)
    s3 = graph.add_state(pdf_id=2)
    graph.add_arc(s0, s1) # default weight=1
    graph.add_arc(s1, s1)
    graph.add_arc(s1, s2)
    graph.add_arc(s2, s2)
    graph.add_arc(s2, s3)
    graph.add_arc(s3, s3)
    graph.add_arc(s3, s1)
    graph.add_arc(s1, s4)
    graph.add_arc(s2, s4)
    graph.add_arc(s3, s4)
    graph.normalize()
    return graph
