from unittest import TestCase
from theoretical_est import *

class Test(TestCase) :
    def test_wave_front(self):
        graph = {0: [1, 2], 1: [2, 3], 2: [3, 4], 3: [4]}
        visited = set([0,1])
        prev_wavefront = set([0,1])
        curr_wavefront = waveFront(graph, visited, prev_wavefront)
        if curr_wavefront == {2, 3}:
            print( "wave_front: CORRECT")

    # def test_hidden(self):
    #     print(fakeHiddenStateLength(100, 5))


    def test_nonFirstLayer(self) :
        edge_index = np.array([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)])
        sample_edges = np.array([[0,1], [2,3]])
        hidden_state_length = fakeHiddenStateLength(100, 5)
        batch, whole = nonFirstLayer(edge_index, sample_edges,hidden_state_length, 5, 2, 5)
        print(batch, whole)
# test = Test()
# test.test_wave_front()
# test.test_hidden()
