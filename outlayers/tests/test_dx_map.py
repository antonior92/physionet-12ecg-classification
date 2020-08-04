import unittest
from outlayers.output_encoding import OutputEncoding
from outlayers.dx_map import DxMap
from numpy.testing import assert_array_equal, assert_array_almost_equal


class TestDxMap(unittest.TestCase):
    """No null class map."""
    def setUp(self):
        # Output enconding
        max_targets = [1, 2, 1]
        null_targets = [0, 0, None]
        oe = OutputEncoding(max_targets, null_targets)

        # DxMap
        classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        pairs = [(0, 1), (1, 1), (1, 2), (2, 0), (2, 1), (0, 1), [(0, 1), (1, 1)], [(0, 0), (1, 0)]]
        self.dx = DxMap(classes, pairs, oe)

    def test_target_from_labels(self):

        labels = ['A', 'B', 'C', 'D', 'E',  ['A', 'B'], ['A', 'C'], ['A', 'E'], ['B', 'E'], 'F', ['F', 'B'],
                  'G', ['G', 'E'], 'H', ['H', 'E']]
        targets = [[1, 0, 0],
                   [0, 1, 0],
                   [0, 2, 0],
                   [0, 0, 0],
                   [0, 0, 1],
                   [1, 1, 0],
                   [1, 2, 0],
                   [1, 0, 1],
                   [0, 1, 1],
                   [1, 0, 0],
                   [1, 1, 0],
                   [1, 1, 0],
                   [1, 1, 1],
                   [0, 0, 0],
                   [0, 0, 1]]

        for l, t in zip(labels, targets):
            th = self.dx.target_from_labels(l)
            assert_array_equal(t, th)

    def test_prepare_targets(self):

        targets = [[1, 0, 0],
                   [0, 1, 0],
                   [0, 2, 0],
                   [0, 0, 0],
                   [0, 0, 1],
                   [1, 1, 0],
                   [1, 2, 0],
                   [1, 0, 1],
                   [0, 1, 1]]

        new_target = self.dx.prepare_target(targets, 'A')
        assert_array_equal(new_target, [[1], [0], [0], [0], [0], [1], [1], [1], [0]])

        new_target = self.dx.prepare_target(targets, 'F')
        assert_array_equal(new_target, [[1], [0], [0], [0], [0], [1], [1], [1], [0]])

        new_target = self.dx.prepare_target(targets, ['A', 'B'])
        assert_array_equal(new_target, [[1, 0], [0, 1], [0, 0], [0, 0], [0, 0], [1, 1], [1, 0], [1, 0], [0, 1]])

        new_target = self.dx.prepare_target(targets, ['F', 'B'])
        assert_array_equal(new_target, [[1, 0], [0, 1], [0, 0], [0, 0], [0, 0], [1, 1], [1, 0], [1, 0], [0, 1]])

        new_target = self.dx.prepare_target(targets, ['A', 'D'])
        assert_array_equal(new_target, [[1, 1], [0, 1], [0, 1], [0, 1], [0, 0], [1, 1], [1, 1], [1, 0], [0, 0]])

        new_target = self.dx.prepare_target(targets, ['C', 'E'])
        assert_array_equal(new_target, [[0, 0], [0, 0], [1, 0], [0, 0], [0, 1], [0, 0], [1, 0], [0, 1], [0, 1]])

        new_target = self.dx.prepare_target(targets, ['G'])
        assert_array_equal(new_target, [[0], [0], [0], [0], [0], [1], [0], [0], [0]])

        new_target = self.dx.prepare_target(targets, ['H'])
        assert_array_equal(new_target, [[0], [0], [0], [1], [1], [0], [0], [0], [0]])

    def test_prepare_probabilities(self):
        probs = [[0.1, 0.1, 0.2, 0.5, 0.5],
                 [0.9, 0.7, 0.05, 0.3, 0.7]]

        new_probs = self.dx.prepare_probabilities(probs, ['A', 'B', 'C', 'D'])
        assert_array_equal(new_probs, [[0.1, 0.1, 0.2, 0.5], [0.9, 0.7, 0.05, 0.3]])

        new_probs = self.dx.prepare_probabilities(probs, ['F', 'B', 'C', 'D'])
        assert_array_equal(new_probs, [[0.1, 0.1, 0.2, 0.5], [0.9, 0.7, 0.05, 0.3]])

        new_probs = self.dx.prepare_probabilities(probs, ['A', 'F', 'B', 'C', 'D'])
        assert_array_equal(new_probs, [[0.1, 0.1, 0.1, 0.2, 0.5], [0.9, 0.9, 0.7, 0.05, 0.3]])

        new_probs = self.dx.prepare_probabilities(probs, ['G'])
        assert_array_equal(new_probs, [[0.1*0.1], [0.9*0.7]])

        new_probs = self.dx.prepare_probabilities(probs, ['H'])
        assert_array_almost_equal(new_probs, [[0.9*0.7], [0.1*0.25]])

    def test_repr(self):
        dx = DxMap.from_str(str(self.dx))
        self.assertEqual(set(dx.classes), set(self.dx.classes))

if __name__ == '__main__':
    unittest.main()
