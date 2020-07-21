import unittest
from outlayers.output_encoding import OutputEncoding


class TestOutputEncoding(unittest.TestCase):
    def test_initialization(self):
        max_targets = [1, 2, 1, 3]
        null_targets = [0, None, 0, 1]

        oe = OutputEncoding(max_targets, null_targets)
        self.assertEqual(oe.dict_from_indice_to_pair[0], (0, 1))
        self.assertEqual(oe.dict_from_indice_to_pair[1], (1, 0))
        self.assertEqual(oe.dict_from_indice_to_pair[2], (1, 1))
        self.assertEqual(oe.dict_from_indice_to_pair[3], (1, 2))
        self.assertEqual(oe.dict_from_indice_to_pair[4], (2, 1))
        self.assertEqual(oe.dict_from_indice_to_pair[5], (3, 0))
        self.assertEqual(oe.dict_from_indice_to_pair[6], (3, 2))
        self.assertEqual(oe.dict_from_indice_to_pair[7], (3, 3))
        self.assertEqual(oe.dict_from_pair_to_indice[(0, 1)], 0)


if __name__ == '__main__':
    unittest.main()
