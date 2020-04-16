import unittest
from codedl.models.resnet import ResBlock1d, ResNet1d
import torch


class TestResBlock1d(unittest.TestCase):
    def test_for_even_kernel_size(self):
        bs, f_in, l_in = 10, 32, 64
        x = torch.ones(bs, f_in, l_in)
        for f_out in [12, 16, 32, 64, 100]:
            for downsample in [1, 2, 3, 4]:
                for kernel_size in [3, 5, 17]:
                    blk = ResBlock1d(n_filters_in=f_in, n_filters_out=f_out, downsample=downsample, kernel_size=kernel_size, dropout_rate=0)
                    y, z = blk(x, x)
                    self.assertEqual(y.shape, (bs, f_out, l_in // downsample))
                    self.assertEqual(z.shape, (bs, f_out, l_in // downsample))

    def test_for_odd_kernel_size(self):
        bs, f_in, l_in = 10, 32, 64
        x = torch.ones(bs, f_in, l_in)
        for f_out in [12, 16, 32, 64, 100]:
            for downsample in [1, 2, 3, 4]:
                for kernel_size in [2, 4, 8]:
                    self.assertRaises(ValueError, ResBlock1d, n_filters_in=f_in, n_filters_out=f_out, downsample=downsample, kernel_size=kernel_size, dropout_rate=0)


class TestResNet1d(unittest.TestCase):
    def test_resnet_1d(self):
        bs, c, l_in = 10, 12, 64
        x = torch.ones(bs, c, l_in)
        for kernel_size in [3, 5, 17]:
            for n_classes in 6, 9, 11:
                for block_dim in [
                    [[16, 32], [32, 16], [64, 8]],
                    [[16, 64], [32, 16], [64, 8]],
                    [[16, 64], [32, 16], [64, 8], [100, 4]],
                ]:
                    # Resnet
                    net = ResNet1d([c, l_in], block_dim, n_classes, kernel_size)
                    # Forward
                    y = net(x)
                    # Check shape
                    self.assertEqual(y.shape, (bs, n_classes))

    def test_resnet_fractionary_downsample(self):
        bs, c, l_in = 10, 12, 64
        x = torch.ones(bs, c, l_in)
        for kernel_size in [3, 5, 17]:
            for n_classes in 6, 9, 11:
                for block_dim in [
                    [[16, 32], [12, 12], [64, 8]],
                    [[16, 64], [32, 13], [64, 8]],
                    [[16, 64], [32, 16], [64, 8], [100, 5]],
                ]:
                    self.assertRaises(ValueError, ResNet1d, [c, l_in], block_dim, n_classes, kernel_size)

    def test_resnet_upsample(self):
        bs, c, l_in = 10, 12, 64
        x = torch.ones(bs, c, l_in)
        for kernel_size in [3, 5, 17]:
            for n_classes in 6, 9, 11:
                for block_dim in [
                    [[16, 32], [12, 16], [64, 32]],
                    [[16, 32], [12, 64], [64, 32]],
                ]:
                    self.assertRaises(ValueError, ResNet1d, [c, l_in], block_dim, n_classes, kernel_size)


if __name__ == '__main__':
    unittest.main()
