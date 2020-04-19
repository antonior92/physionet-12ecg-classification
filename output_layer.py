import torch

output = torch.rand((3, 8))  # AF, I-AVB, LBBB, RBBB, PAC, PVC, STD, STE
target = torch.zeros((3, 6))
target[0, 0] = 1
target[1, 0] = 1
target[2, 0] = 0
target[1, 1] = 1
target[2, 1] = 1

bs = output.size(0)

class OutputLayer(object):
    def __init__(self, bs, softmax_mask, device, dtype=torch.float32):
        # Save zero tensor to be used as 'normal' case in the softmax
        self.normal = torch.zeros((bs, 1), device=device, dtype=dtype)
        self.softmax_mask = softmax_mask
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='sum')
        self.sigmoid =torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)


    def _get_output_components(self, output):
        bs = output.size(0)
        softmax_outputs = []
        n_softmax_outputs = 0
        for mask in self.softmax_mask:
            softmax_outputs.append(torch.cat((self.normal[:bs], output[:, mask]), dim=1))
            n_softmax_outputs += len(mask)
        sigmoid_outputs = output[:, n_softmax_outputs:]
        return softmax_outputs, sigmoid_outputs

    def _get_target_components(self, target):
        softmax_targets = []
        i = 0
        for _ in self.softmax_mask:
            softmax_targets.append(target[:, i])
            i += 1
        sigmoid_targets = target[:, i:]
        return softmax_targets, sigmoid_targets

    def loss(self, output, target):
        softmax_outputs, sigmoid_outputs = self._get_output_components(output)
        softmax_targets, sigmoid_targets = self._get_target_components(target)


        loss = 0
        for i, _ in enumerate(self.softmax_mask):
            loss += self.ce_loss(softmax_outputs[i], softmax_targets[i].to(dtype=torch.long))
        loss += self.bce_loss(sigmoid_outputs, sigmoid_targets.to(dtype=torch.float32))

        return loss

    def get_output(self, output):
        softmax_outputs, sigmoid_outputs = self._get_output_components(output)

        outputs = []
        for i, mask in enumerate(self.softmax_mask):
            outputs.append(self.softmax(softmax_outputs[i])[..., -len(mask):])
        outputs.append(self.sigmoid(sigmoid_outputs))

        return torch.cat(outputs, dim=-1)
