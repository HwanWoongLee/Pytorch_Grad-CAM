import torch
import torch.nn as nn


def GetGradCAMModule(model, target_layer):
    return QGradCAM(model, target_layer)


class QGradCAM(nn.Module):
    def __init__(self, model, target_layer):
        super(QGradCAM, self).__init__()

        self.model = model
        self.target_layer = None

        for name, layer in model.named_modules():
            if name == target_layer:
                self.target_layer = layer

        if not self.target_layer:
            print('target layer is None')

        self.forward_result = None
        self.backward_result = None

        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_backward_hook(self.backward_hook)

    def forward(self, x):
        return self.model.forward(x)

    def forward_hook(self, module, input, output):
        print('forward hook')
        self.forward_result = torch.squeeze(output)

    def backward_hook(self, module, grad_input, grad_output):
        print('backward hook')
        self.backward_result = torch.squeeze(grad_output[0])

