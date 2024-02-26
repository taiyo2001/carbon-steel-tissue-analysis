from torch import nn, Tensor
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """
    Combined Dice Loss and Binary Cross Entropy (BCE) Loss for semantic segmentation.

    Parameters:
    - weight (torch.Tensor, optional): A manual rescaling weight given to each class. If given, it has to be a Tensor of size C.
    - size_average (bool, optional): Deprecated (see :attr:`reduction`). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field size_average is set to False, the losses are instead summed for each minibatch. Ignored when reduce is False. Default: True.
    """

    def __init__(self, weight: Tensor = None, size_average: bool = True) -> None:
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs: Tensor, targets: Tensor, smooth: float = 1) -> Tensor:
        """
        Forward pass of the IoU calculation.

        Parameters:
        - inputs (Tensor): Predicted output tensor from the model.
        - targets (Tensor): Ground truth tensor.
        - smooth (float, optional): Smoothing factor to avoid division by zero. Default: 1.

        Returns:
            Tensor: Computed IoU.
        """
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            inputs.sum() + targets.sum() + smooth
        )
        BCE = F.binary_cross_entropy(inputs, targets, reduction="mean")
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class DiceLoss(nn.Module):
    """
    Dice Loss module for semantic segmentation.

    Parameters:
    - weight (Tensor, optional): A manual rescaling weight given to each class. If given, it has to be a Tensor of size C.
    - size_average (bool, optional): Deprecated (see :attr:`reduction`). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field size_average is set to False, the losses are instead summed for each minibatch. Ignored when reduce is False. Default: True.
    """

    def __init__(self, weight: Tensor = None, size_average: bool = True) -> None:
        super(DiceLoss, self).__init__()

    # def forward(self, inputs, targets, smooth=1):

    def forward(self, inputs: Tensor, targets: Tensor, smooth: float = 1) -> Tensor:
        """
        Forward pass of the Dice Loss calculation.

        Parameters:
        - inputs (Tensor): Predicted output tensor from the model.
        - targets (Tensor): Ground truth tensor.
        - smooth (float, optional): Smoothing factor to avoid division by zero. Default: 1.

        Returns:
            Tensor: Computed Dice Loss.
        """
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class IoU(nn.Module):
    """
    Intersection over Union (IoU) module for semantic segmentation.

    Parameters:
    - weight (torch.Tensor, optional): A manual rescaling weight given to each class. If given, it has to be a Tensor of size C.
    - size_average (bool, optional): Deprecated (see :attr:`reduction`). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field size_average is set to False, the losses are instead summed for each minibatch. Ignored when reduce is False. Default: True.
    """

    def __init__(self, weight: Tensor = None, size_average: bool = True) -> None:
        super(IoU, self).__init__()

    def forward(self, inputs: Tensor, targets: Tensor, smooth: float = 1) -> Tensor:
        """
        Forward pass of the IoU calculation.

        Parameters:
        - inputs (torch.Tensor): Predicted output tensor from the model.
        - targets (torch.Tensor): Ground truth tensor.
        - smooth (float, optional): Smoothing factor to avoid division by zero. Default: 1.

        Returns:
            torch.Tensor: Computed IoU.
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return IoU
