import torch
from torch import nn, Tensor


# UNet
class UNet(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        """
        UNet model for semantic segmentation.

        Parameters:
        - input_channels (int): Number of input channels.
        - output_channels (int): Number of output channels.

        Returns:
            output: Segmentation output tensor.
        """
        super().__init__()
        # Convolutional layers for the encoder (FCN part)
        self.conv1 = conv_bn_relu(input_channels, 64)
        self.conv2 = conv_bn_relu(64, 128)
        self.conv3 = conv_bn_relu(128, 256)
        self.conv4 = conv_bn_relu(256, 512)
        self.conv5 = conv_bn_relu(512, 1024)
        self.down_pooling = nn.MaxPool2d(2)

        # Convolutional layers for the decoder (Up Sampling part)
        self.up_pool6 = up_pooling(1024, 512)
        self.conv6 = conv_bn_relu(1024, 512)
        self.up_pool7 = up_pooling(512, 256)
        self.conv7 = conv_bn_relu(512, 256)
        self.up_pool8 = up_pooling(256, 128)
        self.conv8 = conv_bn_relu(256, 128)
        self.up_pool9 = up_pooling(128, 64)
        self.conv9 = conv_bn_relu(128, 64)
        self.conv10 = nn.Conv2d(64, output_channels, 1)

        # Initialize weights using Kaiming Normal initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the UNet model.

        Parameters:
        - x (Tensor): Input tensor.

        Returns:
            Tensor: Segmentation output tensor.
        """
        # 正規化
        x = x / 255.0

        # Forward pass through the encoder (FCN part)
        x1 = self.conv1(x)
        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)
        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)
        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)
        p4 = self.down_pooling(x4)
        x5 = self.conv5(p4)

        # Forward pass through the decoder (Up Sampling part), with Skip Connections using torch.cat
        p6 = self.up_pool6(x5)
        x6 = torch.cat([p6, x4], dim=1)
        x6 = self.conv6(x6)

        p7 = self.up_pool7(x6)
        x7 = torch.cat([p7, x3], dim=1)
        x7 = self.conv7(x7)

        p8 = self.up_pool8(x7)
        x8 = torch.cat([p8, x2], dim=1)
        x8 = self.conv8(x8)

        p9 = self.up_pool9(x8)
        x9 = torch.cat([p9, x1], dim=1)
        x9 = self.conv9(x9)

        output = self.conv10(x9)
        output = torch.sigmoid(output)

        return output


# 畳み込みとバッチ正規化と活性化関数Reluをまとめている
def conv_bn_relu(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
) -> nn.Sequential:
    """
    Utility function to create a sequence of convolution, batch normalization, and ReLU layers.

    Parameters:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    - kernel_size (int): Size of the convolutional kernel.
    - stride (int): Stride value for the convolution operation.
    - padding (int): Padding value for the convolution operation.

    Returns:
        nn.Sequential: Sequence of convolution, batch normalization, and ReLU layers.
    """
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def down_pooling() -> nn.MaxPool2d:
    """
    Utility function to create a max pooling layer.

    Returns:
    - nn.MaxPool2d: Max pooling layer.
    """
    return nn.MaxPool2d(2)


def up_pooling(
    in_channels: int, out_channels: int, kernel_size: int = 2, stride: int = 2
) -> nn.Sequential:
    """
    Utility function to create a sequence of transposed convolution, batch normalization, and ReLU layers.

    Parameters:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    - kernel_size (int): Size of the transposed convolutional kernel.
    - stride (int): Stride value for the transposed convolution operation.

    Returns:
        nn.Sequential: Sequence of transposed convolution, batch normalization, and ReLU layers.
    """
    return nn.Sequential(
        # 転置畳み込み
        nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
