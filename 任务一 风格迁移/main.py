# 导入模块
from torchvision import transforms
from torchvision import models
from PIL import Image
import torch.nn as nn
import torchvision
import argparse
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 定义设备为GPU，若无GPU则使用CPU。


def load_image(image_path, transform=None, max_size=None, shape=None):
    """
    定义函数，读取图像文件并将其转换为张量，并对图像进行尺寸调整等预处理操作。
    :param image_path:代表图像文件的路径地址。
    :param transform:可选参数，表示转换函数，对图像进行归一化、裁剪等于处理。
    :param max_size:可选参数，对图像进行缩放，使最长边不超过设定值。
    :param shape:可选参数，若指定参数则会缩放图像至指定尺寸。
    """
    image = Image.open(image_path)  # 读取图像
    if max_size:
        image = image.resize((max_size, max_size), Image.LANCZOS)
    if shape:
        image = image.resize(shape, Image.LANCZOS)
    if transform:
        image = transform(image).unsqueeze(0)
    return image.to(device)  # 将处理好的张量数据传输到指定设备。


# 主要功能是从预训练的VGG19网络中提取特定层的特征图。
class VGGNet(nn.Module):
    # 定义初始化方法。
    def __init__(self):
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19', '28']  # 存储VGG19网络中需要被提取特征的层的索引。
        self.vgg = models.vgg19(pretrained=True).features  # 加载预训练的VGG19网络的特征提取部分。

    # 定义前向传播方法。
    def forward(self, x):
        features = []
        # 遍历每一层，通过层进行处理。
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            # 遇到索引存在于self.select中的层时，将层输出的特征图添加到feature列表中。
            if name in self.select:
                features.append(x)
        return features


def main(config):
    # 定义图像与处理步骤，包括转换为张量和归一化处理。
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    content = load_image(config.content, transform, max_size=config.max_size)  # 加载内容图像
    style = load_image(config.style, transform, shape=[content.size(2), content.size(3)])  # 加载风格图像
    target = content.clone().requires_grad_(True)  # 用内容图像初始化目标图像
    optimizer = torch.optim.Adam([target], lr=config.lr, betas=[0.5, 0.999])  # 使用Adam优化器对目标图像的像素值进行优化。
    vgg = VGGNet().to(device).eval()  # 加载预训练VGG网络，提取图像特征，并设置为评估模式。
    # 训练循环
    for step in range(config.total_step):
        # 在每次迭代中，分别提取目标图像、内容图像和风格图像在VGG网络中多个层的特征。
        target_features = vgg(target)
        content_features = vgg(content)
        style_features = vgg(style)
        # 初始化损失值
        style_loss = content_loss = 0
        for f1, f2, f3 in zip(target_features, content_features, style_features):
            content_loss += torch.mean((f1 - f2) ** 2)  # 计算内容损失
            # 计算Gram矩阵表示风格
            _, c, h, w = f1.size()
            f1 = f1.view(c, h * w)
            f3 = f3.view(c, h * w)
            f1 = torch.mm(f1, f1.t())
            f3 = torch.mm(f3, f3.t())
            style_loss += torch.mean((f1 - f3) ** 2) / (c * h * w)  # 计算风格损失
        loss = content_loss + config.style_weight * style_loss  # 总损失是内容损失和风格损失的加权和。
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新目标图像的像素值
        # 定期输出相关信息
        if (step + 1) % config.log_step == 0:
            print('Step [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}'
                  .format(step + 1, config.total_step, content_loss.item(), style_loss.item()))
        # 定期保存中间结果
        if (step + 1) % config.sample_step == 0:
            denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
            img = target.clone().squeeze()
            img = denorm(img).clamp_(0, 1)
            torchvision.utils.save_image(img, 'output-{}.png'.format(step + 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建参数解析器对象，用于处理命令行输入。
    parser.add_argument('--content', type=str, default='png/tyut.jpg')  # 指定内容图像的路径。
    parser.add_argument('--style', type=str, default='png/starry_night.jpg')  # 指定风格图像的路径。
    parser.add_argument('--max_size', type=int, default=400)  # 设置图像的最大尺寸
    parser.add_argument('--total_step', type=int, default=2000)  # 设置优化过程的总步数，步数越多，风格融合越充分。
    parser.add_argument('--log_step', type=int, default=10)  # 日志记录的频率，即每多少步输出一次信息。
    parser.add_argument('--sample_step', type=int, default=500)  # 保存中间结果的频率，即每多少步保存一次生成的图像。
    parser.add_argument('--style_weight', type=float, default=100)  # 控制风格迁移的强度，值越大，生成图像越偏向风格图像。
    parser.add_argument('--lr', type=float, default=0.003)  # 设定学习率
    config = parser.parse_args()
    print(config)
    main(config)
