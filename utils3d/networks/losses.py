import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    """二分类焦点损失函数，适用于类别不平衡场景
    
    通过降低易分类样本的权重，使得模型在训练时更关注难分类样本
    论文: <https://arxiv.org/abs/1708.02002>

    Attributes:
        gamma (float): 调节因子，用于调整困难样本的权重
        alpha (float): 类别权重平衡参数，范围(0,1)
        logits (bool): 输入是否为logits（启用sigmoid）
        reduce (bool): 是否对损失进行均值降维
        loss_weight (float): 损失权重系数
    """

    def __init__(self, gamma=2.0, alpha=0.5, logits=True, reduce=True, loss_weight=1.0):
        super(BinaryFocalLoss, self).__init__()
        assert 0 < alpha < 1
        self.gamma = gamma
        self.alpha = alpha
        self.logits = logits
        self.reduce = reduce
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """前向计算函数

        Args:
            pred (torch.Tensor): 模型预测值，形状为(N)
            target (torch.Tensor): 真实标签，形状为(N)。若为类别索引，每个值应为0或1；
                若为概率值，形状需与pred一致

        Returns:
            torch.Tensor: 计算得到的损失值
        """
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        else:
            bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)  # 计算概率p_t
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)  # 动态alpha平衡
        focal_loss = alpha * (1 - pt) ** self.gamma * bce  # 焦点损失公式

        if self.reduce:
            focal_loss = torch.mean(focal_loss)
        return focal_loss * self.loss_weight


class FocalLoss(nn.Module):
    """多分类焦点损失函数，支持类别忽略
    
    适用于多分类任务的焦点损失实现，支持指定忽略类别
    论文: <https://arxiv.org/abs/1708.02002>

    Attributes:
        gamma (float): 困难样本调节因子
        alpha (float/list): 类别权重，可设置为列表指定各类别权重
        reduction (str): 损失降维方式，可选'mean'或'sum'
        loss_weight (float): 损失项的缩放权重
        ignore_index (int): 需要忽略的类别索引
    """

    def __init__(
        self, gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1
    ):
        super(FocalLoss, self).__init__()
        assert reduction in (
            "mean",
            "sum",
        ), "reduction应为'mean'或'sum'"
        assert isinstance(
            alpha, (float, list)
        ), "alpha应为float或list类型"
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        """前向计算函数

        Args:
            pred (torch.Tensor): 模型预测值，形状为(N, C)，C为类别数
            target (torch.Tensor): 真实标签。若为类别索引，形状为(N)，值范围为0~C-1；
                若为概率，形状需与pred一致

        Returns:
            torch.Tensor: 计算得到的损失值
        """
        # 调整张量形状
        pred = pred.transpose(0, 1).reshape(pred.size(0), -1).transpose(0, 1).contiguous()
        target = target.view(-1).contiguous()
        
        # 验证形状一致性
        assert pred.size(0) == target.size(0), "预测与标签形状不匹配"
        
        # 过滤忽略索引
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        if len(target) == 0:
            return 0.0

        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes)  # 转换为one-hot编码

        # 处理alpha参数
        alpha = self.alpha
        if isinstance(alpha, list):
            alpha = pred.new_tensor(alpha)
        
        # 计算焦点权重
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * one_minus_pt.pow(self.gamma)

        # 计算加权损失
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none") * focal_weight
        
        # 降维处理
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return self.loss_weight * loss


class DiceLoss(nn.Module):
    """Dice系数损失函数，适用于分割任务
    
    通过测量预测与标签的相似度进行优化，尤其适用于类别不平衡场景
    论文: <https://arxiv.org/abs/1606.04797>

    Attributes:
        smooth (float): 平滑系数，防止除零
        exponent (int): 指数参数，控制计算方式
        loss_weight (float): 损失项的缩放权重
        ignore_index (int): 需要忽略的类别索引
    """

    def __init__(self, smooth=1, exponent=2, loss_weight=1.0, ignore_index=-1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        """前向计算函数

        Args:
            pred (torch.Tensor): 模型预测值，形状为(B, C, d1, d2, ...)
            target (torch.Tensor): 真实标签，形状为(B, d1, d2, ...)

        Returns:
            torch.Tensor: 计算得到的损失值
        """
        # 调整张量形状
        pred = pred.transpose(0, 1).reshape(pred.size(0), -1).transpose(0, 1).contiguous()
        target = target.view(-1).contiguous()
        
        # 验证形状一致性
        assert pred.size(0) == target.size(0), "预测与标签形状不匹配"
        
        # 过滤忽略索引
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        # 计算softmax概率
        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes
        )

        # 逐类别计算Dice损失
        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                numerator = torch.sum(torch.mul(pred[:, i], target[:, i])) * 2 + self.smooth
                denominator = torch.sum(pred[:, i].pow(self.exponent) + target[:, i].pow(self.exponent)) + self.smooth
                dice_loss = 1 - numerator / denominator
                total_loss += dice_loss
        
        # 平均损失并加权
        loss = total_loss / num_classes
        return self.loss_weight * loss
    
    
    
def dice_loss_multi_classes(input, target, epsilon=1e-5, weight=None):
    r"""
    多类别Dice损失函数，用于语义分割任务，计算每个类别的Dice系数并转化为损失。
    修改自：https://github.com/wolny/pytorch-3dunet/blob/.../losses.py 的compute_per_channel_dice

    参数：
        input (Tensor): 模型预测的输出，形状为(batch_size, num_classes, [depth, height, width])
        target (Tensor): 真实标签的one-hot编码，形状需与input相同
        epsilon (float): 平滑系数，防止分母为零，默认为1e-5
        weight (Tensor, optional): 各类别的权重，形状应为(num_classes, )

    返回：
        Tensor: 每个类别的Dice损失，形状为(num_classes, )

    注意：
        - 输入和目标的维度必须完全一致
        - 本实现暂未使用weight参数，如需加权需后续手动处理
    """
    # 校验输入形状一致性
    assert input.size() == target.size(), "'input'和'target'的维度必须完全相同"

    # 调整维度顺序，将类别通道移至第0维，便于逐类别计算
    # 原维度假设为(batch, num_classes, ...)，调整后为(num_classes, batch, ...)
    axis_order = (1, 0) + tuple(range(2, input.dim()))
    input = input.permute(axis_order)
    target = target.permute(axis_order)

    # 转换目标类型为float，确保与预测值类型匹配
    target = target.float()

    # 计算逐类别的Dice系数
    # 分子：2 * 预测与目标的逐元素乘积之和（按批次和空间维度求和）
    # 分母：预测值的平方和 + 目标值的平方和 + 平滑项
    per_channel_dice = (2 * torch.sum(input * target, dim=1) + epsilon) / \
                       (torch.sum(input * input, dim=1) + torch.sum(target * target, dim=1) + 1e-4 + epsilon)

    # 将Dice系数转换为损失（1 - Dice）
    loss = 1. - per_channel_dice

    # 若需类别加权，可在此处添加权重计算（当前实现未使用weight参数）
    # if weight is not None:
    #     loss = loss * weight

    return loss