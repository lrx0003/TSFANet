import torch
from torchsummary import summary
from thop import profile
from thop import clever_format
import torchvision.models as models

from src.TSFANet import net

# 设置设备为 CUDA（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型并将其移动到设备上
model = net().to(device)
model.eval()

# 随机生成一个输入张量并将其移动到设备上
input = torch.randn(1, 3, 448, 448).to(device)  # 确保输入张量也在正确的设备上

# 使用 thop 分析模型的运算量和参数量
MACs, params = profile(model, inputs=(input,))

# 将结果转换为更易于阅读的格式
MACs, params = clever_format([MACs, params], '%.3f')

print(f"运算量：{MACs}, 参数量：{params}")
