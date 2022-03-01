import torch
from mlp_model.mlp_mixer import MLP_Mixer
from utils.inference_fun import get_img_tensor

# 推理图像路径
img_path = './inference/o_1d1_14_Ch1.ome.bmp'

# 模型对象
mlp_net = MLP_Mixer(image_size=45, patch_size=5, dim=256, num_classes=3, num_blocks=8, token_dim=256, channel_dim=2048)

# 加载模型
mlp_net.load_state_dict(torch.load('./trained_model/best_model.pth'))

# 图像tensor
img_tensor = get_img_tensor(img_path)

# 模型输出
with torch.no_grad():
    output = mlp_net(img_tensor)

# 预测序号
label = torch.argmax(output, dim=1).item()

label_dict = {0: '星形胶质细胞', 1: '少突胶质细胞', 2: '神经元细胞'}

print(label_dict[label])
print(output)
