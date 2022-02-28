import torch
from mlp_model.mlp_mixer import MLP_Mixer
from dataset import get_val_loader

# 模型对象
mlp_net = MLP_Mixer(image_size=45, patch_size=5, dim=256, num_classes=3, num_blocks=8, token_dim=256, channel_dim=2048)

# 加载模型
mlp_net.load_state_dict(torch.load('./trained_model/best_model.pth'))

# 验证数据加载器
val_loader = get_val_loader(batch_size=128)
