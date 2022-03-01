import torch
from mlp_model.mlp_mixer import MLP_Mixer
from dataset import get_test_loader
from tqdm import tqdm
from utils.test_fun import cal_pr_index

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 模型对象
mlp_net = MLP_Mixer(image_size=45, patch_size=5, dim=256, num_classes=3, num_blocks=8, token_dim=256, channel_dim=2048)
mlp_net.to(device)

# 加载模型
mlp_net.load_state_dict(torch.load('./trained_model/best_model.pth'))

# 验证数据加载器
test_loader = get_test_loader(batch_size=128, test_name='ln')

tps = 0
fns = 0

with torch.no_grad():
    for batch in tqdm(test_loader):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = mlp_net(inputs)
        predicts = torch.argmax(outputs, dim=1)
        tp, fn = cal_pr_index(predicts)
        tps += tp
        fns += fn

print('tps = ', tps)
print('fns = ', fns)
print('recall = ', tps / (tps + fns))
