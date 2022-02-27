import torch
from dataset import get_train_loader, get_val_loader
from mlp_model.mlp_mixer import MLP_Mixer
import torch.optim as optim
import os

# 总训练轮数
epochs = 2
# 学习率
lr = 0.001

# 定义模型对象
mlp_net = MLP_Mixer(image_size=45, patch_size=5, dim=256, num_classes=3, num_blocks=8, token_dim=256, channel_dim=2048)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp_net.parameters(), lr=lr)

# 获取数据加载器
train_loader = get_train_loader(batch_size=128)
val_loader = get_val_loader(batch_size=128)

# 打开训练日志文件
train_log_f = open('./log/train-loss.log', 'a')
# 打开验证日志文件
val_log_f = open('./log/val-loss.log', 'a')

# 判断是否支持gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 在支持gpu的情况下将模型载入gpu
mlp_net.to(device)

# 判断是否存在模型保存路径
if not os.path.exists('./trained_model'):
    os.mkdir('./trained_model')
model_path = './trained_model'

# 最佳验证loss
best_val_loss = 999

# 训练模型
for epoch in range(2):
    loader = iter(train_loader)
    for step, data in enumerate(loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = mlp_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # 每5步打印一次训练loss
        if step % 5 == 0:
            loss_str = f'epoch: {epoch} step: {step} train_loss: {loss} \n'
            train_log_f.write(loss_str)
            print(loss_str)
    # 每训练一轮计算一次验证集loss
    v_loader = iter(val_loader)
    val_losses = []
    for val_step, val_data in enumerate(v_loader):
        inputs, labels = val_data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = mlp_net(inputs)
        loss = criterion(outputs, labels)
        val_losses.append(loss)
    val_loss = torch.tensor(val_losses).mean().detach().item()
    val_loss_str = f'epoch: {epoch} val_loss: {val_loss} \n'
    val_log_f.write(val_loss_str)
    print(val_loss_str)
    # 保存在验证集上loss最低的模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(mlp_net.state_dict(), os.path.join(model_path, 'best_model.pth'))
# 最后再保存一次模型
torch.save(mlp_net.state_dict(), os.path.join(model_path, 'final_model.pth'))

# 关闭日志文件
train_log_f.close()
val_log_f.close()


