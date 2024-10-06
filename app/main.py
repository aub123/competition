import os
import numpy as np
import pandas as pd
import cv2
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# 定义U-Net模型
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(1, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = self.conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = self.conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.encoder4 = self.conv_block(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = self.conv_block(256, 512)

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(256, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(128, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(64, 32)

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.decoder4(torch.cat([self.upconv4(bottleneck), enc4], dim=1))
        dec3 = self.decoder3(torch.cat([self.upconv3(dec4), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([self.upconv2(dec3), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([self.upconv1(dec2), enc1], dim=1))

        return torch.sigmoid(self.final_conv(dec1))

# 自定义数据集
class ThyroidDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# 数据加载
train_img_dir = '/tcdata/train/img'
train_mask_dir = '/tcdata/train/label'
train_images = [os.path.join(train_img_dir, img) for img in os.listdir(train_img_dir)]
train_masks = [os.path.join(train_mask_dir, mask) for mask in os.listdir(train_mask_dir)]

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = ThyroidDataset(train_images, train_masks, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)



# 初始化模型、损失函数和优化器
model = UNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# 模型文件路径
model_path = 'unet_model.pth'

# 检查模型文件是否存在
if os.path.exists(model_path):
    # 加载模型权重
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 切换到评估模式
    print("模型已加载。")
else:
    print("模型不存在，将开始训练。")    
    # 训练模型
    for epoch in tqdm(range(100),desc=f"training:"):  
        model.train()
        for images, masks in train_loader:
            optimizer.zero_grad()
            
            # 确保 images 和 masks 的维度是 [batch_size, 1, height, width]
            outputs = model(images)
            loss = criterion(outputs, masks.float())  # masks 也直接使用
            loss.backward()
            optimizer.step()
    # 保存模型
    torch.save(model.state_dict(), model_path)        
        
# 模型预测（假设 test_images 是测试集）
test_img_dir = '/tcdata/test/img'
test_images = np.array([os.path.join(test_img_dir, img) for img in os.listdir(test_img_dir)])
model.eval()
predictions = []

for img_path in test_images:
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # 使用 ToTensor 转换为 tensor 并增加 batch 维度 [batch_size, channels, height, width]
    image_tensor = transform(image).unsqueeze(0)  # 增加 batch 维度, 变成 [1, 1, height, width]

    with torch.no_grad():
        output = model(image_tensor)  # 输出的形状为 [1, 1, height, width]
    
    mask = (output.squeeze().numpy() > 0.5).astype(np.uint8)  # 二值化，输出 [height, width]
    predictions.append(mask)
    
    
# 保存预测结果和生成label.csv
os.makedirs('/app/submit', exist_ok=True)
os.makedirs('/app/submit/label', exist_ok=True)

label_data = []

for i, img_name in enumerate(os.listdir(test_img_dir)):
    mask_path = f'/app/submit/label/{img_name}'
    cv2.imwrite(mask_path, predictions[i] * 255)

    case_name = img_name.split('.')[0]
    prob = np.random.rand()  # 这里可以替换为实际预测的恶性概率
    label_data.append({'case': case_name, 'prob': prob})

label_df = pd.DataFrame(label_data)
label_df.to_csv('/app/submit/label.csv', index=False)

# 打包提交文件
with zipfile.ZipFile('/app/submit.zip', 'w') as zipf:
    for root, dirs, files in os.walk('/app/submit/submit'):
        for file in files:
            zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), '/app/submit'))

print("提交文件已生成：/app/submit.zip")
