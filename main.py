import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
import cv2
import os
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免图表显示问题
import matplotlib.pyplot as plt

# 创建保存图表的文件夹
FIGURES_DIR = Path("./figures")
FIGURES_DIR.mkdir(exist_ok=True)
print(f"图表将保存到: {FIGURES_DIR.absolute()}")

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def setup_device():
    """设置训练设备"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
    else:
        print("使用CPU进行训练")
    return device

DEVICE = setup_device()

DISEASE_CLASSES = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

CLASS_DISPLAY_NAMES = [
    '细菌性斑点病',
    '早疫病',
    '晚疫病',
    '叶霉病',
    '斑枯病',
    '红蜘蛛',
    '靶斑病',
    '黄化曲叶病毒病',
    '花叶病毒病',
    '健康叶片'
]

CLASS_TO_ID = {cls_name: i for i, cls_name in enumerate(DISEASE_CLASSES)}
ID_TO_CLASS = {i: cls_name for i, cls_name in enumerate(DISEASE_CLASSES)}
ID_TO_DISPLAY = {i: display_name for i, display_name in enumerate(CLASS_DISPLAY_NAMES)}

print(f"总共有 {len(DISEASE_CLASSES)} 个病害类别")

# ============================ 轻量化模型 ============================
class EnhancedTomatoModel(nn.Module):
    def __init__(self, num_classes=10):
        super(EnhancedTomatoModel, self).__init__()

        # 更深的特征提取器
        self.features = nn.Sequential(
            # 输入: 3x256x256
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x128x128

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128x64x64

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 256x32x32

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 512x16x16

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 512x8x8
        )

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 更强的分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ============================ 数据集处理 ============================
class TomatoDatasetManager:
    """番茄病害数据集"""

    @staticmethod
    def load_dataset(dataset_path="TomatoDataset"):
        """加载整个数据集"""
        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")

        # 收集所有图像
        image_paths = []
        labels = []

        for class_idx, class_name in enumerate(DISEASE_CLASSES):
            class_dir = dataset_path / class_name

            if not class_dir.exists():
                # 尝试不同的命名格式
                class_name_alt = class_name.replace('___', '__')
                class_dir = dataset_path / class_name_alt

            if not class_dir.exists():
                print(f"类别目录不存在: {class_name}")
                continue

            # 收集所有图片
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            class_images = []
            for ext in image_extensions:
                class_images.extend(class_dir.glob(ext))

            if len(class_images) == 0:
                print(f"警告: 类别 {class_name} 中没有找到图片")
                continue

            for img_path in class_images:
                image_paths.append(str(img_path))
                labels.append(class_idx)

            print(f"  {CLASS_DISPLAY_NAMES[class_idx]:20s}: {len(class_images)} 张图片")

        print(f"总共加载 {len(image_paths)} 张图片")

        # 检查是否有数据
        if len(image_paths) == 0:
            raise ValueError("没有找到任何图片文件，请检查数据集路径和结构")

        return image_paths, labels

    @staticmethod
    def create_dataloaders(image_paths, labels, batch_size=16, img_size=256):
        """创建数据加载器"""
        print("\n创建数据加载器...")

        # 划分数据集 (70%训练, 15%验证, 15%测试)
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels, test_size=0.3, stratify=labels, random_state=42
        )

        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
        )

        print(f" 数据集划分:")
        print(f"  训练集: {len(train_paths)} 张图片")
        print(f"  验证集: {len(val_paths)} 张图片")
        print(f"  测试集: {len(test_paths)} 张图片")

        # 更强的数据增强配置
        train_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, p=0.2),
            A.RandomRain(p=0.1),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        val_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        # 创建数据集
        train_dataset = SimpleImageDataset(train_paths, train_labels, train_transform)
        val_dataset = SimpleImageDataset(val_paths, val_labels, val_transform)
        test_dataset = SimpleImageDataset(test_paths, test_labels, val_transform)

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=0, pin_memory=True)

        return train_loader, val_loader, test_loader

class SimpleImageDataset(Dataset):
    """简单的图像数据集类"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 读取图像
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"无法读取图片: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f" 错误读取图片 {img_path}: {e}")
            # 创建默认图像
            image = np.ones((256, 256, 3), dtype=np.uint8) * 128

        # 应用变换
        if self.transform:
            try:
                transformed = self.transform(image=image)
                image = transformed['image']
            except Exception as e:
                print(f" 数据增强错误: {e}")
                # 简单转换
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return image, torch.tensor(label, dtype=torch.long)

# ============================ 训练器类 ============================
class TomatoDiseaseTrainer:
    """番茄病害训练器"""

    def __init__(self, num_classes=10, model_path=None):
        self.device = DEVICE
        self.num_classes = num_classes

        # 初始化增强模型
        self.model = EnhancedTomatoModel(num_classes=num_classes).to(self.device)

        # 使用带标签平滑的损失函数
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 使用SGD优化器（通常比Adam在CNN上表现更好）
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=1e-4
        )

        # 使用余弦退火学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=30, eta_min=1e-6
        )

        # 训练历史
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'learning_rates': []
        }

        # 如果提供了模型路径，则加载模型
        if model_path and Path(model_path).exists():
            self.load_model(model_path)

        print(f" 模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")

    def train_epoch(self, dataloader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(dataloader, desc=f'训练 Epoch {epoch+1}')

        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 统计信息
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条
            acc = 100. * correct / total
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.2f}%'
            })

        avg_loss = total_loss / len(dataloader)
        avg_acc = 100. * correct / total

        return avg_loss, avg_acc

    def validate(self, dataloader, desc="验证"):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=desc):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(dataloader)
        avg_acc = 100. * correct / total

        return avg_loss, avg_acc

    def train(self, train_loader, val_loader, epochs=40, model_name="tomato_model"):
        """完整训练流程"""
        print("\n 开始训练模型...")

        best_val_acc = 0
        patience = 8  # 增加早停耐心值
        patience_counter = 0

        for epoch in range(epochs):
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(train_loader, epoch)

            # 验证
            val_loss, val_acc = self.validate(val_loader)

            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)

            # 打印进度
            print(f"\nEpoch {epoch+1}/{epochs}:")
            print(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
            print(f"  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
            print(f"  学习率: {current_lr:.6f}")

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(f"{model_name}_best.pth")
                print(f"  保存最佳模型，验证准确率: {val_acc:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  早停触发，训练停止")
                    break

            # 定期保存检查点
            if (epoch + 1) % 5 == 0:
                self.save_model(f"{model_name}_epoch{epoch+1}.pth")
                # 每5个epoch绘制一次训练历史
                self.plot_training_history_partial(epoch+1)

        # 保存最终模型
        self.save_model(f"{model_name}_final.pth")
        print(f"\n训练完成 最佳验证准确率: {best_val_acc:.2f}%")

        # 绘制完整的训练历史
        self.plot_complete_training_history()

        return best_val_acc

    def evaluate(self, test_loader):
        """评估模型性能"""
        print("\n 评估模型性能...")

        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="测试"):
                images = images.to(self.device)
                outputs = self.model(images)
                probabilities = F.softmax(outputs, dim=1)
                _, predictions = outputs.max(1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # 计算总体准确率
        accuracy = accuracy_score(all_labels, all_predictions) * 100
        print(f"测试准确率: {accuracy:.2f}%")

        # 生成分类报告
        print("\n分类报告:")
        report = classification_report(all_labels, all_predictions,
                                       target_names=CLASS_DISPLAY_NAMES, digits=4, output_dict=True)
        print(classification_report(all_labels, all_predictions,
                                    target_names=CLASS_DISPLAY_NAMES, digits=4))

        # 绘制混淆矩阵
        self.plot_confusion_matrix(all_labels, all_predictions)

        # 绘制精度-召回率曲线
        self.plot_precision_recall_curve(report)

        # 计算每个类别的准确率
        self.print_class_accuracy(all_labels, all_predictions)

        # 绘制各类别准确率条形图
        self.plot_class_accuracy_barchart(all_labels, all_predictions)

        return accuracy

    def plot_confusion_matrix(self, y_true, y_pred):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                   xticklabels=CLASS_DISPLAY_NAMES,
                   yticklabels=CLASS_DISPLAY_NAMES,
                   cbar_kws={'shrink': 0.8})
        plt.title('混淆矩阵', fontsize=18, fontweight='bold')
        plt.xlabel('预测标签', fontsize=14)
        plt.ylabel('真实标签', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        # 保存混淆矩阵
        cm_path = FIGURES_DIR / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图表，避免内存泄漏
        print(f"混淆矩阵已保存为 '{cm_path}'")

    def plot_precision_recall_curve(self, report):
        """绘制精度-召回率曲线"""
        # 提取每个类别的精度和召回率
        classes = CLASS_DISPLAY_NAMES
        precisions = [report[cls]['precision'] for cls in classes]
        recalls = [report[cls]['recall'] for cls in classes]
        f1_scores = [report[cls]['f1-score'] for cls in classes]

        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 精度条形图
        bars1 = axes[0].bar(range(len(classes)), precisions, color='skyblue')
        axes[0].set_xlabel('病害类别', fontsize=12)
        axes[0].set_ylabel('精度', fontsize=12)
        axes[0].set_title('各类别精度', fontsize=14, fontweight='bold')
        axes[0].set_xticks(range(len(classes)))
        axes[0].set_xticklabels(classes, rotation=45, ha='right')
        axes[0].set_ylim([0, 1.05])

        # 在柱状图上添加数值
        for bar, val in zip(bars1, precisions):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        # 召回率条形图
        bars2 = axes[1].bar(range(len(classes)), recalls, color='lightgreen')
        axes[1].set_xlabel('病害类别', fontsize=12)
        axes[1].set_ylabel('召回率', fontsize=12)
        axes[1].set_title('各类别召回率', fontsize=14, fontweight='bold')
        axes[1].set_xticks(range(len(classes)))
        axes[1].set_xticklabels(classes, rotation=45, ha='right')
        axes[1].set_ylim([0, 1.05])

        # 在柱状图上添加数值
        for bar, val in zip(bars2, recalls):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        # F1分数条形图
        bars3 = axes[2].bar(range(len(classes)), f1_scores, color='lightcoral')
        axes[2].set_xlabel('病害类别', fontsize=12)
        axes[2].set_ylabel('F1分数', fontsize=12)
        axes[2].set_title('各类别F1分数', fontsize=14, fontweight='bold')
        axes[2].set_xticks(range(len(classes)))
        axes[2].set_xticklabels(classes, rotation=45, ha='right')
        axes[2].set_ylim([0, 1.05])

        # 在柱状图上添加数值
        for bar, val in zip(bars3, f1_scores):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        # 保存精度-召回率曲线
        pr_path = FIGURES_DIR / "precision_recall_curve.png"
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"精度-召回率曲线已保存为 '{pr_path}'")

    def plot_class_accuracy_barchart(self, y_true, y_pred):
        """绘制各类别准确率条形图"""
        class_accuracies = []

        for class_idx in range(self.num_classes):
            class_mask = np.array(y_true) == class_idx
            if np.sum(class_mask) > 0:
                class_correct = np.sum((np.array(y_pred)[class_mask] == class_idx))
                class_total = np.sum(class_mask)
                class_accuracy = 100. * class_correct / class_total
                class_accuracies.append(class_accuracy)
            else:
                class_accuracies.append(0)

        # 创建条形图
        plt.figure(figsize=(14, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(CLASS_DISPLAY_NAMES)))
        bars = plt.bar(range(len(CLASS_DISPLAY_NAMES)), class_accuracies, color=colors)

        plt.xlabel('病害类别', fontsize=14)
        plt.ylabel('准确率 (%)', fontsize=14)
        plt.title('各类别测试准确率', fontsize=16, fontweight='bold')
        plt.xticks(range(len(CLASS_DISPLAY_NAMES)), CLASS_DISPLAY_NAMES, rotation=45, ha='right')
        plt.ylim([0, 105])

        # 在柱状图上添加数值
        for bar, acc in zip(bars, class_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        # 保存各类别准确率条形图
        ca_path = FIGURES_DIR / "class_accuracy_barchart.png"
        plt.savefig(ca_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"各类别准确率条形图已保存为 '{ca_path}'")

    def print_class_accuracy(self, y_true, y_pred):
        """打印每个类别的准确率"""
        print("\n各类别准确率:")
        for class_idx in range(self.num_classes):
            class_mask = np.array(y_true) == class_idx
            if np.sum(class_mask) > 0:
                class_correct = np.sum((np.array(y_pred)[class_mask] == class_idx))
                class_total = np.sum(class_mask)
                class_accuracy = 100. * class_correct / class_total
                print(f"  {CLASS_DISPLAY_NAMES[class_idx]:25s}: {class_accuracy:6.2f}% "
                      f"({class_correct:3d}/{class_total:3d})")

    def plot_training_history_partial(self, current_epoch):
        """绘制部分训练历史（用于训练过程中）"""
        if len(self.history['train_loss']) == 0:
            return

        epochs = range(1, len(self.history['train_loss']) + 1)

        # 创建损失图
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.history['train_loss'], 'b-', label='训练损失', linewidth=2, marker='o')
        plt.plot(epochs, self.history['val_loss'], 'r-', label='验证损失', linewidth=2, marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'训练历史 - 损失 (Epoch {current_epoch})', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # 保存损失图
        loss_path = FIGURES_DIR / f"training_loss_epoch{current_epoch}.png"
        plt.savefig(loss_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 创建准确率图
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.history['train_acc'], 'b-', label='训练准确率', linewidth=2, marker='o')
        plt.plot(epochs, self.history['val_acc'], 'r-', label='验证准确率', linewidth=2, marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(f'训练历史 - 准确率 (Epoch {current_epoch})', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # 保存准确率图
        acc_path = FIGURES_DIR / f"training_accuracy_epoch{current_epoch}.png"
        plt.savefig(acc_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"训练历史图已保存 (Epoch {current_epoch})")

    def plot_complete_training_history(self):
        """绘制完整的训练历史"""
        if len(self.history['train_loss']) == 0:
            print("没有训练历史可绘制")
            return

        epochs = range(1, len(self.history['train_loss']) + 1)

        # 创建4个子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 训练/验证损失
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='训练损失', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='验证损失', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('训练历史 - 损失', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 训练/验证准确率
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='训练准确率', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='验证准确率', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('训练历史 - 准确率', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 学习率变化
        axes[1, 0].plot(epochs, self.history['learning_rates'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('学习率变化', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # 损失-准确率关系
        axes[1, 1].scatter(self.history['val_loss'], self.history['val_acc'],
                          c=range(len(self.history['val_loss'])), cmap='viridis', s=80)
        axes[1, 1].set_xlabel('验证损失')
        axes[1, 1].set_ylabel('验证准确率 (%)')
        axes[1, 1].set_title('损失-准确率关系', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        # 添加颜色条
        scatter = axes[1, 1].collections[0]
        plt.colorbar(scatter, ax=axes[1, 1], label='Epoch')

        plt.tight_layout()

        # 保存完整训练历史图
        history_path = FIGURES_DIR / "complete_training_history.png"
        plt.savefig(history_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"完整训练历史图已保存为 '{history_path}'")

    def predict_single_image(self, image_path, show_result=True):
        """预测单张图片"""
        self.model.eval()

        # 读取图像
        if not Path(image_path).exists():
            raise FileNotFoundError(f"图片不存在: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        original_image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 预处理
        transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        transformed = transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        result = {
            'class_id': predicted.item(),
            'class_name': DISEASE_CLASSES[predicted.item()],
            'display_name': CLASS_DISPLAY_NAMES[predicted.item()],
            'confidence': confidence.item(),
            'probabilities': probabilities[0].cpu().numpy(),
            'top3_predictions': []
        }

        # 获取前3个预测结果
        top3_indices = np.argsort(result['probabilities'])[-3:][::-1]
        for idx in top3_indices:
            result['top3_predictions'].append({
                'class_id': idx,
                'display_name': CLASS_DISPLAY_NAMES[idx],
                'probability': result['probabilities'][idx]
            })

        # 保存预测结果图
        if show_result:
            self.visualize_prediction(original_image, result, Path(image_path).stem)

        return result

    def visualize_prediction(self, image, result, image_name):
        """可视化预测结果"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # 左侧：原始图像
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"输入图像\n预测: {result['display_name']}\n置信度: {result['confidence']:.2%}",
                         fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # 右侧：概率分布
        y_pos = np.arange(len(CLASS_DISPLAY_NAMES))
        colors = ['lightcoral' if i == result['class_id'] else 'lightblue'
                  for i in range(len(CLASS_DISPLAY_NAMES))]
        bars = axes[1].barh(y_pos, result['probabilities'], color=colors)
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(CLASS_DISPLAY_NAMES)
        axes[1].set_xlabel('概率', fontsize=12)
        axes[1].set_title('各类别概率分布', fontsize=14, fontweight='bold')
        axes[1].set_xlim([0, 1])

        # 添加概率值
        for i, (bar, prob) in enumerate(zip(bars, result['probabilities'])):
            axes[1].text(prob + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{prob:.3f}', va='center', fontsize=9)

        plt.tight_layout()

        # 保存预测结果图
        prediction_path = FIGURES_DIR / f"prediction_{image_name}.png"
        plt.savefig(prediction_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"预测结果图已保存为 '{prediction_path}'")

    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'num_classes': self.num_classes,
            'classes': DISEASE_CLASSES,
            'display_classes': CLASS_DISPLAY_NAMES,
            'device': str(self.device)
        }, path)
        print(f"模型已保存到: {path}")

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'history' in checkpoint:
            self.history = checkpoint['history']

        print(f"已加载模型: {path}")

# ============================ 主程序 ============================
def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 加载数据集
    print("\n加载数据集...")
    try:
        image_paths, labels = TomatoDatasetManager.load_dataset("TomatoDataset")
    except Exception as e:
        print(f"错误: {e}")
        print("请确保数据集路径正确，结构为: TomatoDataset/各类别文件夹/图片文件")
        return

    # 创建数据加载器（使用更大的图像尺寸）
    train_loader, val_loader, test_loader = TomatoDatasetManager.create_dataloaders(
        image_paths, labels, batch_size=16, img_size=256  # 增加图像尺寸
    )

    # 检查是否有预训练模型
    print("\n检查预训练模型...")
    model_files = ['tomato_model_best.pth', 'tomato_disease_best.pth']
    existing_models = [f for f in model_files if Path(f).exists()]

    if existing_models:
        print(f"找到预训练模型: {existing_models[0]}")
        load_existing = input("是否加载现有模型？(y/n, 默认y): ").strip().lower()
        load_existing = load_existing if load_existing else 'y'
        load_existing = load_existing != 'n'
    else:
        print("未找到预训练模型")
        load_existing = False

    # 初始化训练器
    print("\n初始化训练器...")
    trainer = TomatoDiseaseTrainer(num_classes=len(DISEASE_CLASSES))

    if load_existing and existing_models:
        trainer.load_model(existing_models[0])

    # 训练新模型
    train_new = input("\n是否训练新模型？(y/n, 默认y): ").strip().lower()
    train_new = train_new if train_new else 'y'
    train_new = train_new == 'y'

    if train_new:
        epochs = input("训练轮数 (默认40): ").strip()
        epochs = int(epochs) if epochs else 40
        trainer.train(train_loader, val_loader, epochs=epochs, model_name="tomato_model")

    # 评估模型
    print("\n评估模型性能...")
    test_accuracy = trainer.evaluate(test_loader)

    # 保存模型信息
    print("\n保存模型信息...")
    model_info = {
        'model_name': 'EnhancedTomatoModel',
        'num_classes': len(DISEASE_CLASSES),
        'classes': DISEASE_CLASSES,
        'display_classes': CLASS_DISPLAY_NAMES,
        'test_accuracy': test_accuracy,
        'model_parameters': sum(p.numel() for p in trainer.model.parameters()),
        'device': str(trainer.device),
        'training_history_length': len(trainer.history['train_loss']),
        'dataset_size': len(image_paths),
        'image_size': 256,
        'batch_size': 16
    }

    try:
        model_info_path = FIGURES_DIR / "model_info.json"
        with open(model_info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        print(f"模型信息已保存为 '{model_info_path}'")
    except Exception as e:
        print(f"保存模型信息时出错: {e}")

    # 示例预测
    print("\n示例预测...")
    # 从测试集中随机选择图片
    test_indices = [i for i, path in enumerate(image_paths)
                   if path in test_loader.dataset.image_paths]

    if test_indices:
        # 从每个类别中选一张图片
        selected_images = []
        selected_labels = []

        for class_idx in range(len(DISEASE_CLASSES)):
            class_test_indices = [i for i in test_indices if labels[i] == class_idx]
            if class_test_indices:
                selected_idx = np.random.choice(class_test_indices)
                selected_images.append(image_paths[selected_idx])
                selected_labels.append(labels[selected_idx])

        # 预测选中的图片（最多3张）
        for i, (img_path, true_label) in enumerate(zip(selected_images[:3], selected_labels[:3])):
            print(f"\n{'='*60}")
            print(f"预测示例 {i+1}: {Path(img_path).name}")
            print(f"真实类别: {CLASS_DISPLAY_NAMES[true_label]}")

            try:
                result = trainer.predict_single_image(img_path, show_result=True)

                print(f"预测类别: {result['display_name']}")
                print(f"置信度: {result['confidence']:.2%}")

                if result['class_id'] == true_label:
                    print("预测正确!")
                else:
                    print("预测错误")

                print("\n前3个预测结果:")
                for j, pred in enumerate(result['top3_predictions']):
                    print(f"  {j+1}. {pred['display_name']}: {pred['probability']:.2%}")

                # 绘制预测结果对比图
                plot_prediction_comparison(result, true_label, i+1)

            except Exception as e:
                print(f"预测时出错: {e}")

    # 绘制数据集分布图
    plot_dataset_distribution(labels)

    # 显示总结信息

    print(f"模型名称: EnhancedTomatoModel")
    print(f"模型参数量: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print(f"测试准确率: {test_accuracy:.2f}%")
    print(f"病害类别数: {len(DISEASE_CLASSES)}")
    print(f"数据集大小: {len(image_paths)} 张图片")
    print(f"图像尺寸: 256x256")
    print(f"训练设备: {trainer.device}")
    print(f"\n图表保存位置: {FIGURES_DIR.absolute()}")
    print("\n生成的文件:")
    print(f"  {FIGURES_DIR}/confusion_matrix.png - 混淆矩阵")
    print(f"  {FIGURES_DIR}/precision_recall_curve.png - 精度-召回率曲线")
    print(f"  {FIGURES_DIR}/class_accuracy_barchart.png - 各类别准确率条形图")
    print(f"  {FIGURES_DIR}/complete_training_history.png - 完整训练历史图")
    print(f"  {FIGURES_DIR}/model_info.json - 模型信息")
    print(f"  {FIGURES_DIR}/dataset_distribution.png - 数据集分布图")
    print(f"  {FIGURES_DIR}/prediction_*.png - 预测结果图")
    print(f"  {FIGURES_DIR}/training_*.png - 训练过程图")
    print("=" * 70)

def plot_prediction_comparison(result, true_label, example_num):
    """绘制预测结果对比图"""
    plt.figure(figsize=(10, 6))

    # 创建概率条形图
    classes = CLASS_DISPLAY_NAMES
    probabilities = result['probabilities']

    # 设置颜色
    colors = []
    for i in range(len(classes)):
        if i == result['class_id'] and i == true_label:
            colors.append('green')  # 预测正确
        elif i == result['class_id']:
            colors.append('red')    # 预测错误
        elif i == true_label:
            colors.append('orange') # 真实类别
        else:
            colors.append('lightgray')

    bars = plt.bar(range(len(classes)), probabilities, color=colors)

    plt.xlabel('病害类别', fontsize=12)
    plt.ylabel('概率', fontsize=12)
    plt.title(f'预测结果对比 (示例 {example_num})', fontsize=14, fontweight='bold')
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.ylim([0, 1.05])

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='预测正确'),
        Patch(facecolor='red', label='预测类别'),
        Patch(facecolor='orange', label='真实类别'),
        Patch(facecolor='lightgray', label='其他类别')
    ]
    plt.legend(handles=legend_elements)

    # 在柱状图上添加数值
    for bar, prob in zip(bars, probabilities):
        if prob > 0.05:  # 只显示概率大于5%的数值
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # 保存对比图
    comparison_path = FIGURES_DIR / f"prediction_comparison_example{example_num}.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"预测结果对比图已保存为 '{comparison_path}'")

def plot_dataset_distribution(labels):
    """绘制数据集分布图"""
    # 统计每个类别的数量
    class_counts = np.zeros(len(DISEASE_CLASSES), dtype=int)
    for label in labels:
        class_counts[label] += 1

    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 条形图
    colors = plt.cm.Set3(np.linspace(0, 1, len(DISEASE_CLASSES)))
    bars = axes[0].bar(range(len(DISEASE_CLASSES)), class_counts, color=colors)
    axes[0].set_xlabel('病害类别', fontsize=12)
    axes[0].set_ylabel('图片数量', fontsize=12)
    axes[0].set_title('数据集类别分布 (条形图)', fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(len(DISEASE_CLASSES)))
    axes[0].set_xticklabels(CLASS_DISPLAY_NAMES, rotation=45, ha='right')

    # 在柱状图上添加数值
    for bar, count in zip(bars, class_counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{count}', ha='center', va='bottom', fontsize=10)

    # 饼图
    wedges, texts, autotexts = axes[1].pie(class_counts, labels=CLASS_DISPLAY_NAMES,
                                           autopct='%1.1f%%', startangle=90,
                                           colors=colors, textprops={'fontsize': 9})
    axes[1].set_title('数据集类别分布 (饼图)', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # 保存数据集分布图
    distribution_path = FIGURES_DIR / "dataset_distribution.png"
    plt.savefig(distribution_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"数据集分布图已保存为 '{distribution_path}'")

    # 打印数据集统计信息
    print("\n数据集统计:")
    total_images = sum(class_counts)
    for i, (cls_name, count) in enumerate(zip(CLASS_DISPLAY_NAMES, class_counts)):
        percentage = 100.0 * count / total_images
        print(f"  {cls_name:25s}: {count:4d} 张 ({percentage:.1f}%)")

if __name__ == "__main__":
    # 检查PyTorch版本
    print(f"\nPyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")

    # 运行主程序
    main()
    print(f"所有图表已保存到: {FIGURES_DIR.absolute()}")