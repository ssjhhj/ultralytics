import os
import cv2
import numpy as np
import random
import shutil
from pathlib import Path

class YOLODataAugmentor:
    def __init__(self, images_dir, labels_dir, output_dir):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_dir = Path(output_dir)
        
        # 创建输出目录
        (self.output_dir / 'images').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels').mkdir(parents=True, exist_ok=True)
        
    def copy_original_data(self):
        """复制原始数据到增强目录"""
        print("正在复制原始数据...")
        
        # 复制图片
        img_count = 0
        for img_file in self.images_dir.glob('*.*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                shutil.copy2(img_file, self.output_dir / 'images' / img_file.name)
                img_count += 1
        
        # 复制标签（确保每个图片都有对应标签文件，无标签则创建空文件）
        label_count = 0
        for img_file in self.images_dir.glob('*.*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                label_file = self.labels_dir / f"{img_file.stem}.txt"
                output_label_path = self.output_dir / 'labels' / f"{img_file.stem}.txt"
                
                # 若原始标签存在则复制，不存在则创建空文件
                if label_file.exists():
                    shutil.copy2(label_file, output_label_path)
                else:
                    open(output_label_path, 'w', encoding='utf-8').close()
                label_count += 1
        
        print(f"原始数据复制完成，图片: {img_count}, 标签: {label_count}")
        return img_count
    
    def adjust_brightness(self, image, factor):
        """增强1：亮度调整"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def adjust_contrast(self, image, factor):
        """增强2：对比度调整"""
        mean = np.mean(image, axis=(0, 1))
        adjusted = (image - mean) * factor + mean
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    def adjust_saturation(self, image, factor):
        """增强3：饱和度调整"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def add_gaussian_noise(self, image, mean=0, sigma=15):
        """增强4：轻微高斯噪声"""
        noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
        noisy_image = image.astype(np.float32) + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    def apply_gaussian_blur(self, image, kernel_size=3):
        """增强5：轻度高斯模糊"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def random_occlusion(self, image, num_patches=2, patch_size=0.05):
        """增强6：随机遮挡图像的小区域"""
        h, w = image.shape[:2]
        patch_h, patch_w = int(h * patch_size), int(w * patch_size)
        
        # 确保patch尺寸至少为1
        patch_h = max(1, patch_h)
        patch_w = max(1, patch_w)
        
        augmented = image.copy()
        for _ in range(num_patches):
            x1 = random.randint(0, w - patch_w)
            y1 = random.randint(0, h - patch_h)
            # 用黑色块遮挡
            augmented[y1:y1+patch_h, x1:x1+patch_w] = 0
        
        return augmented
    
    def grayscale(self, image):
        """增强7：全局灰度化"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    def add_salt_pepper_noise(self, image, salt_prob=0.005, pepper_prob=0.005):
        """增强8：轻微椒盐噪声"""
        noisy = image.copy()
        # 盐噪声（白点）
        salt_mask = np.random.random(image.shape[:2]) < salt_prob
        noisy[salt_mask] = 255
        # 椒噪声（黑点）
        pepper_mask = np.random.random(image.shape[:2]) < pepper_prob
        noisy[pepper_mask] = 0
        return noisy
    
    def random_brightness_patches(self, image, num_patches=2, patch_size=0.1):
        """增强9：随机亮度块"""
        h, w = image.shape[:2]
        patch_h, patch_w = int(h * patch_size), int(w * patch_size)
        
        # 确保patch尺寸至少为1
        patch_h = max(1, patch_h)
        patch_w = max(1, patch_w)
        
        augmented = image.copy()
        for _ in range(num_patches):
            x1 = random.randint(0, w - patch_w)
            y1 = random.randint(0, h - patch_h)
            
            # 随机调整亮度
            brightness_factor = random.uniform(0.5, 1.5)
            patch = augmented[y1:y1+patch_h, x1:x1+patch_w]
            hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            hsv_patch = hsv_patch.astype(np.float32)
            hsv_patch[:, :, 2] = hsv_patch[:, :, 2] * brightness_factor
            hsv_patch[:, :, 2] = np.clip(hsv_patch[:, :, 2], 0, 255)
            hsv_patch = hsv_patch.astype(np.uint8)
            augmented[y1:y1+patch_h, x1:x1+patch_w] = cv2.cvtColor(hsv_patch, cv2.COLOR_HSV2BGR)
        
        return augmented
    
    def apply_augmentations(self):
        """应用所有增强方法（已移除翻转，标签直接复制）"""
        print("开始数据增强...")
        
        # 首先复制原始数据
        original_count = self.copy_original_data()
        
        # 获取所有图片文件
        image_files = list(self.images_dir.glob('*.*'))
        image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        total_augmentations = 0
        
        for img_file in image_files:
            print(f"\n处理图片: {img_file.name}")
            
            # 读取图片
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"  无法读取图片: {img_file}")
                continue
            
            # 检查原始标签是否存在（无标签则停止增强）
            original_label_path = self.labels_dir / f"{img_file.stem}.txt"
            if not original_label_path.exists() or original_label_path.stat().st_size == 0:
                print(f"  无有效标签，停止该图片的增强")
                continue
            
            # 定义增强方法列表（共10种，标签直接复制）
            augmentations = [
                ("brightness", lambda img: self.adjust_brightness(img, 1.1)),
                ("darkness", lambda img: self.adjust_brightness(img, 0.9)),
                ("contrast", lambda img: self.adjust_contrast(img, 1.2)),
                ("saturation", lambda img: self.adjust_saturation(img, 1.2)),
                ("gaussian_noise", lambda img: self.add_gaussian_noise(img)),
                ("gaussian_blur", lambda img: self.apply_gaussian_blur(img, 5)),
                ("occlusion", lambda img: self.random_occlusion(img)),
                ("grayscale", lambda img: self.grayscale(img)),
                ("salt_pepper", lambda img: self.add_salt_pepper_noise(img)),
                ("brightness_patches", lambda img: self.random_brightness_patches(img)),
            ]
            
            # 应用所有增强（标签直接复制并重命名）
            for aug_name, aug_func in augmentations:
                try:
                    augmented_image = aug_func(image)
                    
                    # 保存增强后的图片
                    output_img_name = f"{img_file.stem}_{aug_name}{img_file.suffix}"
                    output_img_path = self.output_dir / 'images' / output_img_name
                    
                    # 检查文件是否已存在
                    if output_img_path.exists():
                        print(f"  跳过 {output_img_name} - 文件已存在")
                        continue
                    
                    success = cv2.imwrite(str(output_img_path), augmented_image)
                    
                    if success:
                        # 直接复制原始标签并修改文件名
                        output_label_name = f"{img_file.stem}_{aug_name}.txt"
                        output_label_path = self.output_dir / 'labels' / output_label_name
                        shutil.copy2(original_label_path, output_label_path)
                        
                        total_augmentations += 1
                        print(f"  ✓ 生成增强: {aug_name}")
                    else:
                        print(f"  ✗ 保存图片失败: {aug_name}")
                    
                except Exception as e:
                    print(f"  ✗ 增强 {aug_name} 失败: {e}")
        
        print(f"\n数据增强完成！")
        print(f"原始图片数量: {original_count}")
        print(f"生成的增强图片数量: {total_augmentations}")
        print(f"总图片数量: {original_count + total_augmentations}")
        print(f"输出目录: {self.output_dir}")
        
        # 验证输出
        self.verify_output()

    def verify_output(self):
        """验证输出文件"""
        print("\n验证输出文件...")
        aug_images = list((self.output_dir / 'images').glob('*.*'))
        aug_labels = list((self.output_dir / 'labels').glob('*.txt'))
        
        print(f"增强后图片数量: {len(aug_images)}")
        print(f"增强后标签数量: {len(aug_labels)}")
        
        # 检查图片和标签是否匹配
        image_names = {img.stem for img in aug_images}
        label_names = {label.stem for label in aug_labels}
        
        missing_labels = image_names - label_names
        missing_images = label_names - image_names
        
        if missing_labels:
            print(f"警告: {len(missing_labels)} 张图片没有对应的标签文件")
        if missing_images:
            print(f"警告: {len(missing_images)} 个标签文件没有对应的图片")
        
        if not missing_labels and not missing_images:
            print("✓ 所有图片和标签文件都匹配")

def main():
    # 配置路径
    images_dir = r"D:\Desktop\XLWD\dataset\insulator_and_line\2025121_process\insulator_line\images"
    labels_dir = r"D:\Desktop\XLWD\dataset\insulator_and_line\2025121_process\insulator_line\labels"
    output_dir = r"D:\Desktop\XLWD\dataset\insulator_and_line\2025121_process\insulator_line\aug"
    
    # 创建增强器并执行增强
    augmentor = YOLODataAugmentor(images_dir, labels_dir, output_dir)
    augmentor.apply_augmentations()

if __name__ == "__main__":
    main()