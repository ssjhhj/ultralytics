import os
import random
import shutil
import argparse
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from scipy.ndimage import convolve


class DataAugmentationOnDetection:
    """数据增强工具类（针对目标检测任务，支持四边形边界框）"""
    
    def random_flip_horizon(self, img, boxes, p=1.0):
        """水平翻转图像和边界框（针对归一化坐标的正确实现）"""
        if np.random.random() < p and len(boxes) > 0:
            # 1. 翻转图像
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
            # 2. 翻转x坐标（归一化坐标逻辑：新x = 1 - 原x）
            boxes[:, [1, 3, 5, 7]] = 1.0 - boxes[:, [1, 3, 5, 7]]
            
            # 3. 调整四边形顶点顺序（镜像翻转后顶点顺序需反转）
            # 原始顶点顺序：(x1,y1)、(x2,y2)、(x3,y3)、(x4,y4)
            # 翻转后应变为：(x1',y1)、(x4',y4)、(x3',y3)、(x2',y2)
            # 交换第2个点和第4个点的坐标（索引对应关系）
            boxes[:, [3,4,7,8]] = boxes[:, [7,8,3,4]]  # 修正顶点顺序
            
        return img, boxes

    def random_flip_vertical(self, img, boxes, p=1.0):
        """垂直翻转（上下翻转）图像和边界框（针对归一化坐标）"""
        if np.random.random() < p:
            # 翻转图像（上下颠倒）
            img_flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
            
            # 调整边界框坐标（只修改y坐标）
            if len(boxes) > 0:
                # 四边形框格式：[class, x1, y1, x2, y2, x3, y3, x4, y4]
                # 垂直翻转后，所有y坐标变为 1 - 原y坐标
                boxes[:, [2, 4, 6, 8]] = 1.0 - boxes[:, [2, 4, 6, 8]]
            
            return img_flipped, boxes
        else:
            # 不翻转时返回原图和原框
            return img, boxes

    def center_crop(self, img, boxes, crop_size):
        """中心裁剪（针对归一化坐标的正确转换）"""
        old_w, old_h = img.size  # 原始图像宽高
        # 计算裁剪区域（确保不超出原图范围）
        crop_size = min(crop_size, old_w, old_h)  # 防止裁剪尺寸大于原图
        left = (old_w - crop_size) // 2
        top = (old_h - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        
        # 裁剪图像
        img_cropped = img.crop((left, top, right, bottom))
        new_w, new_h = img_cropped.size  # 裁剪后图像宽高（应为crop_size）
        
        new_boxes = []
        if len(boxes) > 0:
            for box in boxes:
                cls = box[0]
                # 提取4个点的归一化坐标 (x1,y1,x2,y2,x3,y3,x4,y4)
                norm_coords = box[1:].reshape(-1, 2)  # 形状：[4,2]
                
                # 步骤1：归一化坐标 → 原始图像绝对像素坐标
                abs_coords = []
                for (x_norm, y_norm) in norm_coords:
                    x_abs = x_norm * old_w  # 还原x绝对坐标
                    y_abs = y_norm * old_h  # 还原y绝对坐标
                    abs_coords.append((x_abs, y_abs))
                
                # 步骤2：调整坐标到裁剪后的图像（减去裁剪区域左上角偏移）
                crop_coords = []
                valid = False  # 标记是否有坐标在裁剪区域内
                for (x_abs, y_abs) in abs_coords:
                    x_crop = x_abs - left
                    y_crop = y_abs - top
                    # 检查点是否在裁剪区域内（0 ≤ x ≤ new_w，0 ≤ y ≤ new_h）
                    if 0 <= x_crop <= new_w and 0 <= y_crop <= new_h:
                        valid = True
                    crop_coords.append((x_crop, y_crop))
                
                # 若目标完全在裁剪区域外，跳过该框
                if not valid:
                    continue
                
                # 步骤3：裁剪后绝对坐标 → 裁剪后图像的归一化坐标
                new_norm_coords = []
                for (x_crop, y_crop) in crop_coords:
                    x_new = x_crop / new_w  # 相对于裁剪后宽度归一化
                    y_new = y_crop / new_h  # 相对于裁剪后高度归一化
                    # 限制坐标在[0,1]范围内（防止边缘点因浮点误差超出）
                    x_new = max(0.0, min(1.0, x_new))
                    y_new = max(0.0, min(1.0, y_new))
                    new_norm_coords.extend([x_new, y_new])
                
                # 组合类别和新坐标，添加到结果
                new_box = [cls] + new_norm_coords
                new_boxes.append(new_box)
        
        # 转换为tensor返回（保持与其他方法一致的格式）
        return img_cropped, torch.tensor(new_boxes) if new_boxes else boxes

    def random_bright(self, img, lower=0.5, upper=1.5, p=1.0):
        """随机调整亮度"""
        if np.random.random() < p:
            alpha = np.random.uniform(lower, upper)
            img = img * alpha
            img = img.clamp(min=0, max=1.0)
        return img

    def random_contrast(self, img, lower=0.5, upper=1.5, p=1.0):
        """随机调整对比度"""
        if np.random.random() < p:
            alpha = np.random.uniform(lower, upper)
            mean = img.mean()
            img = (img - mean) * alpha + mean
            img = img.clamp(min=0, max=1.0)
        return img

    def random_saturation(self, img, lower=0.5, upper=1.5, p=1.0):
        """随机调整饱和度（针对RGB图像的G通道）"""
        if np.random.random() < p:
            alpha = np.random.uniform(lower, upper)
            img[1] = img[1] * alpha  # 饱和度在RGB的G通道（针对PIL的转换）
            img[1] = img[1].clamp(min=0, max=1.0)
        return img

    def add_gaussian_noise(self, img, mean=0, std=0.1, p=1.0):
        """添加高斯噪声（修正方法名拼写错误）"""
        if np.random.random() < p:
            noise = torch.normal(mean, std, img.shape)
            img += noise
            img = img.clamp(min=0, max=1.0)
        return img

    def add_salt_noise(self, img, p=1.0):
        """添加盐噪声"""
        if np.random.random() < p:
            noise = torch.rand(img.shape)
            alpha = np.random.random() / 5 + 0.7
            img[noise > alpha] = 1.0
        return img

    def add_pepper_noise(self, img, p=1.0):
        """添加椒噪声"""
        if np.random.random() < p:
            noise = torch.rand(img.shape)
            alpha = np.random.random() / 5 + 0.7
            img[noise > alpha] = 0
        return img
    
    def random_scale(self, img, boxes, scale_range=(0.5, 1.5)):
        """随机缩放（按比例缩放图像，归一化坐标无需调整）"""
        scale = np.random.uniform(scale_range[0], scale_range[1])
        old_w, old_h = img.size
        new_w = int(old_w * scale)
        new_h = int(old_h * scale)
        
        # 调整图像尺寸
        img_scaled = img.resize((new_w, new_h), Image.BILINEAR)
        
        # 关键修正：归一化坐标不随缩放改变（相对位置不变）
        # 原代码的坐标缩放逻辑删除，直接返回原boxes
        return img_scaled, boxes

    def random_erase_snow(self, img, boxes, p=1.0, snow_count_range=(50, 200), snow_size_range=(5, 30)):
        """随机雪花效果擦除（生成多个小色块模拟雪花遮挡）"""
        if np.random.random() < p:
            w, h = img.size
            img_np = np.array(img).copy()
            h_img, w_img = img_np.shape[:2]  # 获取图像的高和宽（注意numpy的shape是[h, w, c]）
            
            # 随机生成雪花数量
            snow_count = np.random.randint(snow_count_range[0], snow_count_range[1] + 1)
            
            for _ in range(snow_count):
                # 随机雪花尺寸
                snow_size = np.random.randint(snow_size_range[0], snow_size_range[1] + 1)
                snow_size = min(snow_size, w, h)  # 防止雪花尺寸超出图像
                
                # 随机雪花位置（确保在图像范围内）
                x1 = np.random.randint(0, max(1, w - snow_size))
                y1 = np.random.randint(0, max(1, h - snow_size))
                x2 = x1 + snow_size
                y2 = y1 + snow_size
                
                # 雪花颜色倾向于白色（高亮度），取值范围[180, 255]
                color = np.random.randint(180, 256, 3, dtype=np.uint8)
                
                # 绘制雪花色块
                img_np[y1:y2, x1:x2] = color
            
            # 转换回PIL图像
            img = Image.fromarray(img_np)
        
        return img, boxes

    def illumination_distortion(self, img, boxes, p=1.0, brightness_range=(0.3, 1.7), contrast_range=(0.3, 1.7)):
        """光照畸变（组合亮度和对比度调整模拟光照变化）"""
        if np.random.random() < p:
            to_tensor = transforms.ToTensor()
            to_image = transforms.ToPILImage()
            img_tensor = to_tensor(img)
            
            # 随机亮度调整
            brightness = np.random.uniform(brightness_range[0], brightness_range[1])
            img_tensor = img_tensor * brightness
            
            # 随机对比度调整
            contrast = np.random.uniform(contrast_range[0], contrast_range[1])
            mean = img_tensor.mean()
            img_tensor = (img_tensor - mean) * contrast + mean
            
            # 截断到有效范围
            img_tensor = torch.clamp(img_tensor, 0, 1.0)
            img = to_image(img_tensor)
        
        return img, boxes

    def gaussian_blur(self, img, boxes, p=1.0, kernel_size_range=(3, 7)):
        """高斯模糊"""
        if np.random.random() < p:
            # 随机选择奇数核大小
            kernel_sizes = list(range(kernel_size_range[0], kernel_size_range[1]+1, 2))
            if kernel_sizes:  # 确保有可用的核大小
                kernel_size = np.random.choice(kernel_sizes)
                blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))
                img = blur(img)
        return img, boxes

    def motion_blur(self, img, boxes, p=1.0, kernel_size=9, direction_range=(-np.pi/4, np.pi/4)):
        """运动模糊（沿特定方向模糊）"""
        if np.random.random() < p and kernel_size > 0:
            img_np = np.array(img)
            h, w = img_np.shape[:2]
            
            # 随机运动方向
            direction = np.random.uniform(direction_range[0], direction_range[1])
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            
            # 生成运动模糊核（沿方向的直线）
            center = kernel_size // 2
            for i in range(kernel_size):
                x = int(center + i * np.cos(direction))
                y = int(center + i * np.sin(direction))
                if 0 <= x < kernel_size and 0 <= y < kernel_size:
                    kernel[y, x] = 1
            
            # 归一化核（避免除零错误）
            kernel_sum = kernel.sum()
            if kernel_sum > 0:
                kernel /= kernel_sum
            else:
                kernel[center, center] = 1.0  # 退化为本征核
            
            # 应用卷积
            if len(img_np.shape) == 3:  # 彩色图像
                for c in range(3):
                    img_np[:, :, c] = convolve(img_np[:, :, c], kernel)
            else:  # 灰度图像
                img_np = convolve(img_np, kernel)
            
            # 确保像素值在有效范围
            img_np = np.clip(img_np, 0, 255)
            img = Image.fromarray(np.uint8(img_np))
        
        return img, boxes

    def to_grayscale(self, img, boxes, p=1.0):
        """灰度化"""
        if np.random.random() < p:
            img = img.convert('L').convert('RGB')  # 转为灰度后保持3通道
        return img, boxes

    def channel_shift(self, img, boxes, p=1.0, shift_range=(-30, 30)):
        """通道偏移（对RGB通道分别添加偏移）"""
        if np.random.random() < p:
            img_np = np.array(img, dtype=np.int16)  # 防止溢出
            
            # 对每个通道添加随机偏移
            for c in range(3):
                shift = np.random.randint(shift_range[0], shift_range[1]+1)
                img_np[:, :, c] = np.clip(img_np[:, :, c] + shift, 0, 255)
            
            img = Image.fromarray(np.uint8(img_np))
        return img, boxes


def get_image_list(image_path):
    """根据图片文件，查找所有图片并返回列表"""
    files_list = []
    for root, sub_dirs, files in os.walk(image_path):
        for special_file in files:
            if special_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # 存储绝对路径避免后续路径问题
                files_list.append(os.path.join(root, special_file))
    # 去重
    return list(set(files_list))


def get_label_file(label_path, image_name):
    """根据图片信息，查找对应的label（四边形格式）"""
    # 从图像路径中提取文件名（处理绝对路径情况）
    base_name = os.path.splitext(os.path.basename(image_name))[0]
    fname = os.path.join(label_path, base_name + ".txt")
    data2 = []
    if not os.path.exists(fname) or os.path.getsize(fname) == 0:
        return data2
    
    with open(fname, 'r', encoding='utf-8') as infile:
        for line in infile:
            data_line = line.strip("\n").split()
            if len(data_line) != 9:  # 1个类别 + 4个点（8个坐标）
                continue  # 跳过格式错误的行
            try:
                # 转换为 [class, x1, y1, x2, y2, x3, y3, x4, y4]
                parsed = [int(data_line[0])] + [float(x) for x in data_line[1:]]
                data2.append(parsed)
            except ValueError:
                continue  # 跳过数值错误的行
    return data2


def save_Yolo(img, boxes, save_path, prefix, image_name):
    """保存增强后的图像和标签（四边形格式）"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    images_dir = os.path.join(save_path, "images")
    labels_dir = os.path.join(save_path, "labels")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    try:
        # 处理图像名（避免路径分隔符问题）
        img_basename = os.path.basename(image_name)
        # 保存图像
        img.save(os.path.join(images_dir, prefix + img_basename))
        
        # 保存标签
        label_basename = prefix + os.path.splitext(img_basename)[0] + ".txt"
        with open(os.path.join(labels_dir, label_basename), 'w', encoding="utf-8") as f:
            if len(boxes) > 0:
                for data in boxes:
                    # 格式：class x1 y1 x2 y2 x3 y3 x4 y4
                    parts = [str(int(data[0]))]  # 类别索引
                    parts.extend([f"{x:.6f}" for x in data[1:]])  # 坐标保留6位小数
                    f.write(" ".join(parts) + '\n')
    except Exception as e:
        print(f"ERROR: {image_name} 保存失败。错误: {e}")


def run_augmentation(image_path, label_path, save_path):
    """运行数据增强"""
    print("开始数据增强...")
    image_list = get_image_list(image_path)
    print(f"找到 {len(image_list)} 张图片")
    
    DAD = DataAugmentationOnDetection()
    
    for i, img_path in enumerate(image_list):
        image_name = os.path.basename(img_path)
        print(f"处理进度: {i+1}/{len(image_list)} - {image_name}")
        
        try:
            img = Image.open(img_path).convert('RGB')  # 确保图像为RGB格式
            boxes = get_label_file(label_path, image_name)
            boxes = torch.tensor(boxes, dtype=torch.float32)  # 明确 dtype

            # 新增：保存原始图像和标签（前缀用"ori_"区分）
            save_Yolo(img.copy(), boxes.clone(), save_path, prefix="ori_", image_name=image_name)
            
            # 1. 水平翻转
            t_img, t_boxes = DAD.random_flip_horizon(img.copy(), boxes.clone())
            save_Yolo(t_img, t_boxes, save_path, prefix="fh_", image_name=image_name)
            
            # 2. 垂直翻转
            t_img, t_boxes = DAD.random_flip_vertical(img.copy(), boxes.clone())
            save_Yolo(t_img, t_boxes, save_path, prefix="fv_", image_name=image_name)
            
            # 3. 中心裁剪
            t_img, t_boxes = DAD.center_crop(img.copy(), boxes.clone(), 1024)
            save_Yolo(t_img, t_boxes, save_path, prefix="cc_", image_name=image_name)
            
            # 4. 亮度变化
            to_tensor = transforms.ToTensor()
            to_image = transforms.ToPILImage()
            img_tensor = to_tensor(img)
            
            t_img = DAD.random_bright(img_tensor.clone())
            save_Yolo(to_image(t_img), boxes, save_path, prefix="rb_", image_name=image_name)
            
            # 5. 对比度变化
            t_img = DAD.random_contrast(img_tensor.clone())
            save_Yolo(to_image(t_img), boxes, save_path, prefix="rc_", image_name=image_name)
            
            # 6. 饱和度变化
            t_img = DAD.random_saturation(img_tensor.clone())
            save_Yolo(to_image(t_img), boxes, save_path, prefix="rs_", image_name=image_name)
            
            # 7. 高斯噪声（修正方法名调用）
            t_img = DAD.add_gaussian_noise(img_tensor.clone())
            save_Yolo(to_image(t_img), boxes, save_path, prefix="gn_", image_name=image_name)
            
            # 8. 盐噪声
            t_img = DAD.add_salt_noise(img_tensor.clone())
            save_Yolo(to_image(t_img), boxes, save_path, prefix="sn_", image_name=image_name)
            
            # 9. 椒噪声
            t_img = DAD.add_pepper_noise(img_tensor.clone())
            save_Yolo(to_image(t_img), boxes, save_path, prefix="pn_", image_name=image_name)

            # 10. 随机缩放（修改前缀避免与饱和度冲突）
            t_img, t_boxes = DAD.random_scale(img.copy(), boxes.clone())
            save_Yolo(t_img, t_boxes, save_path, prefix="rsc_", image_name=image_name)

            # 11. 随机雪花擦除
            t_img, t_boxes = DAD.random_erase_snow(img.copy(), boxes.clone())
            save_Yolo(t_img, t_boxes, save_path, prefix="re_", image_name=image_name)

            # 12. 光照畸变
            t_img, t_boxes = DAD.illumination_distortion(img.copy(), boxes.clone())
            save_Yolo(t_img, t_boxes, save_path, prefix="id_", image_name=image_name)

            # 13. 高斯模糊
            t_img, t_boxes = DAD.gaussian_blur(img.copy(), boxes.clone())
            save_Yolo(t_img, t_boxes, save_path, prefix="gb_", image_name=image_name)

            # 14. 运动模糊
            t_img, t_boxes = DAD.motion_blur(img.copy(), boxes.clone())
            save_Yolo(t_img, t_boxes, save_path, prefix="mb_", image_name=image_name)

            # 15. 灰度化
            t_img, t_boxes = DAD.to_grayscale(img.copy(), boxes.clone())
            save_Yolo(t_img, t_boxes, save_path, prefix="gs_", image_name=image_name)

            # 16. 通道偏移
            t_img, t_boxes = DAD.channel_shift(img.copy(), boxes.clone())
            save_Yolo(t_img, t_boxes, save_path, prefix="cs_", image_name=image_name)
            
        except Exception as e:
            print(f"处理 {image_name} 时出错: {e}")
            continue
    
    print("数据增强完成！")


def split_dataset(aug_path, save_dir, is_original_data=False):
    """划分数据集为训练集、验证集和测试集"""
    print("开始划分数据集...")
    
    # 创建文件夹结构
    images_dir = os.path.join(save_dir, 'images')
    labels_dir = os.path.join(save_dir, 'labels')
    
    img_train_path = os.path.join(images_dir, 'train')
    img_val_path = os.path.join(images_dir, 'val')
    img_test_path = os.path.join(images_dir, 'test')
    
    label_train_path = os.path.join(labels_dir, 'train')
    label_val_path = os.path.join(labels_dir, 'val')
    label_test_path = os.path.join(labels_dir, 'test')
    
    # 创建目录（确保父目录存在）
    for path in [img_train_path, img_val_path, img_test_path, 
                 label_train_path, label_val_path, label_test_path]:
        os.makedirs(path, exist_ok=True)
    
    # 数据集划分比例：训练集80%，验证集15%，测试集5%
    train_percent = 0.8
    val_percent = 0.15
    test_percent = 0.05
    
    if is_original_data:
        aug_labels_dir = aug_path
        aug_images_dir = os.path.join(os.path.dirname(aug_path), 'images')
    else:
        aug_labels_dir = os.path.join(aug_path, 'labels')
        aug_images_dir = os.path.join(aug_path, 'images')
    
    # 验证标签目录存在
    if not os.path.exists(aug_labels_dir):
        print(f"错误：标签目录不存在: {aug_labels_dir}")
        return
        
    total_txt = [f for f in os.listdir(aug_labels_dir) if f.endswith('.txt')]
    num_txt = len(total_txt)
    
    if num_txt == 0:
        print("警告：没有找到标签文件！")
        return
    
    print(f"总共找到 {num_txt} 个标签文件")
    
    # 计算各集合数量（确保至少有一个样本）
    num_train = max(1, int(num_txt * train_percent))
    num_val = max(1, int(num_txt * val_percent))
    num_test = max(1, num_txt - num_train - num_val)
    
    # 随机打乱（设置种子确保可复现性）
    random.seed(42)
    random.shuffle(total_txt)
    
    # 划分数据集
    train_files = total_txt[:num_train]
    val_files = total_txt[num_train:num_train + num_val]
    test_files = total_txt[num_train + num_val:]
    
    print(f"训练集: {len(train_files)}, 验证集: {len(val_files)}, 测试集: {len(test_files)}")
    
    # 复制文件
    def copy_files(file_list, img_dst, label_dst):
        for txt_file in file_list:
            base_name = os.path.splitext(txt_file)[0]
            
            # 复制标签
            src_label = os.path.join(aug_labels_dir, txt_file)
            if os.path.exists(src_label):
                dst_label = os.path.join(label_dst, txt_file)
                shutil.copy2(src_label, dst_label)
            else:
                print(f"警告：标签文件不存在: {src_label}")
                continue
            
            # 复制图像
            found = False
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
                img_file = base_name + ext
                src_img = os.path.join(aug_images_dir, img_file)
                if os.path.exists(src_img):
                    dst_img = os.path.join(img_dst, img_file)
                    shutil.copy2(src_img, dst_img)
                    found = True
                    break
            if not found:
                print(f"警告：未找到 {base_name} 对应的图像文件")
    
    # 复制训练集
    copy_files(train_files, img_train_path, label_train_path)
    print(f"训练集文件复制完成: {len(os.listdir(img_train_path))} 张图片, {len(os.listdir(label_train_path))} 个标签")
    
    # 复制验证集
    copy_files(val_files, img_val_path, label_val_path)
    print(f"验证集文件复制完成: {len(os.listdir(img_val_path))} 张图片, {len(os.listdir(label_val_path))} 个标签")
    
    # 复制测试集
    copy_files(test_files, img_test_path, label_test_path)
    print(f"测试集文件复制完成: {len(os.listdir(img_test_path))} 张图片, {len(os.listdir(label_test_path))} 个标签")
    
    print("数据集划分完成！")


def main(image_path, label_path, augmentation_mode=1, split_dataset_flag=1):
    """主函数：执行数据增强和可选的数据集划分"""
    print("=" * 50)
    print("数据增强和数据集划分一体化脚本（支持四边形框）")
    print("=" * 50)
    
    parent_dir = os.path.dirname(label_path)
    print(f"上级目录: {parent_dir}")
    
    aug_path = os.path.join(parent_dir, 'aug')
    print(f"数据增强结果保存路径: {aug_path}")
    
    end_path = os.path.join(parent_dir, 'end')
    print(f"划分后数据集保存路径: {end_path}")
    
    # 检查输入路径
    if not os.path.exists(image_path):
        print(f"错误：图像路径不存在: {image_path}")
        return
    
    if not os.path.exists(label_path):
        print(f"错误：标签路径不存在: {label_path}")
        return
    
    if augmentation_mode == 0:
        print("\n跳过数据增强，直接处理原始数据...")
        aug_path = label_path
        print(f"使用原始数据路径: {aug_path}")
    else:
        print("\n第一步：执行数据增强...")
        run_augmentation(image_path, label_path, aug_path)
    
    # 根据参数决定是否进行数据集划分
    if split_dataset_flag == 1:
        print("\n第二步：划分数据集...")
        split_dataset(aug_path, end_path, is_original_data=(augmentation_mode == 0))
    else:
        print("\n跳过数据集划分步骤")
    
    print("\n" + "=" * 50)
    print("所有操作完成！")
    if augmentation_mode != 0:
        print(f"数据增强结果保存在: {aug_path}")
    if split_dataset_flag == 1:
        print(f"划分后的数据集保存在: {end_path}")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='数据增强和数据集划分一体化脚本（支持四边形边界框）')
    parser.add_argument('--image-path', type=str, 
                       default=r'D:\Desktop\XLWD\project\work11\粒子分流yolo界面开发\10微米粒子\images',
                       help='图像文件夹路径')
    parser.add_argument('--label-path', type=str,
                       default=r'D:\Desktop\XLWD\project\work11\粒子分流yolo界面开发\10微米粒子\labels',
                       help='标签文件夹路径')
    parser.add_argument('--augmentation-mode', type=int, default=1, choices=[0, 1],
                       help='数据增强模式：1=进行数据增强，0=跳过数据增强')
    parser.add_argument('--split-dataset', type=int, default=1, choices=[0, 1],
                       help='是否划分数据集：1=划分（默认），0=不划分')
    
    args = parser.parse_args()
    main(args.image_path, args.label_path, args.augmentation_mode, args.split_dataset)