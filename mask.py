import numpy as np
import os
import sys
# 打开原始图像
patch_number = 1

def block_index_to_coordinates(block_index):
    if not 0 <= block_index < 64:
        raise ValueError("块索引必须在0到63之间")
    # 每个块的大小
    block_size = 16

    # 计算块的行号和列号
    row_index = block_index // 8  # 8个块为一行
    col_index = block_index % 8   # 8个块为一列

    # 计算块的左上角坐标
    top_left_x = col_index * block_size
    top_left_y = row_index * block_size

    return (top_left_x, top_left_y)

def mixup_region(image_array, x, y, block_size):
    print(image_array.shape)
    _, h, w = image_array.shape
    print(h, w)
    x_start = max(x - block_size, 0)
    x_end = min(x + 2*block_size, w)
    y_start = max(y - block_size, 0)
    y_end = min(y + 2*block_size, h)
    # print(x_start, x_end, y_start, y_end)
    # 将目标区域与其周围区域混合
    region = image_array[:, y:y+block_size, x:x+block_size]
    surrounding_regions = []
    for i in range(int(x_start / block_size), int(x_end / block_size)):
        for j in range(int(y_start / block_size), int(y_end / block_size)):
            print(i ,j)
            surrounding_region = image_array[:, j*block_size:(j+1)*block_size, i*block_size:(i+1)*block_size]
            surrounding_regions.append(surrounding_region)
    print(len(surrounding_regions))
    mixed_region = 0
    for i in range(len(surrounding_regions)):
        # print(surrounding_regions[i].shape)
        mixed_region += surrounding_regions[i] / len(surrounding_regions)

    # 将混合后的区域放回原图像
    image_array[:, y:y+block_size, x:x+block_size] = mixed_region.astype(np.uint8)
    return image_array

# 加载注意力映射和图片
attention_file_path = './data/all_attention_maps.npy'#833, 10, 1, 64, 64
attention_map = np.load(attention_file_path, allow_pickle=True)

image_file_path = './data/SEVIR_IR069_STORMEVENTS_2018_0101_0630.npy'# 833, 20, 128, 128
images = np.load(image_file_path, allow_pickle=True)

# 确保注意力映射和图片的数量相同
if len(attention_map) != len(images):
    print(f"注意力映射数量: {len(attention_map)}, 图片数量: {len(images)}")
    # sys.exit()
else:
    print("注意力映射和图片数量一致")

patch_number = 1
s = 0

processed_images = []

for batch_attention_maps, image_array in zip(attention_map, images):
    # 只处理图像的前半段
    image_array_front = image_array[:10, :, :]  # 提取前半段
    image_array_back = image_array[10:, :, :]   # 提取后半段
    # print(batch_attention_maps.shape)
    # 计算每个块的列和
    column_sums = np.sum(batch_attention_maps, axis=2)
    min_weight_columns = np.argpartition(column_sums, patch_number, axis=2)[:, :, :patch_number]
    min_weight_columns = min_weight_columns.reshape(batch_attention_maps.shape[0], -1)
    # print(min_weight_columns)

    # 遍历每个注意力映射
    for col_index_all in min_weight_columns:
        for col_index in col_index_all:
            x, y = block_index_to_coordinates(col_index)
            # print(y.dtype)
            image_array = mixup_region(image_array_front, x, y, 16)  # 使用块大小16进行mixup
    # 拼接前半段和后半段
    processed_image = np.concatenate((image_array, image_array_back), axis=0)
    # 保存处理后的图像到列表
    processed_images.append(processed_image)

# 将处理后的图像数组保存到新的 .npy 文件
processed_images = np.array(processed_images)
print(processed_images.shape)
np.save('./data/mask.npy', processed_images)
