import cv2
import glob
import numpy as np
import os
import shutil

def npz():
    #原图像路径
    path = r'/home/yjr/Swin-Unet/data/Synapse/images/*.png'
    #项目中存放训练所用的npz文件路径
    path2 = r'/home/yjr/Swin-Unet/data/Synapse/train_npz/'
    path3 = r'/home/yjr/Swin-Unet/data/Synapse/test'
    
    for i, img_path in enumerate(glob.glob(path)):
        # 读入图像
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 读入标签
        label_path = img_path.replace('images', 'labels')
        label = cv2.imread(label_path, flags = cv2.IMREAD_GRAYSCALE)
        print("Image shape:", image.shape)
        print("Label shape:", label.shape)
        label[label != 255] = 1  # 将非白色像素设置为1
        label[label == 255] = 0  # 将白色像素设置为0
    
        # 获取图片文件名（不带路径和扩展名）
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # count_label_255 = np.sum(label == 255)#白色
        # count_label_not_255 = np.sum(label != 255)#黑色
        # 保存npz
        np.savez(os.path.join(path2, f'{img_name}.npz'), image=image, label=label)
        print('------------', i)
        #cv2.imwrite(os.path.join(path3, f'{img_name}_label.png'), label) #可视化
        # print(f"Number of pixels with label 255: {count_label_255}")
        # print(f"Number of pixels with label not 255: {count_label_not_255}")
    
    # for i,img_path in enumerate(glob.glob(path)):
    # 	#读入图像
    #     image = cv2.imread(img_path)
    #     image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #     #读入标签
    #     label_path = img_path.replace('images','labels')
    #     label = cv2.imread(label_path,flags=0)
    #     #将非目标像素设置为0
    #     label[label!=255]=0
    #     #将目标像素设置为1
    #     label[label==255]=1
	# 	#保存npz
    #     np.savez(path2+str(i),image=image,label=label)
    #     print('------------',i)

    # 加载npz文件
    # data = np.load(r'G:\dataset\Unet\Swin-Unet-ori\data\Synapse\train_npz\0.npz', allow_pickle=True)
    # image, label = data['image'], data['label']
    print('ok')

npz()

# def count():
#     #原图像路径
#     path = r'/home/yjr/Swin-Unet/data/Synapse/images/*.png'

#     count_label_255 = 0
#     count_label_not_255 = 0

#     for i, img_path in enumerate(glob.glob(path)):
#         # 读入标签
#         label_path = img_path.replace('images', 'labels')
#         label = cv2.imread(label_path, flags=cv2.IMREAD_GRAYSCALE)
#         if(i==0):
#         # 统计像素个数
#             count_label_255 = np.sum(label == 255)#白色
#             count_label_not_255 = np.sum(label != 255)#黑色

#     print(f"Number of pixels with label 255: {count_label_255}")
#     print(f"Number of pixels with label not 255: {count_label_not_255}")

# count()

# def convert_to_npz(input_path, output_path, array_name='image'):
#     # 读取灰度图像
#     image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

#     # 创建包含图像数据的 NumPy 数组
#     image_array = np.array(image)

#     # 保存为 NPZ 文件
#     np.savez(output_path, **{array_name: image_array})
#     print("convert to npz")

# # 输入灰度图像路径
# input_image_path = '/home/yjr/Swin-Unet/data/Synapse/images/1.png'

# # 输出 NPZ 文件路径
# output_npz_path = '/home/yjr/Swin-Unet/data/Synapse/test_vol_h5/1.npz'

# # 调用函数进行转换
# convert_to_npz(input_image_path, output_npz_path)




#--------------灰度图转成npz格式----------
# def convert_to_npz(input_folder, output_folder):
#     # 确保输出文件夹存在
#     os.makedirs(output_folder, exist_ok=True)

#     # 获取输入文件夹中的所有PNG文件
#     input_path = os.path.join(input_folder, '*.png')

#     for i, img_path in enumerate(glob.glob(input_path)):
#         # 读取图像
#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # 构建对应的标签路径
#         label_path = img_path.replace('images', 'labels') 
#         # 判断标签文件是否存在
#         if os.path.exists(label_path):
#             # 读取标签
#             label = cv2.imread(label_path, flags=0)

#             # 将标签转换为二进制（0或1）255白色 0黑色流线
#             label[label != 255] = 1
#             label[label == 255] = 0

#             # 提取原始图片的基本名称（不带路径和扩展名）
#             img_name = os.path.splitext(os.path.basename(img_path))[0]
            
#             # 保存为 .npz
#             npz_filename = os.path.join(output_folder, f'{img_name}.npz')
#             # np.savez(os.path.join(output_folder, f'{i}.npz'), image=image, label=label)
#             np.savez(npz_filename, image=image, label=label)
#             print(f'------------ Saved {img_name}.npz')
#         else:
#             print(f'Label not found for image {img_path}. Skipping...')


# input_folder = '/home/yjr/Swin-Unet/data/Synapse/images/'
# output_folder = '/home/yjr/Swin-Unet/data/Synapse/train_npz'
# convert_to_npz(input_folder, output_folder)
# print('转换完成。')





#----------------找出800 704 中公共图片----

# input_folder = "/home/yjr/Swin-Unet/data/Synapse/image"  # 你的输入文件夹路径
# output_folder = "/home/yjr/Swin-Unet/data/Synapse/Temp"  # 你的输出文件夹路径

# # 确保输出文件夹存在，如果不存在就创建
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # 遍历输入文件夹中的文件
# for filename in os.listdir(input_folder):
#     input_path = os.path.join(input_folder, filename)

#     # 检查是否为文件
#     if os.path.isfile(input_path):
#         # 构造输出路径
#         output_path = os.path.join(output_folder, filename)

#         # 检查是否存在对应的标签文件
#         label_path = input_path.replace('image', 'labels')  # 你的标签文件路径
#         if os.path.exists(label_path):
#             # 复制图像到新目录
#             shutil.copy(input_path, output_path)
#             print(f'------------ Copied {filename} to {output_folder}')