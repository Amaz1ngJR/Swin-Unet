import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume
import cv2
from torchvision.transforms import Resize
from torchvision.transforms import RandomCrop
import tensorflow as tf
#from torchvision.transforms.functional import crop

def random_crop_images(image1, image2, crop_size=(256, 256), seed=None):
    # 获取图像的大小
    width = 512 
    height = 512

    # 设置种子
    if seed is not None:
        random.seed(seed)

    # 随机生成裁剪框的左上角坐标
    left = random.randint(0, width - crop_size[0])
    top = random.randint(0, height - crop_size[1])

    # 裁剪张量
    cropped_tensor1 = image1[:, :, top:top + crop_size[1], left:left + crop_size[0]]
    cropped_tensor2 = image2[:, top:top + crop_size[1], left:left + crop_size[0]]

    return cropped_tensor1, cropped_tensor2

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes  # 2
    batch_size = args.batch_size * args.n_gpu # 2
    # 初始化数据集 对图片进行随机旋转、翻转 读入image(512,512,3)->(3,512,512) 将label转为long
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    
    #损失函数和优化器初始化
    ce_loss = CrossEntropyLoss()
    #ce_loss = nn.L1Loss()
    #ce_loss = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    
    # 主训练循环
    i = 0
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            # image_batch.size() [1,3,512,512] label_batch.size() [1,512,512]
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            # # 使用相同的种子确保图像和标签在相同的位置被裁剪
            # seed_ = torch.randint(0, 2**32 - 1, (1,))

            # resize_transform = Resize(512) # 设置resize大小

            # # 使用 crop 进行裁剪
            # image_batch, label_batch = random_crop_images(image_batch,
            #         label_batch, seed = seed_)
            
            # image_batch = resize_transform(image_batch)
            # label_batch = resize_transform(label_batch)

            outputs = model(image_batch)
            outputs = outputs.squeeze(0)
            # i = i + 1
            # if i in [5000, 29000, 20000]:
            #image_batch = image_batch.squeeze(0)
            # cv2.imshow("img", image_batch.permute(1,2,0).detach().cpu().numpy().astype(np.uint8) * 255)
            # cv2.imshow("GT", label_batch.permute(1,2,0).detach().cpu().numpy().astype(np.uint8) * 255)
            # cv2.imshow("outputs", outputs.permute(1,2,0).detach().cpu().numpy().astype(np.uint8) * 255)
            # cv2.waitKey(0)
            loss_ce = ce_loss(outputs, label_batch)

            # optimizer.zero_grad()
            # loss_ce.backward()
            # optimizer.step()

            loss = 0.4 * loss_ce 

            # cv2.imshow("GT", label_batch.permute(1,2,0).detach().cpu().numpy().astype(np.uint8) * 255)
            # cv2.imshow("outputs", outputs.permute(1,2,0).detach().cpu().numpy().astype(np.uint8) * 255)
            # cv2.waitKey(0)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9  #学习率逐渐降低
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break
    writer.close()
    return "Training Finished!"      
