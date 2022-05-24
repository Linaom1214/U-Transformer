import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nets.centernet import CenterNet_Resnet50 ,CenterNet_Swin
from utils.utils import focal_loss, reg_l1_loss
from utils.dataloader import CenternetDataset, centernet_dataset_collate

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(net, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda):
    total_r_loss = 0
    total_c_loss = 0
    total_loss = 0
    val_loss = 0

    net.train()
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            with torch.no_grad():
                if cuda:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in batch]
                else:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in batch]

            batch_images, batch_hms, batch_regs, batch_reg_masks = batch
            optimizer.zero_grad()

            hm, offset = net(batch_images)
            c_loss = focal_loss(hm, batch_hms)
            off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)
            loss = c_loss + off_loss
            total_loss += loss.item()
            total_c_loss += c_loss.item()
            total_r_loss += off_loss.item()

            loss.backward()
            optimizer.step()

            pbar.set_postfix(**{'total_r_loss': total_r_loss / (iteration + 1),
                                'total_c_loss': total_c_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            with torch.no_grad():
                if cuda:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in batch]
                else:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in batch]

                batch_images, batch_hms, batch_regs, batch_reg_masks = batch

                hm, offset = net(batch_images)
                c_loss = focal_loss(hm, batch_hms)
                off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)
                loss = c_loss + off_loss

                val_loss += loss.item()

            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    print('Saving state, iter:', str(epoch + 1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
    (epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
    return  {'total_loss': total_loss / (epoch_size + 1),'val_loss': val_loss / (epoch_size_val + 1)}


if __name__ == "__main__":
    # ----------------------------------------#
    # Configuration
    input_shape = (256, 256, 3)
    classes_path = 'data/classes.txt'
    annotation_path = 'data/train2022.txt'
    lr = 1e-3
    Batch_size = 8
    Init_Epoch = 0
    Total_Epoch = 50
    backbone = "swin"
    weights = ''
    Cuda = True
    # ----------------------------------------#
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    assert backbone in ['resnet50', "swin"]
    if backbone == "resnet50":
        model = CenterNet_Resnet50(num_classes)
    else:
        model = CenterNet_Swin(num_classes)
    print('Now Using %s'%backbone)

    if weights:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(weights, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

    # ------------------Tensorboard---------------------------#
    link_mask = time.strftime('%Y%m%d-%H%M', time.localtime())
    tb_writer = SummaryWriter(log_dir=f"logs/{link_mask}")


    net = model.train()

    if Cuda:
        cudnn.benchmark = True
        net = net.cuda()

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val


    optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

    train_dataset = CenternetDataset(lines[:num_train], input_shape, num_classes, True)
    val_dataset = CenternetDataset(lines[num_train:], input_shape, num_classes, False)
    gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=2, pin_memory=True,
                     drop_last=True, collate_fn=centernet_dataset_collate)
    gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=2, pin_memory=True,
                         drop_last=True, collate_fn=centernet_dataset_collate)

    epoch_size = num_train // Batch_size
    epoch_size_val = num_val // Batch_size
    for epoch in range(Init_Epoch, Total_Epoch):
        loss = fit_one_epoch(net, epoch, epoch_size, epoch_size_val, gen, gen_val, Total_Epoch, Cuda)
        lr_scheduler.step(loss['val_loss'])
        tb_writer.add_scalar('val_loss', loss['val_loss'], epoch)
        tb_writer.add_scalar('total_loss', loss['total_loss'], epoch)