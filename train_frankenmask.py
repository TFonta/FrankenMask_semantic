"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import data
from options.train_options import TrainOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
import numpy as np
import sys
import random
from util import util
from torchvision import transforms
import torchvision
import torchvision.transforms as TT
import os
import copy
import torch
import torch.nn as nn

from encdec.frankenmask_models import AE_19


def set_grad_parameters(m, requires_grad = True):
    for p in m.parameters():
        p.requires_grad = requires_grad


def swap_codes(VAEs_res, p):
    lat = [VAEs_res[i][2] for i in p]  # [2] is re-parameterized z
    for i, l in enumerate(lat):
        tmp = l[:l.shape[0] // 2].clone()
        l[:l.shape[0] // 2] = l[l.shape[0] // 2:]
        l[l.shape[0] // 2:] = tmp
        VAEs_res[p[i]][2] = l
    return VAEs_res


def swap_z(zc):
    z = [f.clone() for f in zc]
    for i, f in enumerate(z):
        tmp = f[:f.shape[0]//2, :, :, :].clone()
        f[:f.shape[0]//2, :, :, :] = f[f.shape[0]//2:, :, :, :]
        f[f.shape[0] // 2:, :, :, :] = tmp
        z[i] = f
    return z


def one_hot(targets, nclasses):
    targets_extend = targets.clone()        
    targets_extend.unsqueeze_(1)  # convert to Nx1xHxW
    one_hot = torch.FloatTensor(targets_extend.size(0), nclasses, targets_extend.size(2), targets_extend.size(3)).zero_()
    one_hot = one_hot.cuda()
    one_hot.scatter_(1, targets_extend, 1)
    return one_hot


def swap_parts(input_semantics, p, randp=False, k=1):
    # Select existing part in both images
    sumParts = np.sum(input_semantics.cpu().numpy(), axis=(2, 3))
    input_semantics_sw = input_semantics.clone()
    #mask_1 = input_semantics[:real_image.shape[0] // 2, :, :, :].clone()
    #mask_2 = input_semantics[real_image.shape[0] // 2:, :, :, :].clone()
    
    half_batch = input_semantics.size(0)//2

    swap_label = torch.zeros(input_semantics.size(0), input_semantics.size(1)).cuda()
    for el_idx in range(half_batch):
        commonParts  = np.all([sumParts[el_idx, :], sumParts[el_idx + half_batch, :]], axis=0)
        
        idx = [i for i, x in enumerate(commonParts) if x]
        if randp:
            pToSwap = random.choices(idx, k=k)
        else:
            pToSwap = list(set(p) & set(idx))

        
        for el in pToSwap:
            swap_label[el_idx,el] = 1.
            swap_label[el_idx + half_batch,el] = 1.
		
        if len(pToSwap) > 0:
            input_semantics_sw[el_idx, pToSwap, :, :] = input_semantics[el_idx + half_batch, pToSwap, :, :]
            input_semantics_sw[el_idx + half_batch, pToSwap, :, :] = input_semantics[el_idx, pToSwap, :, :]

    return input_semantics_sw, swap_label

def reverse_swap(gen_semantics_sw, input_semantics, p):
    gen_semantics_sw_sw = gen_semantics_sw.clone()
    for el_idx in range(input_semantics.size(0)):
        pToSwap = list(set(p))
        #gen_semantics_sw_sw[el_idx, pToSwap] = gen_semantics_sw_sw[el_idx, pToSwap].detach()
        gen_semantics_sw_sw[el_idx, pToSwap] = input_semantics[el_idx, pToSwap]
    return gen_semantics_sw_sw


def remove_part_with_skin(input_semantics, p):
    semantics_wo_part = input_semantics.clone()
    semantics_wo_part[:, 1, :, :] += semantics_wo_part[:, p, :, :].squeeze()
    semantics_wo_part[:, p, :, :] = 0
    return semantics_wo_part


def create_template_mask(input_semantics, p):
    semantics_wo_part = input_semantics.clone()
    semantics_wo_part[:, 1, :, :] += semantics_wo_part[:, p, :, :].sum(1)
    semantics_wo_part[:, p, :, :] = 0
    return semantics_wo_part


def fuse_mask_with_parts(input_semantics, p):
    semantics_wo_part = input_semantics.clone()
    semantics_wo_part[:, 1, :, :] -= semantics_wo_part[:, p, :, :].sum(1)
    return semantics_wo_part

def separate_channels(image):
    im_chann = {}
    for c in range(image.size(1)):
        im_chann[str(c)] = image[:,c:c+1:,]
    return im_chann

def add_background_channel(x):

    bg_mask = torch.sum(x, dim = 1) > 0
    bg_mask = 1. - bg_mask.unsqueeze(1).float()
    #bg_mask = torch.ones(x.size(0),1,x.size(2),x.size(3)).cuda().float()
    return torch.cat((bg_mask, x), dim = 1)

    

device = 'cuda:0'

label_list = ['bkgrnd', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
              'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
gen_parts = [1, 2, 4, 5, 6, 7, 10, 11, 12, 13]



#print(len(label_list))
fixed_parts = np.setdiff1d(np.array(range(19)), np.array(gen_parts))
CE_lw = 1
CE_hw = 1
CE_w = torch.ones(len(label_list))*CE_lw
CE_w[gen_parts] = CE_hw
CE_w = CE_w.cuda()

#opt = TestOptions().parse()
opt = TrainOptions().parse()
opt.status = 'train'
resume = False
pretrain_encDec = False
opt.serial_batches = False


opt.isTrain = True
dataloader = data.create_dataloader(opt)

opt.isTrain = False
opt.serial_batches = True

dataloader_test = data.create_dataloader(opt)

opt.serial_batches = False

visualizer = Visualizer(opt)


# -----------------------------------
it_test = iter(dataloader_test)
fixed_batch = next(it_test)
fixed_batch_2 = next(it_test)


PART_SWAPPED = 1

opt.nc = 19

opt.continue_train = False
opt.isTrain = True
# Keep SEAN frozen i.e. non trainable
sean = Pix2PixModel(opt).cuda()
sean.eval()
opt.isTrain = False


U = AE_19(18,32,32,opt.style_dim).cuda() 

optimU = torch.optim.Adam(U.parameters(), lr=1e-5)

# Losses
U_CE = nn.CrossEntropyLoss()
MSE_encDec = nn.MSELoss(reduction='sum')
BCE = nn.BCEWithLogitsLoss()
L1_loss = nn.L1Loss()

# Paths for results and models
examples_path = "./examples/FrankenMask/"
checkpoint_path = './checkpoints/FrankenMask/'
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path, exist_ok = True)

os.makedirs(opt.checkpoints_dir, exist_ok = True)

if resume:
    state_dict = torch.load(checkpoint_path + 'swapNet_epoch.pt', map_location='cpu')
    U.load_state_dict(state_dict['model_U_state_dict'])
    #D.load_state_dict(state_dict['model_D_state_dict'])
    optimU.load_state_dict(state_dict['optimizer_U_state_dict'])
    #optimD.load_state_dict(state_dict['optimizer_D_state_dict'])
    start_epoch = state_dict['epoch']

    print("TRAINING RESUMED")
else:
    start_epoch = 0

epochs = 50
#writer = SummaryWriter()

real_label_tensor = torch.FloatTensor(1).fill_(1.0).expand(opt.batchSize).requires_grad_(False).cuda()
fake_label_tensor = torch.FloatTensor(1).fill_(0.0).expand(opt.batchSize).requires_grad_(False).cuda()

print("DATALOADER: ", len(dataloader))
print("DATALOADER TEST: ", len(dataloader_test))


dataloader_2 = copy.deepcopy(dataloader)
for epoch in range(start_epoch, epochs):
    
    #print(type(dataloader))
    it = iter(dataloader_2)
    #for i in range(1): #range(len(dataloader)//2):
    for i, data_i in enumerate(dataloader):    
        
        # Generate masks as 19 channel images
        input_semantics, real_image = sean.preprocess_input(data_i)
        input_semantics.cuda()
        real_image.cuda()
        # remove background
        input_semantics_no_bg = input_semantics[:,1:]

        # Pick random part
        p = random.choices(gen_parts, k=PART_SWAPPED)
        # Reconstruct input 
        #gen_semantics, mu, lv = U(input_semantics_no_bg)
        gen_semantics = U(input_semantics_no_bg)
        # add background
        gen_semantics = add_background_channel(gen_semantics)
        # Reconstruction loss
        rec_loss = U_CE(gen_semantics, data_i['label'].squeeze())
        #rec_loss = MSE_encDec(gen_semantics, input_semantics)
        # ---------------------------
        #kl_loss = U.kl_loss(mu, lv)
        
        vae_loss = rec_loss #+ 0.000001*kl_loss

        U.zero_grad()
        vae_loss.backward()
        optimU.step()

        # ----------------------------------------------
		

        if i % 10 == 0:
            print('Epoch: {} Iter: {} ---- Gloss: Rec: {:.4f}'.format(epoch, i,        
                                                                        rec_loss.item()), flush = True)
        
        
        
        if i % 50 == 0:

            #fixed_batch = next(iter(dataloader_test))
            input_semantics_test, real_image_test = sean.preprocess_input(fixed_batch)
            input_semantics_test.cuda()
            input_semantics_test_no_bg = input_semantics_test[:,1:]
            real_image_test.cuda()

            U.eval()

            with torch.no_grad():

                p = random.choices(gen_parts, k=PART_SWAPPED)

                # Swap parts in semantic mask for TEST
                input_semantics_sw, _ = swap_parts(input_semantics_test, p, randp=False, k=len(p))
                input_semantics_sw_no_bg = input_semantics_sw[:,1:]

                input_semantics_sw_bg = add_background_channel(input_semantics_sw_no_bg)
                cls_idx_sw = torch.argmax(input_semantics_sw_bg, dim=1)
                input_semantics_sw_oneHot = one_hot(cls_idx_sw, len(label_list)).cpu()

                gen_semantics= U(input_semantics_test_no_bg)
                gen_semantics = add_background_channel(gen_semantics)
                cls_idx = torch.argmax(gen_semantics, dim=1)
                gen_semantics_oneHot = one_hot(cls_idx, len(label_list)).cpu()                
                
                gen_semantics_sw = U(input_semantics_sw_no_bg)  
                gen_semantics_sw_bg = add_background_channel(gen_semantics_sw)                                                                                 
                cls_idx_sw = torch.argmax(gen_semantics_sw_bg, dim=1)
                gen_semantics_sw_oneHot = one_hot(cls_idx_sw, len(label_list)).cpu()

                grid_img_real = util.tensor2im(torchvision.utils.make_grid(real_image_test).detach().cpu())
                grid_label_real = util.tensor2label(torchvision.utils.make_grid(input_semantics_test).detach().cpu(), opt.nc)

                grid_label_input_real = util.tensor2label(torchvision.utils.make_grid(input_semantics_sw_oneHot).detach().cpu(), 19)

                grid_label_sw = util.tensor2label(torchvision.utils.make_grid(gen_semantics_sw_oneHot).detach().cpu(), opt.nc)

                grid_label_o = util.tensor2label(torchvision.utils.make_grid(gen_semantics_oneHot).detach().cpu(), opt.nc)

                grid_list = [grid_img_real, grid_label_real, grid_label_input_real, grid_label_sw, grid_label_o]
                grid = np.concatenate(grid_list, axis=0)

              
                
                os.makedirs(opt.sample_dir + str(epoch) + '/',exist_ok=True)
                
                transforms.ToPILImage()(grid).save(opt.sample_dir + str(epoch) + '/' + str(i) + '_' + label_list[p[0]] + '.png')

            U.train()

    torch.save({'model_U_state_dict': U.state_dict(),
                'epoch': epoch,
                'optimizer_U_state_dict': optimU.state_dict()
                },
               opt.checkpoints_dir + 'swapNet_epoch.pt')

