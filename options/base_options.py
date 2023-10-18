"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
import argparse
import os
from util import util
import torch
import models
import data
import pickle


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--pose_txt_path', type=str, default='/media/fontanini/Discone/DOTTORATO/REPOSITORY/trunk/SEAN-maskSwap-master/datasets/CelebA-HQ/CelebAMask-HQ-pose-anno.txt', help='models are saved here')
        parser.add_argument('--use_poses', action='store_true', help='load poses numbers')
        parser.add_argument('--input_styles', action='store_true', help='use styles as input in the decoder instead of mask codes')
        parser.add_argument('--linear_enc', action='store_true', help='use smaller MLP as encoder for mask parts')
        parser.add_argument('--sean_style_encoder', action='store_true', help='use super bad Sean style Encoder')
        parser.add_argument('--use_T', action='store_true', help='use Transformer encoder')
        parser.add_argument('--chann_out_21', action='store_true', help='output both rgb and mask')
        parser.add_argument('--D_sw', action='store_true', help='train discriminator using swapped images')
        parser.add_argument('--no_self_att', action='store_true', help='remove sel attention from cross att blocks')
        parser.add_argument('--cross_att_all_layers', action='store_true', help='use cross attention in all the decoder layers')
        
        parser.add_argument('--multi_scale_style_enc', action='store_true', help='use a multy scale style encoder')
        parser.add_argument('--elegant_solution', action='store_true', help='concatenate along words')

        parser.add_argument('--no_embedding', action='store_true', help='do not embed the mask')
        
        parser.add_argument('--no_self_last_layers', action='store_true', help='do not use self attention in the last two layers')
        parser.add_argument('--exclude_bg', action='store_true', help='remove gb from input mask')
        parser.add_argument('--single_layer_mask_enc', action='store_true', help='use a single layer for encoding masks')

        parser.add_argument('--use_noise', action='store_true', help='generate images using noise')
        parser.add_argument('--style_feat_loss', action='store_true', help='calculate style feature loss')
        parser.add_argument('--ds_loss', action='store_true', help='calculate diversity loss')

        parser.add_argument('--mask_gen', action='store_true', help='generate mask')

        parser.add_argument('--load_D', action='store_true', help='load discriminator')

        parser.add_argument('--no_model_load', action='store_true', help='gdo not load model in any case')

        parser.add_argument('--style_enc_kernel', type=int, default=5, help='kernel of the first layer of style encoder')
        parser.add_argument('--style_enc_feat_dim', type=int, default=4, help='feature dimention of the style encoder')

        parser.add_argument('--style_dim', type=int, default=256, help='style dimention')

        parser.add_argument('--head_dim', type=int, default=64, help='cross att heads dimention')
        parser.add_argument('--head_num', type=int, default=8, help='number of heads in cross att')

        parser.add_argument('--SEAN', action='store_true', help='load original SEAN model')

        parser.add_argument('--otf_style', action='store_true', help='new style encoder without pooling')

        parser.add_argument('--use', action='store_true', help='unsupervised style extraction')
        parser.add_argument('--ce_weights', action='store_true', help='use cross entropy weights')
        parser.add_argument('--float_mask', action='store_true', help='use continue_mask')
        parser.add_argument('--att_sigmoid', action='store_true', help='use sigmoid instead of softmax in attention')

        # experiment specifics
        parser.add_argument('--name', type=str, default='label2coco', help='name of the experiment. It decides where to store samples and models')

        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--sample_dir', type=str, default='./samples', help='samples are saved here')
        parser.add_argument('--test_dir', type=str, default='./test', help='test samples are saved here')
        parser.add_argument('--model', type=str, default='pix2pix', help='which model to use')
        parser.add_argument('--norm_G', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--norm_E', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        # input/output sizes
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--preprocess_mode', type=str, default='scale_width_and_crop', help='scaling and cropping of images at load time.', choices=("resize_and_crop", "crop", "scale_width", "scale_width_and_crop", "scale_shortside", "scale_shortside_and_crop", "fixed", "none"))
        parser.add_argument('--load_size', type=int, default=1024, help='Scale images to this size. The final image will be cropped to --crop_size.')
        parser.add_argument('--crop_size', type=int, default=512, help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
        parser.add_argument('--label_nc', type=int, default=182, help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.')
        parser.add_argument('--contain_dontcare_label', action='store_true', help='if the label map contains dontcare label (dontcare=255)')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for setting inputs
        parser.add_argument('--dataroot', type=str, default='./datasets/cityscapes/')
        parser.add_argument('--dataset_mode', type=str, default='coco')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
        parser.add_argument('--nThreads', default=28, type=int, help='# threads for loading data')
        parser.add_argument('--max_dataset_size', type=int, default=sys.maxsize, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--load_from_opt_file', action='store_true', help='load the options from checkpoints and use that as default')
        parser.add_argument('--cache_filelist_write', action='store_true', help='saves the current filelist into a text file, so that it loads faster')
        parser.add_argument('--cache_filelist_read', action='store_true', help='reads from the file list cache')

        # for displays
        parser.add_argument('--display_winsize', type=int, default=400, help='display window size')

        # for generator
        parser.add_argument('--netG', type=str, default='spade', help='selects model to use for netG (pix2pixhd | spade)')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
        parser.add_argument('--z_dim', type=int, default=256,
                            help="dimension of the latent z vector")

        # for instance-wise features
        parser.add_argument('--no_instance', action='store_true', help='if specified, do *not* add instance map as input')
        parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
        parser.add_argument('--use_vae', action='store_true', help='enable training with an image encoder.')

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)

        # modify dataset-related parser options
        dataset_mode = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_mode)
        parser = dataset_option_setter(parser, self.isTrain)

        opt, unknown = parser.parse_known_args()

        # if there is opt_file, load it.
        # The previous default options will be overwritten
        if opt.load_from_opt_file:
            parser = self.update_options_from_file(parser, opt)

        opt = parser.parse_args()
        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if makedir:
            util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def parse(self, save=False):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        self.print_options(opt)
        if opt.isTrain:
            self.save_options(opt)

        # Set semantic_nc based on the option.
        # This will be convenient in many places
        opt.semantic_nc = opt.label_nc + \
            (1 if opt.contain_dontcare_label else 0) + \
            (0 if opt.no_instance else 1)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        #opt.gpu_ids = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
        
        print("Devices num: {}".format(len(opt.gpu_ids)))
        for i in range(len(opt.gpu_ids)):
            print(torch.cuda.get_device_name(opt.gpu_ids[i]), flush = True)
            print(torch.cuda.get_device_properties(opt.gpu_ids[i]), flush = True)
            print ('Current cuda device ', torch.cuda.current_device(), flush = True)
            print("----------------")

        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        assert len(opt.gpu_ids) == 0 or opt.batchSize % len(opt.gpu_ids) == 0, \
            "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
            % (opt.batchSize, len(opt.gpu_ids))

        self.opt = opt
        return self.opt
