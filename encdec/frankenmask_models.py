import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from torch import Tensor
from torch.autograd import Variable


class VAE_19(nn.Module):
    def __init__(self, nc, ngf, ndf, latent_variable_size, nc_input = 1, nc_output = 18):
        super(VAE_19, self).__init__()
        #self.cuda = True
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
        self.norm = nn.InstanceNorm2d

        # encoder
        self.encs = nn.Sequential(
            nn.Conv2d(nc_input, ndf, 4, 2, 1), 
            self.norm(ndf),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1),
            self.norm(ndf*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1),
            self.norm(ndf*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1),
            self.norm(ndf*8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*8, ndf*16, 4, 2, 1),
            self.norm(ndf*16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*16, ndf*32, 4, 2, 1),
            self.norm(ndf*32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*32, ndf*64, 4, 2, 1),
            self.norm(ndf*64),
            nn.LeakyReLU(0.2)


        )

        #self.vaes = nn.Linear(ndf*16*8*8, self.latent_variable_size)
        self.log_var = nn.Linear(ndf*64*2*2, self.latent_variable_size)
        self.mean = nn.Linear(ndf*64*2*2, self.latent_variable_size)
                                  
        #self.fc1 = nn.Linear(ndf*64*4*4, latent_variable_size)
        #self.fc2 = nn.Linear(ndf*64*4*4, latent_variable_size)

        # decoder
        #self.d1 = nn.Linear(self.nc*(self.latent_variable_size), ngf*16*8*8)
        self.d1 = nn.Linear(self.nc*(self.latent_variable_size), ngf*64*2*2)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(ngf*64, ngf*32, 3, 1)
        self.bn8 = self.norm(ngf*32)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(ngf*32, ngf*16, 3, 1)
        self.bn9 = self.norm(ngf*16)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(ngf*16, ngf*8, 3, 1)
        self.bn10 = self.norm(ngf*8)

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd4 = nn.ReplicationPad2d(1)
        self.d5 = nn.Conv2d(ngf*8, ngf*4, 3, 1)
        self.bn11 = self.norm(ngf*4)

        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd5 = nn.ReplicationPad2d(1)
        self.d6 = nn.Conv2d(ngf*4, ngf*2, 3, 1)
        self.bn12 = self.norm(ngf*2)

        self.up6 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd6 = nn.ReplicationPad2d(1)
        self.d7 = nn.Conv2d(ngf*2, ngf, 3, 1)
        self.bn13 = self.norm(ngf)

        self.up7 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd7 = nn.ReplicationPad2d(1)
        self.d8 = nn.Conv2d(ngf, nc_output, 3, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d((2, 2), (2, 2))

        self.pos_enc = nn.Parameter(torch.zeros(self.nc, 1, self.latent_variable_size))
        #self.pos_enc = PositionalEncoding(d_model = self.latent_variable_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_variable_size, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

        #self.decoder_skilled = Decoder()
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode_mask_parts(self, x):
        mu = []
        lv = []
        z = []

        for i in range(self.nc):
            x_in = x[:,i,:,:,].unsqueeze(1)

            enc_x_c = self.encs(x_in)
            enc_x_c = enc_x_c.view(-1, self.ndf*64*2*2) #.view(-1, self.ndf*16*8*8) #
            log_var = self.log_var(enc_x_c)
            mean = self.mean(enc_x_c)
            
            rep_z = self.reparametrize(mean, log_var)

            mu.append(mean) 
            lv.append(log_var) 
            z.append(rep_z) 

        mu = torch.stack(mu).permute(1,0,2)  # bs, 18, 512
        lv = torch.stack(lv).permute(1,0,2)  # bs, 18, 512
        z = torch.stack(z)  
        return mu, lv, z
    def transformer_pass(self, mu):
        # self-attention
        #mu = mu * math.sqrt(self.latent_variable_size)
        #mu = self.pos_enc(mu)
        
        mu = mu + self.pos_enc
        mu = self.transformer_encoder(mu)

        return mu.permute(1,0,2)

    def encode(self, x):
        
        mu, lv, z = self.encode_mask_parts(x) #[[bs,512]x19]

        z = self.transformer_pass(z)
        
        return  mu, lv, z

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.ngf*64, 2, 2) #.view(-1, self.ngf*16, 8, 8)#
        h2 = self.leakyrelu(self.bn8(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn9(self.d3(self.pd2(self.up2(h2)))))
        h4 = self.leakyrelu(self.bn10(self.d4(self.pd3(self.up3(h3)))))
        h5 = self.leakyrelu(self.bn11(self.d5(self.pd4(self.up4(h4)))))
        h6 = self.leakyrelu(self.bn12(self.d6(self.pd5(self.up5(h5)))))
        h7 = self.leakyrelu(self.bn13(self.d7(self.pd6(self.up6(h6)))))
        return self.d8(self.pd7(self.up7(h7)))

    def kl_loss(self, mu, log_var):
        kld_loss = 0 #torch.Tensor([0]).cuda()
        
        # for i in range(mu.size(1)):
        #     mu_el = mu[:,i]
        #     lv_el = log_var[:,i]

        #     #kld_loss += (-0.5 * (1 + lv_el - mu_el ** 2 - lv_el.exp())).mean() #, dim = 1) torch.mean(, dim = 0)
        #     kld_loss += torch.mean(-0.5 * torch.sum(1 + lv_el - mu_el ** 2 - lv_el.exp(), dim = 1), dim = 0)

        mu_cat = mu.reshape(-1,self.nc*self.latent_variable_size)
        log_var_cat = log_var.reshape(-1,self.nc*self.latent_variable_size)

        kld_loss += torch.mean(-0.5 * torch.sum(1 + log_var_cat - mu_cat ** 2 - log_var_cat.exp(), dim = 1), dim = 0)
        
        return kld_loss #torch.mean(kld_loss)

    def get_latent_var(self, x):
        mu, lv, z = self.encode(x)
        return mu, lv, z 

    def forward(self, x):
        mu, lv, z = self.encode(x)
        z = z.reshape(-1,self.nc*self.latent_variable_size)
        res = self.decode(z)
        
        return res, mu, lv

class AE_19(nn.Module):
    def __init__(self, nc, ngf, ndf, latent_variable_size, nc_input = 1, nc_output = 18):
        super(AE_19, self).__init__()
        #self.cuda = True
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
        self.norm = nn.InstanceNorm2d

        # encoder
        self.encs = nn.Sequential(
            nn.Conv2d(nc_input, ndf, 4, 2, 1), 
            self.norm(ndf),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1),
            self.norm(ndf*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1),
            self.norm(ndf*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1),
            self.norm(ndf*8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*8, ndf*16, 4, 2, 1),
            self.norm(ndf*16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*16, ndf*32, 4, 2, 1),
            self.norm(ndf*32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*32, ndf*64, 4, 2, 1),
            self.norm(ndf*64),
            nn.LeakyReLU(0.2)
        )

        self.vaes = nn.Linear(ndf*64*2*2, self.latent_variable_size)
                                

        # decoder
        self.d1 = nn.Linear(self.nc*(self.latent_variable_size), ngf*64*2*2)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(ngf*64, ngf*32, 3, 1)
        self.bn8 = self.norm(ngf*32)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(ngf*32, ngf*16, 3, 1)
        self.bn9 = self.norm(ngf*16)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(ngf*16, ngf*8, 3, 1)
        self.bn10 = self.norm(ngf*8)

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd4 = nn.ReplicationPad2d(1)
        self.d5 = nn.Conv2d(ngf*8, ngf*4, 3, 1)
        self.bn11 = self.norm(ngf*4)

        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd5 = nn.ReplicationPad2d(1)
        self.d6 = nn.Conv2d(ngf*4, ngf*2, 3, 1)
        self.bn12 = self.norm(ngf*2)

        self.up6 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd6 = nn.ReplicationPad2d(1)
        self.d7 = nn.Conv2d(ngf*2, ngf, 3, 1)
        self.bn13 = self.norm(ngf)

        self.up7 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd7 = nn.ReplicationPad2d(1)
        self.d8 = nn.Conv2d(ngf, nc_output, 3, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d((2, 2), (2, 2))

        self.pos_enc = nn.Parameter(torch.zeros(self.nc, 1, self.latent_variable_size))
        #self.pos_enc = PositionalEncoding(d_model = self.latent_variable_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_variable_size, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

        #self.decoder_skilled = Decoder()

    def encode_mask_parts(self, x):
        mu = []
        for i in range(self.nc):
            x_in = x[:,i,:,:,].unsqueeze(1)

            enc_x_c = self.encs(x_in)
            
            enc_x_c = enc_x_c.view(-1, self.ndf*64*2*2) #.view(-1, self.ndf*16*8*8) #
            
            enc_x_c = self.vaes(enc_x_c)
            
            mu.append(
                enc_x_c
            )    
        mu = torch.stack(mu)  
        return mu
    def transformer_pass(self, mu):
        # self-attention
        #mu = mu * math.sqrt(self.latent_variable_size)
        #mu = self.pos_enc(mu)
        
        mu = mu + self.pos_enc
        mu = self.transformer_encoder(mu)

        return mu.permute(1,0,2)

    def encode(self, x):
        
        mu = self.encode_mask_parts(x)
        mu = self.transformer_pass(mu)
        
        return mu

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.ngf*64, 2, 2) #.view(-1, self.ngf*16, 8, 8)#
        h2 = self.leakyrelu(self.bn8(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn9(self.d3(self.pd2(self.up2(h2)))))
        h4 = self.leakyrelu(self.bn10(self.d4(self.pd3(self.up3(h3)))))
        h5 = self.leakyrelu(self.bn11(self.d5(self.pd4(self.up4(h4)))))
        h6 = self.leakyrelu(self.bn12(self.d6(self.pd5(self.up5(h5)))))
        h7 = self.leakyrelu(self.bn13(self.d7(self.pd6(self.up6(h6)))))
        return self.d8(self.pd7(self.up7(h7)))

    def get_latent_var(self, x):
        mu = self.encode(x)
        return mu

    def forward(self, x):
        mu = self.encode(x)
        mu = mu.reshape(-1,self.nc*self.latent_variable_size)
        res = self.decode(mu)
        
        return res #, x, mu, logvar