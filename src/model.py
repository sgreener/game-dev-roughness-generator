import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """U-Net Generator for pix2pix"""
    
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super(UNet, self).__init__()
        
        # Encoder
        self.e1 = nn.Conv2d(input_nc, ngf, 4, 2, 1)  # 256 -> 128
        self.e2 = self.encoder_block(ngf, ngf * 2)    # 128 -> 64
        self.e3 = self.encoder_block(ngf * 2, ngf * 4)  # 64 -> 32
        self.e4 = self.encoder_block(ngf * 4, ngf * 8)  # 32 -> 16
        self.e5 = self.encoder_block(ngf * 8, ngf * 8)  # 16 -> 8
        self.e6 = self.encoder_block(ngf * 8, ngf * 8)  # 8 -> 4
        self.e7 = self.encoder_block(ngf * 8, ngf * 8)  # 4 -> 2
        self.e8 = self.encoder_block(ngf * 8, ngf * 8, norm=False)  # 2 -> 1
        
        # Decoder
        self.d1 = self.decoder_block(ngf * 8, ngf * 8, dropout=True)  # 1 -> 2
        self.d2 = self.decoder_block(ngf * 16, ngf * 8, dropout=True)  # 2 -> 4
        self.d3 = self.decoder_block(ngf * 16, ngf * 8, dropout=True)  # 4 -> 8
        self.d4 = self.decoder_block(ngf * 16, ngf * 8)  # 8 -> 16
        self.d5 = self.decoder_block(ngf * 16, ngf * 4)  # 16 -> 32
        self.d6 = self.decoder_block(ngf * 8, ngf * 2)   # 32 -> 64
        self.d7 = self.decoder_block(ngf * 4, ngf)       # 64 -> 128
        
        self.d8 = nn.ConvTranspose2d(ngf * 2, output_nc, 4, 2, 1)  # 128 -> 256
        
    def encoder_block(self, in_channels, out_channels, norm=True):
        layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, 4, 2, 1)]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def decoder_block(self, in_channels, out_channels, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)
        
        # Decoder with skip connections
        d1 = self.d1(e8)
        d2 = self.d2(torch.cat([d1, e7], 1))
        d3 = self.d3(torch.cat([d2, e6], 1))
        d4 = self.d4(torch.cat([d3, e5], 1))
        d5 = self.d5(torch.cat([d4, e4], 1))
        d6 = self.d6(torch.cat([d5, e3], 1))
        d7 = self.d7(torch.cat([d6, e2], 1))
        d8 = self.d8(torch.cat([d7, e1], 1))
        
        return torch.tanh(d8)


class PatchGANDiscriminator(nn.Module):
    """PatchGAN Discriminator for pix2pix"""
    
    def __init__(self, input_nc=6, ndf=64):
        super(PatchGANDiscriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1)
        self.conv2 = self.discriminator_block(ndf, ndf * 2)
        self.conv3 = self.discriminator_block(ndf * 2, ndf * 4)
        self.conv4 = self.discriminator_block(ndf * 4, ndf * 8, stride=1)
        self.conv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 1)
        
    def discriminator_block(self, in_channels, out_channels, stride=2):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        return x5


class Pix2PixGAN:
    """Complete pix2pix model with generator and discriminator"""
    
    def __init__(self, device='cuda', lr=0.0002, beta1=0.5, lambda_l1=100):
        self.device = device
        self.lambda_l1 = lambda_l1
        
        # Initialize networks
        self.generator = UNet().to(device)
        self.discriminator = PatchGANDiscriminator().to(device)
        
        # Loss functions
        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()
        
        # Optimizers
        self.optimizer_g = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(beta1, 0.999)
        )
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999)
        )
    
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad for all the networks to avoid unnecessary computations"""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def train_step(self, real_a, real_b):
        """Single training step"""
        batch_size = real_a.size(0)
        
        # Real and fake labels
        real_label = torch.ones(batch_size, 1, 30, 30, device=self.device)
        fake_label = torch.zeros(batch_size, 1, 30, 30, device=self.device)
        
        # Train Generator
        self.set_requires_grad(self.discriminator, False)
        self.optimizer_g.zero_grad()
        
        fake_b = self.generator(real_a)
        fake_ab = torch.cat([real_a, fake_b], 1)
        pred_fake = self.discriminator(fake_ab)
        
        loss_g_gan = self.criterion_gan(pred_fake, real_label)
        loss_g_l1 = self.criterion_l1(fake_b, real_b) * self.lambda_l1
        loss_g = loss_g_gan + loss_g_l1
        
        loss_g.backward()
        self.optimizer_g.step()
        
        # Train Discriminator
        self.set_requires_grad(self.discriminator, True)
        self.optimizer_d.zero_grad()
        
        # Real pair
        real_ab = torch.cat([real_a, real_b], 1)
        pred_real = self.discriminator(real_ab)
        loss_d_real = self.criterion_gan(pred_real, real_label)
        
        # Fake pair
        fake_ab = torch.cat([real_a, fake_b.detach()], 1)
        pred_fake = self.discriminator(fake_ab)
        loss_d_fake = self.criterion_gan(pred_fake, fake_label)
        
        loss_d = (loss_d_real + loss_d_fake) * 0.5
        loss_d.backward()
        self.optimizer_d.step()
        
        return {
            'loss_g': loss_g.item(),
            'loss_g_gan': loss_g_gan.item(),
            'loss_g_l1': loss_g_l1.item(),
            'loss_d': loss_d.item(),
            'loss_d_real': loss_d_real.item(),
            'loss_d_fake': loss_d_fake.item()
        }
    
    def save_checkpoint(self, filepath, epoch, losses):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'losses': losses,
        }, filepath)
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        
        # Return epoch, not losses, as losses are not needed for resuming
        return checkpoint.get('epoch', 0)  # Use .get for backward compatibility
