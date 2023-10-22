import torch
import torch.nn as nn
from helper import ResidualBlock, NonLocalBlock, UpSampleBlock, GroupNorm, Swish
import argparse
from eg3d.eg3d.training.networks_stylegan2 import Generator

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.torch.nn.init.normal_(m.weight)
        m.bias.data.fill_(0.1)

class SmallGenerator(nn.Module):
    def __init__(self,
                input_dim=256,
                out_reso=64,
                out_channels=4):
        super(SmallGenerator, self).__init__()
        self.model = nn.Sequential(
                        nn.Linear(input_dim, input_dim*2),
                        nn.Linear(input_dim*2, out_reso*out_reso*out_channels*2),
                        nn.Linear(out_reso*out_reso*out_channels*2, out_reso*out_reso*out_channels),
                    )
        self.model.apply(init_weights)
        self.model.require_grad = True
        self.out_reso = out_reso
        self.out_channels = out_channels
    def forward(self, codebook, c=None):
        feats = self.model(codebook)
        feats = feats.reshape(-1, self.out_channels, self.out_reso, self.out_reso)
        return feats

class SmallGenerator2(nn.Module):
    def __init__(self,
                input_dim=256,
                out_reso=64,
                out_channels=4):
        super(SmallGenerator2, self).__init__()
        
        # Initial transformation using Linear layers
        self.linear_layers = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, 256 * 8 * 8),  # Prepare for 8x8 feature maps
            nn.ReLU()
        )
        
        # Up-Sampling using ConvTranspose2d layers
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=1)  # 64x64 -> 64x64 with `out_channels`
        )
        
        self.out_reso = out_reso
        self.out_channels = out_channels

    def forward(self, codebook, c=None):
        # Forward pass through Linear layers
        x = self.linear_layers(codebook)
        
        # Reshape for ConvTranspose2d layers
        x = x.view(-1, 256, 8, 8)
        
        # Forward pass through ConvTranspose2d layers
        x = self.deconv_layers(x)
        
        return x

class Codebook(nn.Module):
    def __init__(self, args):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = args.latent_dim
        self.beta = args.beta
        self.args = args

        if args.codebook_interpolate:
            self.embedding = nn.Embedding(self.num_codebook_vectors*2, self.latent_dim)
        else:
            self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        # self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

        # self.generator = Generator(z_dim=self.latent_dim, # Input latent (Z) dimensionality.
        #                            c_dim=0, # Conditioning label (C) dimensionality.
        #                            w_dim=128, # Intermediate latent (W) dimensionality.
        #                            img_resolution=64, # Output resolution.
        #                            img_channels=4, # Number of output color channels.
        #                            )

        self.generator = SmallGenerator(self.latent_dim)

    def forward(self, B):
        # z = z.permute(0, 2, 3, 1).contiguous()
        # z_flattened = z.view(-1, self.latent_dim)


        # # (a-b)^2
        # d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
        #     torch.sum(self.embedding.weight**2, dim=1) - \
        #     2*(torch.matmul(z_flattened, self.embedding.weight.t()))

        # min_encoding_indices = torch.argmin(d, dim=1)
        # z_q = self.embedding(min_encoding_indices).view(z.shape)

        # loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)

        # z_q = z + (z_q - z).detach()

        # z_q = z_q.permute(0, 3, 1, 2)

        # return z_q, min_encoding_indices, loss
        if self.args.codebook_interpolate:
            indices = torch.randperm(self.num_codebook_vectors*2)[:B*2].to('cuda:0')
            sampled_codebooks1 = self.embedding(indices[0]).unsqueeze(0)
            sampled_codebooks2 = self.embedding(indices[1]).unsqueeze(0)
            alpha = 0.5
            sampled_codebooks = alpha*sampled_codebooks1 + (1-alpha)*sampled_codebooks2
        else:
            indices = torch.randperm(self.num_codebook_vectors)[:B].to('cuda:0')
            sampled_codebooks = self.embedding(indices)
        output_feats = self.generator(sampled_codebooks, None)

        return output_feats # B, C, W, H

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        channels = [512, 256, 256, 128, 128]
        attn_resolutions = [16]
        num_res_blocks = 3
        resolution = 16

        in_channels = channels[0]
        layers = [nn.Conv2d(args.latent_dim, in_channels, 3, 1, 1),
                  ResidualBlock(in_channels, in_channels),
                  NonLocalBlock(in_channels),
                  ResidualBlock(in_channels, in_channels)]

        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i != 0:
                layers.append(UpSampleBlock(in_channels))
                resolution *= 2

        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels, args.image_channels, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def get_parser(**parser_kwargs):

    parser = argparse.ArgumentParser(**parser_kwargs)
    # parameters
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=6, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = get_parser()
    codebook = Codebook(args)
    B = 4
    sample_codebooks = codebook(B)
    
    breakpoint()
