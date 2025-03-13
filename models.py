import torch
import torch.nn as nn
import torchvision.models as models

class ContrastiveVAEClassifier(nn.Module):
    def __init__(self, feature_dim=256, latent_dim=128, num_classes=8):
        super().__init__()
        self.cnn = CNNFeatureExtractor(feature_dim)  # CNN Backbone
        self.vae = VAE(feature_dim, latent_dim)  # VAE for anomaly detection
        self.classifier = nn.Linear(latent_dim, num_classes)  # Classification Head
        
    def forward(self, x):
        features = self.cnn(x)  # Extract CNN features
        recon_features, mu, logvar = self.vae(features)  # Encode & Decode via VAE
        logits = self.classifier(mu)  # Classify based on latent space
        
        # Compute Reconstruction Loss
        recon_loss = F.mse_loss(recon_features, features)
        
        return logits, recon_loss, mu, logvar


class CNNFeatureExtractor(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        base_model = models.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = self.encoder(x).squeeze()
        x = self.fc(x)
        return x



class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) class.
    
    Args:
        input_dim (int): Dimensionality of the input data.
        hidden_dim (int): Dimensionality of the hidden layer.
        latent_dim (int): Dimensionality of the latent space.
    """
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
                
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 8, latent_dim*2), # 2 for mean and variance.
        )
        # self.parametr = nn.Linear(latent_dim, 2 * latent_dim)
        self.softplus = nn.Softplus()
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 8),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 8, hidden_dim // 4),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim, input_dim),
        )
        
    def encode(self, x, eps: float = 1e-8):
        """
        Encodes the input data into the latent space.
        
        Args:
            x (torch.Tensor): Input data.
            eps (float): Small value to avoid numerical instability.
        
        Returns:
            torch.distributions.MultivariateNormal: Normal distribution of the encoded data.
        """
        x = self.encoder(x)
        # x = self.parametr(lat_x)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)
        
        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)
        
    def reparameterize(self, dist):
        """
        Reparameterizes the encoded data to sample from the latent space.
        
        Args:
            dist (torch.distributions.MultivariateNormal): Normal distribution of the encoded data.
        Returns:
            torch.Tensor: Sampled data from the latent space.
        """
        return dist.rsample()
    
    def decode(self, z):
        """
        Decodes the data from the latent space to the original input space.
        
        Args:
            z (torch.Tensor): Data in the latent space.
        
        Returns:
            torch.Tensor: Reconstructed data in the original input space.
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        Performs a forward pass of the VAE.
        
        Args:
            x (torch.Tensor): Input data.
            compute_loss (bool): Whether to compute the loss or not.
        
        Returns:
            VAEOutput: VAE output dataclass.
        """
        dist = self.encode(x)
        z = self.reparameterize(dist)
        recon_x = self.decode(z)
        
        # compute loss terms 
        loss_recon = F.mse_loss(recon_x, x, reduction='none').sum(-1).sqrt()#.mean()
        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1),
        )
        loss_kl = torch.distributions.kl.kl_divergence(dist, std_normal)#.mean()
        # loss_kl = F.kl_div(z, lat_x, reduction='none').mean()
                
        loss = loss_recon + loss_kl
        
        return loss.mean(),loss_recon