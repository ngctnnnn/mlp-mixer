import tensorflow as tf
import jax.numpy as jnp
import torch, torchvision
import jax, jit
import haiku as hk

class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size
    
    def call(self, images):
        batch_size = images.shape[0]
        patches = tf.image.extract_patches(
            images = images,
            sizes = [1, self.patch_size, self.patch_size, 1],
            strides = [1, self.patch_size, self.patch_size, 1],
            rates = [1, 1, 1, 1],
            padding = 'VALID'
        )
        dim = patches.shape[-1]
        patches = jnp.reshape(patches, (batch_size, -1, dim))
        return patches
    
class MLPBlock(torch.nn.Module):
    def __init__(self, S, C, DS, DC):
        """
        Arguments
        S -- a sequence of S non-overlapping image patches
        C -- a desired hidden dimension projects each image patch
        DS -- tunable hidden widths in the token-mixing MLPs
        DC -- tunable hidden widths in the channel-mixing MLPs
        """
        self.layerNorm1 = jax.nn.normalize()
        self.layerNorm2 = jax.nn.normalize()
        self.DS = DS 
        self.DC = DC
        self.W1 = torch.randn(S, DS, dtype=torch.float32, requires_grad = True)
        self.W2 = torch.randn(DS, S, dtype=torch.float32, requires_grad = True)
        self.W3 = torch.randn(C, DC, dtype=torch.float32, requires_grad = True)

    @jit
    def forward(self, X, training: bool):
        batch_size, S, C = X.shape
        
        """ 
        Token mixing
        """
        X_T = torch.transpose(self.layerNorm1(X))
        X_T = X_T.perm(0, 2, 1)
        
        W1X = X_T @ self.W1 
        
        U = torch.transpose(jax.nn.gelu(W1X) @ self.W2)
        U = U.perm(0, 2, 1) + X 
        
        """
        Channel mixing
        """
        W3U = self.layerNorm2(U) @ self.W3 
        Y = (jax.nn.gelu(W3U) @ self.W4) + U
        
        return Y 

class MLPMixer(torch.nn.Module):
    def __init__(self, patch_size, S, C, DS, DC, num_mlp_blocks, image_size, batch_size, num_classes):
        super(MLPMixer, self).__init__()
        self.projection = hk.Linear(C)
        self.mlpBlocks = [MLPBlock(S, C, DS, DC) for _ in range(num_mlp_blocks)]
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.S = S 
        self.C = C
        self.DS = DS
        self.DC = DC 
        self.image_size = image_size 
        self.num_classes = num_classes 
        self.data_augmentation = torch.jit.script(torch.nn.Sequential(
            torchvision.transforms.Normalization(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(degrees=0.02),
        ))
        self.classificationLayer = torch.nn.Sequential(
            hk.AvgPool(),
            hk.dropout(rate=0.2),
            hk.linear(num_classes),
            jax.nn.softmax()
        )
    
    def extract_patches(self, images, patch_size):
        batch_size = images.shape[0]
        patches = tf.image.extract_patches(
            images = images,
            sizes = [1, patch_size, patch_size, 1],
            strides = [1, patch_size, patch_size, 1],
            rates = [1, 1, 1, 1],
            padding = "VALID",
        )
        patches = jnp.reshape(patches, [batch_size, -1, 3 * patch_size ** 2])
        return patches 

    def call(self, images):
        batch_size = images.shape[0]
        augmented_images = self.data_augmentation(images)
        X = self.extract_patches(augmented_images, self.patch_size)
        X = self.projection(X)

        for block in self.mlpBlocks:
            X = block(X)
        
        out = self.classificationLayer(X)
        return out