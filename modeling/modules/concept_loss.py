# Concept Tokenizer note: Concept Guidance loss code for this project; retains GigaTok-related loss naming where present.
"""This file contains code for concept guidance loss and GigaTok-related loss components.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

_CLIP_MEAN = [
    0.48145466,
    0.4578275,
    0.40821073
  ]
_CLIP_STD = [
    0.26862954,
    0.26130258,
    0.27577711
  ]
_SIGLIP_MEAN = [
    0.5,
    0.5,
    0.5
  ]
_SIGLIP_STD = [
    0.5,
    0.5,
    0.5
  ]

class LCOS(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.clip = AutoModel.from_pretrained(model_name).vision_model.eval()
        for param in self.parameters():
            param.requires_grad = False
        if 'clip' in model_name:    # clip-vit-base-patch16
            self.layer = 11
            self.exist_cls = True
            self.scaling_layer = ScalingLayer(224, 'clip')
        elif 'siglip' in model_name:
            self.layer = 12 
            self.exist_cls = False
            self.scaling_layer = ScalingLayer(256, 'siglip')
        else:
            raise ValueError(f"SAE model name {model_name} not supported")

    def forward(self, ori_images, prediction):
        ori_images = self.scaling_layer(ori_images)
        with torch.no_grad():
            self.clip.eval()
            image_tokens = self.clip(pixel_values=ori_images, output_hidden_states=True).hidden_states[self.layer]
            if self.exist_cls:
                image_tokens = image_tokens[:,0,:]   # [B, 768]
            else:
                image_tokens = image_tokens.mean(1)   # [B,768]
        
        cos_sim = torch.nn.functional.cosine_similarity(image_tokens, prediction, dim=1)
        loss = (1-cos_sim).mean()
        return loss
    
    def eval_pred(self, ori_images):
        ori_images = self.scaling_layer(ori_images)
        image_tokens = self.clip(pixel_values=ori_images, output_hidden_states=True).hidden_states[self.layer]
        if self.exist_cls:
            image_tokens = image_tokens[:,0,:]   # [B, 1, 768]
        else:
            image_tokens = image_tokens.mean(1)   # [B, 1, 768]
        return image_tokens

class LCL(nn.Module):
    def __init__(self, model_name, dirs, topk=50):
        super().__init__()
        
        sae_clip = torch.load(dirs,  map_location="cpu")
        self.clip = AutoModel.from_pretrained(model_name).vision_model.eval()
        self.sae_W_enc = nn.Parameter(sae_clip['W_enc'])
        self.sae_b_enc = nn.Parameter(sae_clip['b_enc'])
        self.sae_b_dec = nn.Parameter(sae_clip['b_dec'])
        self.topk = topk
        for param in self.parameters():
            param.requires_grad = False
        if 'clip' in model_name:    # clip-vit-base-patch16
            self.layer = 11
            self.exist_cls = True
            self.scaling_layer = ScalingLayer(224, 'clip')
        elif 'siglip' in model_name:
            self.layer = 12 
            self.exist_cls = False
            self.scaling_layer = ScalingLayer(256, 'siglip')
        else:
            raise ValueError(f"SAE model name {model_name} not supported")
 
    def eval_pred(self, ori_images, prediction, return_clip_tokens=False):
        ori_images = self.scaling_layer(ori_images)
        image_tokens = self.clip(pixel_values=ori_images, output_hidden_states=True).hidden_states[self.layer]
        if self.exist_cls:
            image_tokens = image_tokens[:,0,:]   # [B, 1, 768]
        else:
            image_tokens = image_tokens.mean(1)   # [B, 1, 768]
        sae_in = image_tokens - self.sae_b_dec[None,:]
        sae_hidden = sae_in @ self.sae_W_enc+self.sae_b_enc[None,:]
        # top-k index
        # topk_idx = torch.topk(sae_hidden, self.topk, dim=1).indices  # [B, topk]
       
        if return_clip_tokens:
            return image_tokens
        else:
            return prediction, sae_hidden


    def forward(self, ori_images, prediction):
        ori_images = self.scaling_layer(ori_images)
        with torch.no_grad():
            self.clip.eval()
            image_tokens = self.clip(pixel_values=ori_images, output_hidden_states=True).hidden_states[self.layer]
            if self.exist_cls:
                image_tokens = image_tokens[:,0,:]   # [B, 1, 768]
            else:
                image_tokens = image_tokens.mean(1)   # [B, 1, 768]
            sae_in = image_tokens - self.sae_b_dec[None,:]
            sae_hidden = sae_in @ self.sae_W_enc+self.sae_b_enc[None,:]
            # top-k index
            topk_idx = torch.topk(sae_hidden, self.topk, dim=1).indices  # [B, topk]
            # smoothed label
            smoothed_labels = torch.zeros_like(prediction)  # [B, num_classes]
            smoothed_labels.scatter_(1, topk_idx, 1.0 / self.topk)
        # log_softmax
        log_prob = torch.nn.functional.log_softmax(prediction, dim=1)  # [B, num_classes]
        loss = -torch.sum(log_prob * smoothed_labels, dim=1).mean()
        return loss

 

class PCL(nn.Module):
    # Learned perceptual metric.
    def __init__(self, model_name, dirs, topk=50):
        super().__init__()
 
        sae_clip = torch.load(dirs,  map_location="cpu")
        self.clip = AutoModel.from_pretrained(model_name).vision_model.eval()
        self.sae_W_enc = nn.Parameter(sae_clip['W_enc'])
        self.sae_b_enc = nn.Parameter(sae_clip['b_enc'])
        self.sae_b_dec = nn.Parameter(sae_clip['b_dec'])
        self.topk = topk

        for param in self.parameters():
            param.requires_grad = False
        if 'clip' in model_name:    # clip-vit-base-patch16
            self.layer = 11
            self.exist_cls = True
            self.scaling_layer = ScalingLayer(224, 'clip')
        elif 'siglip' in model_name:
            self.layer = 12 
            self.exist_cls = False
            self.scaling_layer = ScalingLayer(256, 'siglip')    
        else:
            raise ValueError(f"SAE model name {model_name} not supported")

        for param in self.parameters():
            param.requires_grad = False
 
  
    def eval_pred(self, ori_images, prediction):
        ori_images = self.scaling_layer(ori_images)
        with torch.no_grad():
            self.clip.eval()
            image_tokens = self.clip(pixel_values=ori_images, output_hidden_states=True).hidden_states[self.layer]
        if self.exist_cls:
            image_tokens = image_tokens[:,1:,:] 
        
        # Reshape image_tokens to 2D spatial format and resize to match prediction
        B, N, C = image_tokens.shape  # N = D (number of patches)
        B_pred, D_pred, L_pred = prediction.shape

        if N != D_pred:
            D = int(N ** 0.5)  # Assuming N is a perfect square
        
            # Reshape to [B, D, D, C] for spatial format
            image_tokens_2d = image_tokens.view(B, D, D, C)
        
            # Get target spatial dimensions from prediction
            target_size = int(D_pred ** 0.5)  # Assuming D_pred is a perfect square
            
            # Resize to match prediction spatial dimensions
            image_tokens_resized = nn.functional.interpolate(
                image_tokens_2d.permute(0, 3, 1, 2),  # [B, C, D, D]
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)  # [B, target_size, target_size, C]
            
            # Flatten back to [B, target_size^2, C]
            image_tokens = image_tokens_resized.view(B, target_size**2, C)
        
        sae_in = image_tokens - self.sae_b_dec[None,None,:]
        sae_hidden = sae_in @ self.sae_W_enc+self.sae_b_enc[None,None,:]  # [B, D_pred, L]
       
        return prediction, sae_hidden



    def forward(self, ori_images, prediction):
 
        ori_images = self.scaling_layer(ori_images)
        with torch.no_grad():
            self.clip.eval()
            image_tokens = self.clip(pixel_values=ori_images, output_hidden_states=True).hidden_states[self.layer]
        if self.exist_cls:
            image_tokens = image_tokens[:,1:,:] 
        
        # Reshape image_tokens to 2D spatial format and resize to match prediction
        B, N, C = image_tokens.shape  # N = D (number of patches)
        B_pred, D_pred, L_pred = prediction.shape

        if N != D_pred:
            D = int(N ** 0.5)  # Assuming N is a perfect square
        
            # Reshape to [B, D, D, C] for spatial format
            image_tokens_2d = image_tokens.view(B, D, D, C)
        
            # Get target spatial dimensions from prediction
            target_size = int(D_pred ** 0.5)  # Assuming D_pred is a perfect square
            
            # Resize to match prediction spatial dimensions
            image_tokens_resized = nn.functional.interpolate(
                image_tokens_2d.permute(0, 3, 1, 2),  # [B, C, D, D]
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)  # [B, target_size, target_size, C]
            
            # Flatten back to [B, target_size^2, C]
            image_tokens = image_tokens_resized.view(B, target_size**2, C)
        
        sae_in = image_tokens - self.sae_b_dec[None,None,:]
        sae_hidden = sae_in @ self.sae_W_enc+self.sae_b_enc[None,None,:]  # [B, D_pred, L]
        
        # top-k index along L dimension
        _, topk_idx = torch.topk(sae_hidden, self.topk, dim=2)  # [B, D_pred, topk]
        
        # smoothed label - create labels for each patch (D_pred dimension)
        B, D_pred, L = sae_hidden.shape
        smoothed_labels = torch.zeros(B, D_pred, L, device=prediction.device)  # [B, D_pred, L]
        
        # Scatter top-k values to create smoothed labels
        batch_indices = torch.arange(B, device=prediction.device)[:, None, None].expand(-1, D_pred, self.topk)
        dim_indices = torch.arange(D_pred, device=prediction.device)[None, :, None].expand(B, -1, self.topk)
        smoothed_labels[batch_indices, dim_indices, topk_idx] = 1.0 / self.topk
        
        # log_softmax along L dimension
        log_prob = torch.nn.functional.log_softmax(prediction, dim=2)  # [B, D_pred, L]
        
        # Calculate loss: negative log likelihood
        loss = -torch.sum(log_prob * smoothed_labels, dim=2).mean()
        return loss

class ScalingLayer(nn.Module):
    def __init__(self,image_size=224, model_name='clip'):
        super(ScalingLayer, self).__init__()
        if model_name == 'clip':
            self.register_buffer("shift", torch.Tensor(_CLIP_MEAN)[None, :, None, None])
            self.register_buffer("scale", torch.Tensor(_CLIP_STD)[None, :, None, None])
        elif model_name == 'siglip':
            self.register_buffer("shift", torch.Tensor(_SIGLIP_MEAN)[None, :, None, None])
            self.register_buffer("scale", torch.Tensor(_SIGLIP_STD)[None, :, None, None])
        else:
            raise ValueError(f"SAE model name {model_name} not supported")
        self.image_size = image_size

    def forward(self, inp):
         
        # Resize input to 224x224
        if self.image_size != inp.size(2):
            inp = nn.functional.interpolate(inp, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return (inp - self.shift) / self.scale
