import os
import torch
import torch.nn.functional as F
from typing import Optional
from transformers import CLIPModel, CLIPProcessor
from .backbone.vision import VisionProj
from .model import VV

class VVvlm(VV):
    def __init__(self, config: VVConfig = None):
        # Ensure config has necessary attributes for VV
        if not hasattr(config, 'hidden_dim') and hasattr(config, 'hidden_size'):
            config.hidden_dim = config.hidden_size
            
        super().__init__(config)
        self.config = config
        
        # Vision Projection
        ve_hidden_size = getattr(config, 've_hidden_size', 768)
        self.vision_proj = VisionProj(ve_hidden_size=ve_hidden_size, hidden_size=config.hidden_dim)
        
        # Load Vision Model
        vision_model_path = getattr(config, 'vision_model_path', "./model/vision_model/clip-vit-base-patch16")
        self.vision_encoder, self.processor = self.get_vision_model(vision_model_path)

    @staticmethod
    def get_vision_model(model_path: str):
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        if not os.path.exists(model_path):
            # Try to load from default location or return None/Warning
            print(f"Warning: Vision model path {model_path} does not exist.")
            return None, None
        
        try:
            model = CLIPModel.from_pretrained(model_path)
            processor = CLIPProcessor.from_pretrained(model_path)
            # Freeze vision encoder parameters
            for param in model.parameters():
                param.requires_grad = False
            return model.eval(), processor
        except Exception as e:
            print(f"Error loading vision model: {e}")
            return None, None

    @staticmethod
    def image2tensor(image, processor):
        if processor is None:
            return None
        if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")['pixel_values']
        return inputs

    @staticmethod
    def get_image_embeddings(image_tensors, vision_model):
        if vision_model is None:
            return None
        with torch.no_grad():
            outputs = vision_model.vision_model(pixel_values=image_tensors)
        img_embedding = outputs.last_hidden_state[:, 1:, :].squeeze()
        return img_embedding

    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
        if vision_tensors is None:
            return h
            
        def find_indices(tokens, image_ids):
            # Handle list or single int image_ids
            if isinstance(image_ids, int):
                image_ids = [image_ids]
                
            image_ids_tensor = torch.tensor(image_ids).to(tokens.device)
            len_image_ids = len(image_ids)
            if len_image_ids > tokens.size(1):
                return None
            tokens_view = tokens.unfold(1, len_image_ids, 1)
            matches = (tokens_view == image_ids_tensor).all(dim=2)
            return {
                batch_idx: [(idx.item(), idx.item() + len_image_ids - 1) for idx in
                            matches[batch_idx].nonzero(as_tuple=True)[0]]
                for batch_idx in range(tokens.size(0)) if matches[batch_idx].any()
            } or None

        image_ids = getattr(self.config, 'image_ids', [])
        if not image_ids:
            return h
            
        image_indices = find_indices(tokens, image_ids)
        
        if vision_tensors is not None and image_indices:
            vision_proj = self.vision_proj(vision_tensors)
            if len(vision_proj.shape) == 3:
                vision_proj = vision_proj.unsqueeze(0)
            new_h = []
            for i in range(h.size(0)):
                if i in image_indices:
                    h_i = h[i]
                    img_idx = 0
                    for start_idx, end_idx in image_indices[i]:
                        if img_idx < vision_proj.size(1):
                            vp = vision_proj[i][img_idx]
                            if vp.dim() == 1:
                                vp = vp.unsqueeze(0) # (1, hidden_dim)
                            
                            # Concatenate
                            h_i = torch.cat((h_i[:start_idx], vp, h_i[end_idx + 1:]), dim=0)
                            img_idx += 1
                    
                    # Truncate to original seqlen if needed
                    if h_i.size(0) > seqlen:
                        h_i = h_i[:seqlen]
                    new_h.append(h_i)
                else:
                    new_h.append(h[i])
            return torch.stack(new_h, dim=0)
        return h

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.FloatTensor] = None,
                **kwargs): 
        batch_size, seq_length = input_ids.shape
        hidden_states = self.token_embedding_table(input_ids)
        if pixel_values is not None:
            if len(pixel_values.shape) == 6:
                pixel_values = pixel_values.squeeze(2)
            
            # Check if we have vision encoder loaded
            if self.vision_encoder is not None:
                bs, num, c, im_h, im_w = pixel_values.shape
                stack_dim = 1 if bs > 1 else 0
                
                vision_tensors = torch.stack([
                    self.get_image_embeddings(pixel_values[:, i, :, :, :], self.vision_encoder)
                    for i in range(num)
                ], dim=stack_dim)
                
                hidden_states = self.count_vision_proj(tokens=input_ids, h=hidden_states, vision_tensors=vision_tensors,
                                                       seqlen=seq_length)

        hidden_states = self.blocks(hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
        return (loss, logits)


