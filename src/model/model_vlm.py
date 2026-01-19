import os
import torch
import torch.nn.functional as F
from typing import Optional
from transformers import CLIPVisionModel, CLIPImageProcessor
from .backbone.vision import VisionProjector
from .model import VV

class VisualVV(VV):
    def __init__(self, config=None, freeze_llm=True):
        super().__init__(config)
        if freeze_llm:
            for param in self.parameters():
                param.requires_grad = False
            print(f"[Model] 已冻结 LLM 参数，仅训练 Vision Projector")
        # Vision Projector (将视觉特征映射到文本维度)
        self.projector = VisionProjector(vision_hidden_dim=config.vision_hidden_dim, hidden_size=config.hidden_dim)
        self.vision_encoder, self.processor = self.get_vision_model(config.vision_model_path)
        self.apply(self._init_weights)

    @staticmethod
    def get_vision_model(model_path: str):
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        if not os.path.exists(model_path):
            return None, None
        model = CLIPVisionModel.from_pretrained(model_path)
        processor = CLIPImageProcessor.from_pretrained(model_path)
        # 冻结 vision_encoder 的所有参数
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    @staticmethod
    def image2tensor(image, processor):
        if image.mode != 'RGB': image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")['pixel_values']
        return inputs

    @staticmethod
    def get_image_embeddings(image_tensors, vision_model):
        with torch.no_grad():
            outputs = vision_model(pixel_values=image_tensors)
        img_embedding = outputs.last_hidden_state[:, 1:, :].squeeze()
        return img_embedding

    def inject_visual_embeddings(self, tokens, hidden_states, vision_tensors=None, seqlen=512):
        ''' 将视觉特征嵌入到文本隐藏状态中
        Args:
            tokens: 输入 token 序列, shape (batch_size, seq_len)
            hidden_states: 文本隐藏状态, shape (batch_size, seq_len, hidden_dim)
            vision_tensors: 视觉特征, shape (batch_size, num_patches, vision_hidden_dim)
            seqlen: 最大序列长度
        Returns:
            new_hidden_states: 融合视觉特征后的隐藏状态
        '''
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

        image_indices = find_indices(tokens, self.config.image_ids)
        if vision_tensors is not None and image_indices:
            vision_proj = self.projector(vision_tensors)
            if len(vision_proj.shape) == 3:
                vision_proj = vision_proj.unsqueeze(0)
            new_h = []
            for i in range(hidden_states.size(0)):
                if i in image_indices:
                    hidden_states_i = hidden_states[i]
                    # 修改：倒序遍历以处理多图插入时的索引偏移问题
                    # 必须倒序，因为插入操作会改变后续元素的索引
                    current_indices = image_indices[i]
                    img_idx = len(current_indices) - 1
                    for start_idx, end_idx in reversed(current_indices):
                        if img_idx < vision_proj.size(1) and img_idx >= 0:
                            vp = vision_proj[i][img_idx]
                            if vp.dim() == 1:
                                vp = vp.unsqueeze(0) # (1, hidden_dim)
                            # 注意：如果 vp 的长度（图像 patch 数）不等于占位符长度 (end_idx - start_idx + 1)，
                            # 替换后序列长度会发生变化。这会导致后续文本与 Labels 错位（除非 Labels 也做了相应调整）。
                            # 强烈建议：输入文本中的 image_ids 占位符数量应严格等于图像 patch 数量。  
                            hidden_states_i = torch.cat((hidden_states_i[:start_idx], vp, hidden_states_i[end_idx + 1:]), dim=0)
                            img_idx -= 1
                    if hidden_states_i.size(0) > seqlen:
                        hidden_states_i = hidden_states_i[:seqlen]
                    new_h.append(hidden_states_i)
                else:
                    new_h.append(hidden_states[i])
            return torch.stack(new_h, dim=0)
        return hidden_states

    def _apply_vision_embeddings(self, input_ids, hidden_states, pixel_values, seq_length):
        """
        处理视觉输入并将图像嵌入融合到文本隐藏状态中
        """
        if pixel_values is not None:
            # 确保内存连续性，防止 CUDA Error
            pixel_values = pixel_values.contiguous()
            
            if len(pixel_values.shape) == 6:
                pixel_values = pixel_values.squeeze(2)
            
            # pixel_values shape: (bs, num_images, c, h, w)
            bs, num, c, im_h, im_w = pixel_values.shape
            
            # 将 batch 和 image 维度合并，一次性通过 vision_encoder，极大提高利用率
            # (bs * num, c, im_h, im_w)
            flat_pixel_values = pixel_values.view(bs * num, c, im_h, im_w)
            
            try:
                # 获取所有图片的 embedding
                # shape: (bs * num, patches, vision_hidden_dim)
                with torch.no_grad():
                    outputs = self.vision_encoder(pixel_values=flat_pixel_values)
                    # 去掉 CLS token (index 0)
                    vision_tensors = outputs.last_hidden_state[:, 1:, :]
                
                # 恢复成 (bs, num, patches, vision_hidden_dim)
                vision_tensors = vision_tensors.view(bs, num, vision_tensors.size(1), vision_tensors.size(2))
                
            except Exception as e:
                print(f"[ERROR] Vision encoder forward failed: {e}")
                raise e

            hidden_states = self.inject_visual_embeddings(tokens=input_ids, hidden_states=hidden_states, vision_tensors=vision_tensors, seqlen=seq_length)

        return hidden_states

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.FloatTensor] = None,
                **kwargs): 
        
        # --- Debugging: Check input_ids for out-of-bounds values ---
        if input_ids is not None:
            # We use a non-blocking check first to avoid perf hit if possible, 
            # but for debugging we force CPU sync.
            try:
                # Check max value on CPU to avoid device-side assert crashing the process without info
                input_ids_cpu = input_ids.detach().cpu()
                max_val = input_ids_cpu.max().item()
                min_val = input_ids_cpu.min().item()
                
                # Double check embedding weight shape
                if hasattr(self, 'token_embedding_table'):
                    vocab_dim = self.token_embedding_table.weight.shape[0]
                    if vocab_dim != self.config.vocab_size:
                        print(f"\n[FATAL ERROR] Embedding size mismatch! Weight: {vocab_dim}, Config: {self.config.vocab_size}")
                        raise ValueError(f"Embedding mismatch: {vocab_dim} vs {self.config.vocab_size}")
                    if max_val >= vocab_dim:
                         print(f"\n[FATAL ERROR] Input ID {max_val} >= Embedding size {vocab_dim}")
                         bad_indices = (input_ids_cpu >= vocab_dim).nonzero(as_tuple=False)
                         print(f"Bad indices (first 5): {bad_indices[:5]}")
                         print(f"Bad values: {input_ids_cpu[bad_indices[:5][:, 0], bad_indices[:5][:, 1]]}")
                         raise ValueError(f"Input ID out of bounds: {max_val} >= {vocab_dim}")

                if max_val >= self.config.vocab_size:
                    print(f"\n[FATAL ERROR] Found input_id {max_val} >= vocab_size {self.config.vocab_size}")
                    # Find specific indices
                    bad_indices = (input_ids_cpu >= self.config.vocab_size).nonzero(as_tuple=False)
                    print(f"Bad indices (first 5): {bad_indices[:5]}")
                    print(f"Bad values: {input_ids_cpu[bad_indices[:5][:, 0], bad_indices[:5][:, 1]]}")
                    raise ValueError(f"Input ID out of bounds: {max_val} >= {self.config.vocab_size}")
                
                if min_val < 0:
                    print(f"\n[FATAL ERROR] Found negative input_id {min_val}")
                    raise ValueError(f"Input ID negative: {min_val}")

                if labels is not None:
                     labels_cpu = labels.detach().cpu()
                     # Ignore -100
                     mask = labels_cpu != -100
                     if mask.any():
                         max_label = labels_cpu[mask].max().item()
                         if max_label >= self.config.vocab_size:
                             print(f"\n[FATAL ERROR] Found label {max_label} >= vocab_size {self.config.vocab_size}")
                             raise ValueError(f"Label out of bounds: {max_label}")
                     
            except Exception as e:
                print(f"[Debug Check Failed] {e}")
                raise e
        # -----------------------------------------------------------

        batch_size, seq_length = input_ids.shape
        x = self.token_embedding_table(input_ids)

        # 融合视觉嵌入
        x = self._apply_vision_embeddings(input_ids, x, pixel_values, seq_length)
        
        x = self.blocks(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            logits = logits.reshape(-1, logits.size(-1))
            targets = labels.reshape(-1)
            loss = F.cross_entropy(logits, targets, ignore_index=-100)
        return (loss, logits)


