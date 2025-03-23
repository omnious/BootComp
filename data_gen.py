import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from src.decompose_pipeline_xl import StableDiffusionXLPipeline as DecomposePipeline
import torchvision.transforms as T
import torch.nn as nn
import torchvision
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.nn.functional as F



def get_bbox_from_mask(mask: torch.Tensor):

    # Determine which rows and columns contain any non-zero values
    rows = torch.any(mask, dim=1)
    cols = torch.any(mask, dim=0)
    
    if not torch.any(rows) or not torch.any(cols):
        return None
    
    # Get the first and last indices where mask is non-zero
    ymin, ymax = torch.where(rows)[0][[0, -1]]
    xmin, xmax = torch.where(cols)[0][[0, -1]]
    
    return xmin.item(), ymin.item(), xmax.item(), ymax.item()

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path",type=str,default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--unet_ckpt",type=str,default="omniousai/BootComp")
    parser.add_argument("--pretrained_vae_path",type=str,default="madebyollin/sdxl-vae-fp16-fix")
    parser.add_argument("--pretrained_seg_model_path",type=str,default="mattmdjaga/segformer_b2_clothes")
    parser.add_argument("--width",type=int,default=384)
    parser.add_argument("--height",type=int,default=512)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--human_img_path", type=str,)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--num_inference_steps", type=int, default=20)

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():

    args = parse_args()
    accelerator = Accelerator()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(os.path.join(args.output_dir,"segment"), exist_ok=True)
            os.makedirs(os.path.join(args.output_dir,"product"), exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler",rescale_betas_zero_snr=True)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(args.pretrained_vae_path)
    unet = UNet2DConditionModel.from_pretrained(args.unet_ckpt, subfolder="decomp")


    def custom_transformer_block_forward(self,
                                         hidden_states,
                                         attention_mask=None,
                                         encoder_hidden_states=None,
                                         encoder_attention_mask=None,
                                         timestep=None,
                                         cross_attention_kwargs=None,
                                         class_labels=None,
                                         added_cond_kwargs=None):
        
        # 0. Self-Attention


        norm_hidden_states = self.norm1(hidden_states)



        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )


        hidden_states = attn_output + hidden_states


        # 3. Cross-Attention
        if self.attn2 is not None:
            dup_bsz,N,L =hidden_states.shape
            flatted_bsz = dup_bsz // (1+1)
            hidden_states_dup = hidden_states.reshape(flatted_bsz,(1+1)*N, L)
            hidden_states, hidden_states_ref = hidden_states_dup[:, :N, :], hidden_states_dup[:, N:, :]

            norm_hidden_states = self.norm2(hidden_states)
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

            hidden_states = torch.cat([hidden_states,hidden_states_ref], dim=1)
            hidden_states = hidden_states.reshape(flatted_bsz*(1+1), N, L)


        norm_hidden_states = self.norm3(hidden_states)



        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states

        return hidden_states



    def override_instance_call(instance, new_call):
        # 새로운 서브클래스를 정의합니다.
        class CustomCallClass(instance.__class__):
            def __call__(self, *args, **kwargs):
                return new_call(self, *args, **kwargs)
        
        # 해당 인스턴스의 클래스를 새로운 서브클래스로 변경합니다.
        instance.__class__ = CustomCallClass


    def extended_self_attn_call(
        self,
        attn,
        hidden_states,
        encoder_hidden_states= None,
        attention_mask= None,
        temb = None,
        *args,
        **kwargs,):

            dup_bsz,N,L =hidden_states.shape
            #one for noisy latent and one for clean reference garment
            flatted_bsz = dup_bsz // (1+1)
            hidden_states_dup = hidden_states.reshape(flatted_bsz,(1+1)*N, L)
            hidden_states, hidden_states_ref = hidden_states_dup[:, :N, :], hidden_states_dup[:, N:, :]

            batch_size, sequence_length, _ = (
                hidden_states.shape
            )

            query = attn.to_q(hidden_states)

            encoder_hidden_states = hidden_states

            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            query_ref = attn.to_q(hidden_states_ref)
            key_ref = attn.to_k(hidden_states_ref)
            value_ref = attn.to_v(hidden_states_ref)


            key = torch.cat([key, key_ref], dim=1)
            value = torch.cat([value, value_ref], dim=1)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            query_ref = query_ref.view(flatted_bsz, -1, attn.heads, head_dim).transpose(1, 2)
            key_ref = key_ref.view(flatted_bsz, -1, attn.heads, head_dim).transpose(1, 2)
            value_ref = value_ref.view(flatted_bsz, -1, attn.heads, head_dim).transpose(1, 2)

            hidden_states_ref = F.scaled_dot_product_attention(query_ref, key_ref, value_ref,attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
            hidden_states_ref = hidden_states_ref.transpose(1, 2).reshape(flatted_bsz, -1, attn.heads * head_dim)
            hidden_states_ref = hidden_states_ref.to(query.dtype)
            hidden_states_ref = hidden_states_ref.reshape(flatted_bsz,N, L)


            hidden_states = torch.cat([hidden_states,hidden_states_ref], dim=1)


            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            hidden_states = hidden_states / attn.rescale_output_factor
            hidden_states = hidden_states.reshape(flatted_bsz*(1+1), N, L)


            return hidden_states             


    #override forward
    for module in unet.modules():
        import types
        if isinstance(module, BasicTransformerBlock):
            module.forward = types.MethodType(custom_transformer_block_forward, module)

    #override attn call
    for name, proc in unet.attn_processors.items():
        if name.endswith("attn1.processor"):            
            override_instance_call(proc, extended_self_attn_call)

    # Load segmentation model
    seg_processor =  SegformerImageProcessor.from_pretrained(args.pretrained_seg_model_path)
    seg_model = AutoModelForSemanticSegmentation.from_pretrained(args.pretrained_seg_model_path)


    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)
    seg_model.requires_grad_(False)


    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    
    
    seg_model = accelerator.prepare(seg_model)


    with torch.no_grad() and torch.cuda.amp.autocast():

        newpipe = DecomposePipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet,
            vae= vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            scheduler=noise_scheduler,
            torch_dtype=torch.float16,
            add_watermarker=False,
            safety_checker=None,
        ).to(accelerator.device)


        image = Image.open(args.human_img_path).convert("RGB").resize((args.width,args.height))
        tensor_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ]
                )
        image_tensor = tensor_transform(image).unsqueeze(0).to(accelerator.device, dtype=weight_dtype)


        #human parsing
        inputs = seg_processor(images=image, return_tensors="pt")
        inputs_tensor = inputs.pixel_values[0].to(accelerator.device, dtype=weight_dtype)
        outputs = seg_model(inputs_tensor.unsqueeze(0))
        logits = outputs.logits
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=(args.height, args.width),
            mode="bilinear",
            align_corners=False,
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0]
        

        label_map = {
        1: "Hat",
        3: "Sunglasses",
        4: "Upper garment",
        5: "Skirt",
        6: "Pants",
        7: "Dress",
        (9,10) : "Shoes",
        16: "Bag",
        17: "Scarf"}

        prompt_list = []
        category_list = []  
        garm_tensor_list = []
        for category_idx in label_map.keys():
            if isinstance(category_idx, tuple):
                parse_mask = (pred_seg == category_idx[0]) + (pred_seg == category_idx[1])
            else:
                parse_mask = (pred_seg == category_idx)

            if parse_mask.sum() == 0:
                continue
                
            prompt_list.append("A product photo of " + label_map[category_idx])
            category_list.append(label_map[category_idx])
            # Apply the mask to the image tensor
            masked_img = image_tensor * parse_mask

            # Obtain the bounding box.
            bbox = get_bbox_from_mask(parse_mask)
            
            # If no bounding box is found, use the full image dimensions.
            if bbox is None:
                x_min, y_min, x_max, y_max = 0, 0, args.width, args.height
            else:
                x_min, y_min, x_max, y_max = bbox
            # Determine image dimensions and current bounding box size.
            H, W = masked_img.shape[-2], masked_img.shape[-1]
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            
            # Adjust the bounding box to match the target aspect ratio
            target_aspect_ratio = args.width / args.height
            current_aspect_ratio = width / height

            if current_aspect_ratio > target_aspect_ratio:
                # Bounding box is too wide; adjust height.
                new_height = int(width / target_aspect_ratio)
                height_diff = new_height - height
                y_min_adjustment = y_min if (y_min - height_diff // 2 < 0) else (height_diff // 2)
                y_max_adjustment = height_diff - y_min_adjustment
                y_min = max(0, y_min - y_min_adjustment)
                y_max = min(H - 1, y_max + y_max_adjustment)
            else:
                # Bounding box is too tall; adjust width.
                new_width = int(height * target_aspect_ratio)
                width_diff = new_width - width
                x_min_adjustment = x_min if (x_min - width_diff // 2 < 0) else (width_diff // 2)
                x_max_adjustment = width_diff - x_min_adjustment
                x_min = max(0, x_min - x_min_adjustment)
                x_max = min(W - 1, x_max + x_max_adjustment)

            # Crop the masked image based on the adjusted bounding box.
            cropped_image = masked_img[:, :, y_min:y_max+1, x_min:x_max+1]

            # Resize the cropped image to the original masked image dimensions.
            resize_transform = T.Resize((masked_img.shape[-2], masked_img.shape[-1]))
            resized_image = resize_transform(cropped_image)
            
            # Add the processed image to the list.
            garm_tensor_list.append(resized_image)
            
        if len(garm_tensor_list) == 0:
            #print error message and exit the run
            print("No garment found in the image")
            return

        garments_tensor = torch.cat(garm_tensor_list, dim=0).unsqueeze(0)
        images = newpipe(
            prompt=prompt_list,
            num_inference_steps=args.num_inference_steps,
            garment_segment=garments_tensor,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
        )[0]

        img_name,img_extension = os.path.splitext(os.path.basename(args.human_img_path))
        for i,img in enumerate(images):
            each_img_name = img_name + f"_{category_list[i]}" + img_extension
            product_img_path = os.path.join(args.output_dir,"product",each_img_name)
            seg_img_path = os.path.join(args.output_dir,"segment",each_img_name)
            img.save(product_img_path)
            torchvision.utils.save_image((garments_tensor[0][i]+1.0)/2.0, seg_img_path)
                
if __name__ == "__main__":
    main()    
