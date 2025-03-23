import os
import argparse
import json
import torch
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from src.compose_pipeline_xl import StableDiffusionXLPipeline as ComposePipeline
from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_encoder
from PIL import ImageFile
from typing import Literal, Tuple
from accelerate import Accelerator
import torch.utils.data as data
from collections import defaultdict
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True


class GarmentDataset(data.Dataset):
    def __init__(
        self,
        phase: Literal["train", "test"],
        info_path: str,
        size: Tuple[int, int] = (512, 384),      
    ):
        super(GarmentDataset, self).__init__()
        
        self.phase = phase
        self.height = size[0]
        self.width = size[1]
        self.size = size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.toTensor = transforms.ToTensor()
        self.info_path =info_path
        iterate_dict = defaultdict(list)


        with open(os.path.join(self.info_path), 'r') as f:
            data = json.load(f)

        for idx_name, wearing_dict in data.items():
            ref_list =[]
            #pop the text prompt from dictionary
            text_prompt = wearing_dict.pop("text")
            for garm_cat, garm_path in wearing_dict.items():
                if os.path.exists(garm_path) == False:
                    continue
                ref_list.append({"path":garm_path,"category":garm_cat})
                
            iterate_dict[len(ref_list)].append( {"index_name":idx_name+".jpg", "text_prompt": text_prompt, "ref_list":ref_list})

        filtered_list = []
        for key,val in iterate_dict.items():
            # #discard that the remainder of elements in val when divide it with batch_size
            # if len(val) < batch_size:
            #     continue
            # val = val[:len(val)//batch_size*batch_size]
            filtered_list+=val

        self.garmset_list = filtered_list        
        self.caption_dict = {}


        print("all images len: ",len(self.garmset_list))




    def __getitem__(self, index):
        

        info_dict = self.garmset_list[index]
        #info_dict has one key
        ref_img_list = []
        ref_category_list = []
        

        save_name = info_dict["index_name"]
            
        for ref_dict in info_dict["ref_list"]:
            ref_category_list.append(ref_dict["category"])
            ref_img_list.append(ref_dict["path"])
                

        prompt_composed = info_dict["text_prompt"]
        
        prompt_material = ["A photo of " + text for text in ref_category_list]



        material_img = Image.new('RGB', (self.width,len(ref_img_list)*self.height))     
        y_offset = 0
        for img in ref_img_list:
            ref_img = Image.open(img).resize((self.width, self.height))
            material_img.paste(ref_img, (0, y_offset))
            y_offset += ref_img.height
        material_img = self.transform(material_img)


        result = {}
        result["img_material"] = material_img
        result["caption_composed"] = prompt_composed
        result["caption_material"] = prompt_material
        result["save_name"] = save_name

        return result

    def __len__(self):
        return len(self.garmset_list)



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path",type=str,default="SG161222/RealVisXL_V3.0")
    parser.add_argument("--unet_encoder_ckpt",type=str,default="omniousai/BootComp")
    parser.add_argument("--pretrained_encoder_model_name_or_path",type=str,default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--pretrained_vae_path",type=str,default="madebyollin/sdxl-vae-fp16-fix")
    parser.add_argument("--width",type=int,default=576)
    parser.add_argument("--height",type=int,default=768)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--info_path", type=str, default="./example/info.json")
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--cloth_scale", type=float, default=2.0)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_batch_size", type=int, default=4)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args





def main():


    args = parse_args()

    accelerator = Accelerator(
    mixed_precision = args.mixed_precision    
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(os.path.join(args.output_dir,"vis"), exist_ok=True)
            os.makedirs(os.path.join(args.output_dir,"gen"), exist_ok=True)
            

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler",rescale_betas_zero_snr=True)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(args.pretrained_vae_path)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    unet_encoder = UNet2DConditionModel_encoder.from_pretrained(args.unet_encoder_ckpt, subfolder="comp")



    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)
    unet_encoder.requires_grad_(False)


    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device,dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    unet_encoder.to(accelerator.device, dtype=weight_dtype)
    
    test_dataset = GarmentDataset(
        phase="test",
        info_path = args.info_path,
        size=(args.height, args.width),
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        num_workers=4,
        batch_size = args.test_batch_size,
    )
    
    with torch.cuda.amp.autocast() and torch.no_grad():
        newpipe = ComposePipeline.from_pretrained(
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
        newpipe.unet_encoder = unet_encoder
        generator = torch.Generator(newpipe.device).manual_seed(args.seed) if args.seed is not None else None

        for sample in test_dataloader:

            prompt_composed = sample["caption_composed"]
            prompt_material = [item for sublist in [list(x) for x in zip(*sample["caption_material"])] for item in sublist]
                
            with torch.inference_mode():
                (
                    prompt_embeds_garment,
                    _,
                    pooled_prompt_embeds_garment,
                    _,
                ) = newpipe.encode_prompt(
                    prompt_material,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                                
            images = newpipe(
                prompt=prompt_composed,
                num_inference_steps=args.num_inference_steps,
                img_mat=sample["img_material"].to(accelerator.device, dtype=weight_dtype),
                prompt_ref=prompt_embeds_garment.to(accelerator.device, dtype=weight_dtype),
                pooled_prompt_embeds_ref=pooled_prompt_embeds_garment.to(accelerator.device, dtype=weight_dtype),
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                cloth_scale=args.cloth_scale,
                generator=generator,
            )[0]

            for i in range(len(images)):
                images[i].save(os.path.join(args.output_dir,"gen", sample["save_name"][i]))                                    

                ref_tensors = sample["img_material"][i]
                c,h,w = ref_tensors.shape
                num_ref =int( h//(w *4/3))
                one_h = h//num_ref
                ref_images = []

                for j in range(num_ref):
                    ref = ref_tensors[:, j*one_h:(j+1)*one_h]  # Slice the tensor for the current reference
                    ref = (ref + 1.0) / 2.0  # Normalize to [0, 1] range
                    ref = ref.permute(1, 2, 0).cpu().numpy()  # Rearrange tensor dimensions and convert to numpy
                    ref = (ref * 255).astype(np.uint8)  # Convert to uint8 for saving as an image
                    ref_image = Image.fromarray(ref)  # Convert to PIL image
                    ref_images.append(ref_image)  # Append the image to the list

                concatenated_image = images[i]  # Start with the original image
                for ref_image in ref_images:
                    concatenated_image = Image.fromarray(np.hstack((np.array(concatenated_image), np.array(ref_image))))
                concatenated_image.save(os.path.join(args.output_dir, "vis", sample["save_name"][i]))

                    

                
if __name__ == "__main__":
    main()    
