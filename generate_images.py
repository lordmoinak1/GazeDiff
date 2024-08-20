import torch
from diffusers.utils import load_image
from diffusers import StableDiffusionPipeline, DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionControlNetPipeline, StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler

import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


generator = torch.Generator().manual_seed(3407)

def test_stable_diffusion():
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, use_safetensors=True,)

    pipe = pipe.to("cuda")

    prompt = "A chest x-ray with pneumonia"
    pipe.enable_attention_slicing()
    image = pipe(prompt, generator=generator,).images[0]

    print(image)

    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.axis('off')
    plt.savefig('/path/to/images/stable_diffusion.png')

def test_controlnet():
    image = load_image("/home/moibhattacha/canny_cxr.png")
    image = np.array(image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)

    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.enable_xformers_memory_efficient_attention()

    pipe.enable_model_cpu_offload()

    image = pipe("A chest x-ray with pneumonia", image, num_inference_steps=20, generator=generator,).images[0]

    image.save('/path/to/images/controlnet.png')

def test_finetuned_stable_diffusion():
    model_path = "/path/to/stable_diffusion_v1_5_egc1_15k/"
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, revision="fp16")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    generator = torch.manual_seed(3407)
    
    prompt = "A photo of chest x-ray with tuberculosis"
    #"A photo of chest x-ray with pneumonia"
    
    image = pipe(prompt, num_inference_steps=50, generator=generator).images[0]

    image.save("/path/to/images/stable_diffusion_finetuned.png")

def test_finetuned_controlnet():
    base_model_path = "/path/to/stable_diffusion_v1_5_egc1_15k/"
    controlnet_path = "/path/to/controlnet_egc1_sdv1-5_canny/"

    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    # pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    #     base_model_path, controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
    # )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path, controlnet=controlnet, torch_dtype=torch.float16
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    pipe = pipe.to("cuda:0")

    control_image = load_image("/path/to/conditioning_images/7ef3ae2e-41c51122-d1ec5d68-2ad31016-98eab936.png".format(type))

    control_image = np.array(control_image)
    control_image = cv2.resize(control_image, (512, 512))
    control_image = Image.fromarray(control_image)

    control_image.save("/path/to/canny_cxr_temp.png")

    prompt = "A photo of chest x-ray with tuberculosis"
    #"A photo of chest x-ray with pneumonia"

    generator = torch.manual_seed(3407)
    image = pipe(
        prompt, num_inference_steps=50, generator=generator, image=control_image, controlnet_conditioning_scale=0.5, #image=resized_image, control_image=control_image
        negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
    ).images[0]
    image.save("/path/to/controlnet_finetuned.png")

def test_roentgen():
    model_path = "/path/to/roentgen/"
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, revision="fp16")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    generator = torch.manual_seed(3407)
    
    prompt = "A photo of chest x-ray with tuberculosis"
    #"A photo of chest x-ray with pneumonia"
    
    image = pipe(prompt, num_inference_steps=50, generator=generator).images[0]

    image.save("/path/to/roentgen.png")

def test_finetuned_gazediff():
    base_model_path = "/path/to/stable_diffusion_v1_5_egc1_15k/"

    focal_path = '/path/to/controlnet_egc1_sdv1-5_focal/'
    global_path = '/path/to/controlnet_egc1_sdv1-5_global/'

    controlnet_focal = ControlNetModel.from_pretrained(focal_path, torch_dtype=torch.float16).to("cuda")
    controlnet_global = ControlNetModel.from_pretrained(global_path, torch_dtype=torch.float16).to("cuda")

    pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=[controlnet_focal, controlnet_global], safety_checker=None,  torch_dtype=torch.float16)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    pipe = pipe.to("cuda:0")

    count = 0
    for i in tqdm(os.listdir('/path/to/conditioning_images/')):
        image = load_image("/path/to/images/{}".format(i))

        resized_image = np.array(image)
        resized_image = cv2.resize(resized_image, (512, 512))
        resized_image = Image.fromarray(resized_image)
        
        control_image_focal = load_image("/path/to/controlnet_eye_gaze_{}_1/conditioning_images/{}".format('focal', i))
        control_image_global = load_image("/path/to/controlnet_eye_gaze_{}_1/conditioning_images/{}".format('global', i))

        control_image_focal = np.array(control_image_focal)
        control_image_focal = cv2.resize(control_image_focal, (512, 512))
        control_image_focal = Image.fromarray(control_image_focal)

        control_image_global = np.array(control_image_focal)
        control_image_global = cv2.resize(control_image_global, (512, 512))
        control_image_global = Image.fromarray(control_image_global)

        image = [resized_image, resized_image]
        control_image = [control_image_focal, control_image_global]

        prompt = "A photo of chest x-ray with tuberculosis"
        #"A photo of chest x-ray with pneumonia"

        generator = torch.manual_seed(3407)
        image = pipe(
            prompt, num_inference_steps=70, controlnet_conditioning_scale=5.0, image=image, control_image=control_image, generator=generator,
            # negative_prompt="clear lungs, no abnormalities in lungs, low quality, mild opacity", #optional
        ).images[0]
        #### note: should only use control_image=control_image and not image=image, additionally tune num_inference_steps=range(10,100) & controlnet_conditioning_scale=range(0.1,5.0) accordingly####
        image.save("/path/to/images/gazediff/tuberculosis/{}".format(i))


if __name__ == '__main__':
    flag = 0
