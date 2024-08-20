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

    prompt = "A chext xray with pneumonia"
    pipe.enable_attention_slicing()
    image = pipe(prompt, generator=generator,).images[0]

    print(image)

    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.axis('off')
    plt.savefig('/home/moibhattacha/diffusion-classifier/images/stable_diffusion.png')

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

    image = pipe("A chext xray with pneumonia", image, num_inference_steps=20, generator=generator,).images[0]

    image.save('/home/moibhattacha/diffusion-classifier/images/controlnet.png')

def test_finetuned_stable_diffusion():
    model_path = "/home/moibhattacha/model_weights/diffuser_finetuning/stable_diffusion_v1_5_egc1_15k/"
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, revision="fp16")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    generator = torch.manual_seed(3407)
    prompt = "A photo of chext x-ray with tuberculosis"#"A chext xray with pneumonia"
    image = pipe(prompt, num_inference_steps=50, generator=generator).images[0]

    image.save("/home/moibhattacha/diffusion-classifier/images/stable_diffusion_finetuned.png")

def test_finetuned_controlnet():
    base_model_path = "/home/moibhattacha/model_weights/diffuser_finetuning/stable_diffusion_v1_5_egc1_15k/"
    controlnet_path = "/data04/shared/moibhattacha/model_weights/diffusers_finetuning/controlnet_egc1_sdv1-5_canny/"

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

    control_image = load_image("/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_cxr_1/conditioning_images/7ef3ae2e-41c51122-d1ec5d68-2ad31016-98eab936.png".format(type))

    control_image = np.array(control_image)
    control_image = cv2.resize(control_image, (512, 512))
    control_image = Image.fromarray(control_image)

    control_image.save("/home/moibhattacha/canny_cxr_temp.png")

    # image = load_image("/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_cxr_1/images/7ef3ae2e-41c51122-d1ec5d68-2ad31016-98eab936.png")

    # resized_image = np.array(image)
    # resized_image = cv2.resize(resized_image, (512, 512))

    # control_image = np.array(image)
    # control_image = cv2.Canny(control_image, 25, 50)
    # control_image = control_image[:, :, None]
    # control_image = np.concatenate([control_image, control_image, control_image], axis=2)
    # control_image = cv2.resize(control_image, (512, 512))
    # control_image = Image.fromarray(control_image)

    # control_image.save("/home/moibhattacha/canny_cxr_temp.png")

    prompt = "A photo of chext x-ray with tuberculosis"#"A chext xray with pneumonia"

    generator = torch.manual_seed(3407)
    image = pipe(
        prompt, num_inference_steps=50, generator=generator, image=control_image, controlnet_conditioning_scale=0.5, #image=resized_image, control_image=control_image
        negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
    ).images[0]
    image.save("/home/moibhattacha/diffusion-classifier/images/controlnet_finetuned.png")

def test_roentgen():
    model_path = "/data04/shared/moibhattacha/model_weights/diffusers_finetuning/roentgen/"
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, revision="fp16")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    generator = torch.manual_seed(3407)
    prompt = "A photo of chext x-ray with tuberculosis"#"A chext xray with pneumonia"
    image = pipe(prompt, num_inference_steps=50, generator=generator).images[0]

    image.save("/home/moibhattacha/diffusion-classifier/images/roentgen.png")

def test_finetuned_gazediff():
    base_model_path = "/data04/shared/moibhattacha/model_weights/diffusers_finetuning/stable_diffusion_v1_5_egc1_15k/"

    focal_path = '/data04/shared/moibhattacha/model_weights/diffusers_finetuning/controlnet_egc1_sdv1-5_focal/'
    global_path = '/data04/shared/moibhattacha/model_weights/diffusers_finetuning/controlnet_egc1_sdv1-5_global/'

    controlnet_focal = ControlNetModel.from_pretrained(focal_path, torch_dtype=torch.float16).to("cuda")
    controlnet_global = ControlNetModel.from_pretrained(global_path, torch_dtype=torch.float16).to("cuda")

    # pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=[controlnet_focal, controlnet_global], safety_checker=None,  torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=controlnet_focal, safety_checker=None,  torch_dtype=torch.float16)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    pipe = pipe.to("cuda:0")

    count = 0
    for i in tqdm(os.listdir('/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_focal_1/conditioning_images/')):
        image = load_image("/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_cxr_1/images/{}".format(i))

        resized_image = np.array(image)
        resized_image = cv2.resize(resized_image, (512, 512))
        resized_image = Image.fromarray(resized_image)
        
        control_image_focal = load_image("/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_{}_1/conditioning_images/{}".format('focal', i))
        control_image_global = load_image("/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_{}_1/conditioning_images/{}".format('global', i))

        control_image_focal = np.array(control_image_focal)
        control_image_focal = cv2.resize(control_image_focal, (512, 512))
        control_image_focal = Image.fromarray(control_image_focal)

        control_image_global = np.array(control_image_focal)
        control_image_global = cv2.resize(control_image_global, (512, 512))
        control_image_global = Image.fromarray(control_image_global)

        image = resized_image#[resized_image, resized_image]
        control_image = control_image_focal#[control_image_focal, control_image_global]

        prompt = "Tuberculosis"

        # generator = torch.manual_seed(3407)
        image = pipe(
            prompt, num_inference_steps=70, controlnet_conditioning_scale=5.0, image=image, control_image=control_image,#, generator=generator,
            # negative_prompt="clear lungs, no abnormalities in lungs, low quality, mild opacity",
        ).images[0]
        image.save("/home/moibhattacha/diffusion-classifier/images/gazediff/tuberculosis/{}".format(i))

        count += 1
        if count == 50:
            break

def test_finetuned_controlnet_gaze(type='global'):
    base_model_path = "/home/moibhattacha/model_weights/diffuser_finetuning/stable_diffusion_v1_5_egc1_15k/"
    controlnet_path = "/data04/shared/moibhattacha/model_weights/diffusers_finetuning/controlnet_egc1_sdv1-5_{}/".format(type)

    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    # pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    #     base_model_path, controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
    # )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path, controlnet=controlnet, torch_dtype=torch.float16
    )

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # remove following line if xformers is not installed or when using Torch 2.0.
    pipe.enable_xformers_memory_efficient_attention()

    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    pipe = pipe.to("cuda:0")

    control_image = load_image("/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_{}_1/conditioning_images/7ef3ae2e-41c51122-d1ec5d68-2ad31016-98eab936.png".format(type))

    control_image = np.array(control_image)
    control_image = cv2.resize(control_image, (512, 512))
    control_image = Image.fromarray(control_image)

    control_image.save("/home/moibhattacha/diffusion-classifier/images/canny_cxr_{}.png".format(type))

    prompt = "A photo of chext x-ray with pneumonia"

    generator = torch.manual_seed(3407)
    image = pipe(
        prompt, num_inference_steps=50, generator=generator, image=control_image, controlnet_conditioning_scale=0.5, #image=resized_image, control_image=control_image
        negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
    ).images[0]
    image.save("/home/moibhattacha/diffusion-classifier/images/test_finetuned_controlnet_{}.png".format(type))

# def compute_metrics_sd_generate(disease_list, disease_name='edema'):
def compute_metrics_sd_generate(image_name_x):
    model_path = "/data04/shared/moibhattacha/model_weights/diffusers_finetuning/stable_diffusion_v1_5_egc1_15k/"
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, revision="fp16")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    from datasets import load_dataset
    dataset = load_dataset("json", data_files= str('/data04/shared/moibhattacha/3_project_eyegaze/metadata_reflacx.jsonl'))

    count = 0
    for i in tqdm(dataset['train']):
        image_path = i['file_name']
        text = i['text']

        image_name = image_path.split('/')[-1]
        image_name = image_name.split('.png')[0]

        if image_name == image_name_x:
            image = load_image('/data04/shared/moibhattacha/3_project_eyegaze/eye_gaze_cxr_2_png/'+image_path)

            resized_image = np.array(image)
            resized_image = cv2.resize(resized_image, (512, 512))

            generator = torch.manual_seed(3407)
            prompt = text
            image = pipe(prompt, num_inference_steps=50, generator=generator).images[0]

            # image.save("/home/moibhattacha/diffusion-classifier/images/sd/{}/{}.png".format(disease_name, image_name))
            image.save("/home/moibhattacha/diffusion-classifier/images/sd_{}.png".format(image_name))

def compute_metrics_roentgen_generate(disease_list, disease_name='edema'):
    model_path = "/data04/shared/moibhattacha/model_weights/diffusers_finetuning/roentgen/"
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, revision="fp16")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    from datasets import load_dataset
    dataset = load_dataset("json", data_files= str('/data04/shared/moibhattacha/3_project_eyegaze/metadata_reflacx.jsonl'))

    count = 0
    for i in tqdm(dataset['train']):
        image_path = i['file_name']
        text = i['text']

        image_name = image_path.split('/')[-1]
        image_name = image_name.split('.png')[0]

        if image_name in disease_list:
            image = load_image('/data04/shared/moibhattacha/3_project_eyegaze/eye_gaze_cxr_2_png/'+image_path)

            resized_image = np.array(image)
            resized_image = cv2.resize(resized_image, (512, 512))

            generator = torch.manual_seed(3407)
            prompt = text
            image = pipe(prompt, num_inference_steps=50, generator=generator).images[0]

            image.save("/home/moibhattacha/diffusion-classifier/images/roentgen/{}/{}.png".format(disease_name, image_name))

def compute_metrics_controlnet_generate(disease_list, disease_name='edema'):
    base_model_path = "/data04/shared/moibhattacha/model_weights/diffusers_finetuning/stable_diffusion_v1_5_egc1_15k/"
    controlnet_path = "/data04/shared/moibhattacha/model_weights/diffusers_finetuning/controlnet_egc1_sdv1-5_canny/"

    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=controlnet, torch_dtype=torch.float16)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    pipe = pipe.to("cuda:0")

    from datasets import load_dataset
    dataset = load_dataset("json", data_files= str('/data04/shared/moibhattacha/3_project_eyegaze/metadata_reflacx.jsonl'))

    count = 0
    for i in tqdm(dataset['train']):
        image_path = i['file_name']
        text = i['text']

        image_name = image_path.split('/')[-1]
        image_name = image_name.split('.png')[0]

        if image_name in disease_list:
            image = load_image('/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_cxr_2/images/'+image_path)
            resized_image = np.array(image)
            resized_image = cv2.resize(resized_image, (512, 512))
            
            control_image = load_image("/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_cxr_2/conditioning_images/{}.png".format(image_name))
            control_image = np.array(control_image)
            control_image = cv2.resize(control_image, (512, 512))
            control_image = Image.fromarray(control_image)

            prompt = text

            generator = torch.manual_seed(3407)
            image = pipe(
                prompt, num_inference_steps=50, controlnet_conditioning_scale=0.1, image=control_image, generator=generator,#=resized_image, control_image=control_image,
                negative_prompt="clear lungs, no abnormalities in lungs, low quality",
            ).images[0]
            image.save("/home/moibhattacha/diffusion-classifier/images/controlnet/{}/{}.png".format(disease_name, image_name))

def compute_metrics_gazediff_generate(disease_list, disease_name='edema'):
    base_model_path = "/data04/shared/moibhattacha/model_weights/diffusers_finetuning/stable_diffusion_v1_5_egc1_15k/"

    focal_path = '/data04/shared/moibhattacha/model_weights/diffusers_finetuning/controlnet_egc1_sdv1-5_focal/'
    global_path = '/data04/shared/moibhattacha/model_weights/diffusers_finetuning/controlnet_egc1_sdv1-5_global/'

    controlnet_focal = ControlNetModel.from_pretrained(focal_path, torch_dtype=torch.float16).to("cuda")
    controlnet_global = ControlNetModel.from_pretrained(global_path, torch_dtype=torch.float16).to("cuda")

    pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=[controlnet_focal, controlnet_global], safety_checker=None,  torch_dtype=torch.float16)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    pipe = pipe.to("cuda:0")

    from datasets import load_dataset
    dataset = load_dataset("json", data_files= str('/data04/shared/moibhattacha/3_project_eyegaze/metadata_reflacx.jsonl'))

    count = 0
    for i in tqdm(dataset['train']):
        image_path = i['file_name']
        text = i['text']

        image_name = image_path.split('/')[-1]
        image_name = image_name.split('.png')[0]

        if image_name in disease_list:
            image = load_image('/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_cxr_2/images/'+image_path)
            resized_image = np.array(image)
            resized_image = cv2.resize(resized_image, (512, 512))
            
            control_image_focal = load_image("/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_{}_2/conditioning_images/{}.png".format('focal', image_name))
            control_image_global = load_image("/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_{}_2/conditioning_images/{}.png".format('global', image_name))

            control_image_focal = np.array(control_image_focal)
            control_image_focal = cv2.resize(control_image_focal, (512, 512))
            control_image_focal = Image.fromarray(control_image_focal)

            control_image_global = np.array(control_image_focal)
            control_image_global = cv2.resize(control_image_global, (512, 512))
            control_image_global = Image.fromarray(control_image_global)

            control_image = [control_image_focal, control_image_global]

            prompt = text#"A photo of chext x-ray with pneumonia"

            generator = torch.manual_seed(3407)
            image = pipe(
                prompt, num_inference_steps=70, controlnet_conditioning_scale=0.1, image=control_image, generator=generator,#=resized_image, control_image=control_image,
                # negative_prompt="clear lungs, no abnormalities in lungs, low quality",
            ).images[0]
            image.save("/home/moibhattacha/diffusion-classifier/images/gazediff_temp/{}/{}.png".format(disease_name, image_name))
            # image.save("/home/moibhattacha/diffusion-classifier/images/gazediff_{}.png".format(image_name))
                # break

        # count += 1
        # if count == 10:
        #     break

from torchvision.transforms import functional as F
from torchmetrics.image.fid import FrechetInceptionDistance

def calculate_fid(real_dataset_path="/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_cxr_2/images/", generated_dataset_path="/home/moibhattacha/diffusion-classifier/images/roentgen/"):
    real_image_paths = sorted([os.path.join(real_dataset_path, x) for x in os.listdir(generated_dataset_path)])
    real_images = [np.array(Image.open(path).convert("RGB")) for path in real_image_paths]
    # print('Real Images Loaded')

    generated_image_paths = sorted([os.path.join(generated_dataset_path, x) for x in os.listdir(generated_dataset_path)])
    generated_images = [np.array(Image.open(path).convert("RGB")) for path in generated_image_paths]
    # print('Generated Images Loaded')

    def preprocess_image(image):
        image = torch.tensor(image).unsqueeze(0)
        image = image.permute(0, 3, 1, 2) / 255.0
        return F.center_crop(image, (256, 256))

    real_images = torch.cat([preprocess_image(image) for image in real_images])
    generated_images = torch.cat([preprocess_image(image) for image in generated_images])
    # print(real_images.shape, generated_images.shape)

    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_images, real=True)
    fid.update(generated_images, real=False)

    print(f"FID: {float(fid.compute())}")

def calculate_clip_score(path):
    from torchmetrics.functional.multimodal import clip_score
    from functools import partial

    from datasets import load_dataset
    dataset = load_dataset("json", data_files= str('/data04/shared/moibhattacha/3_project_eyegaze/metadata_reflacx.jsonl'))

    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

    def calculate_clip_score_x(images, prompts):
        images_int = (images * 255).astype("uint8")
        clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
        return round(float(clip_score), 4)
    
    all_images = os.listdir(path)

    images = []
    prompts = []
    count = 0
    for i in tqdm(dataset['train']):
        image_path = i['file_name']
        text = i['text']

        image_name = image_path.split('/')[-1]

        if image_name in all_images:

            image = load_image(path+"{}".format(image_name))
            resized_image = np.array(image)
            resized_image = cv2.resize(resized_image, (512, 512))

            images.append(resized_image)
            prompts.append(text)
        
        # count += 1
        # if count == 20:
        #     break

    images = np.array(images)
    # print(images.shape)

    sd_clip_score = calculate_clip_score_x(images, prompts)
    print(f"CLIP score: {sd_clip_score}")

from torchmetrics.image import StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure
def calculate_ssim(real_dataset_path="/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_cxr_2/images/", generated_dataset_path="/home/moibhattacha/diffusion-classifier/images/roentgen/"):
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    
    real_image_paths = sorted([os.path.join(real_dataset_path, x) for x in os.listdir(generated_dataset_path)])
    real_images = [np.array(Image.open(path).convert("RGB")) for path in real_image_paths]
    # print('Real Images Loaded')

    generated_image_paths = sorted([os.path.join(generated_dataset_path, x) for x in os.listdir(generated_dataset_path)])
    generated_images = [np.array(Image.open(path).convert("RGB")) for path in generated_image_paths]
    # print('Generated Images Loaded')

    def preprocess_image(image):
        image = torch.tensor(image).unsqueeze(0)
        image = image.permute(0, 3, 1, 2) / 255.0
        return F.center_crop(image, (256, 256))

    ssim_score = 0
    for real_image, generated_image in tqdm(zip(real_images, generated_images)):
        ssim_score += ssim(preprocess_image(generated_image), preprocess_image(real_image))
    print(ssim_score/len(real_images))    

def calculate_ms_ssim(real_dataset_path="/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_cxr_2/images/", generated_dataset_path="/home/moibhattacha/diffusion-classifier/images/roentgen/"):
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
    
    real_image_paths = sorted([os.path.join(real_dataset_path, x) for x in os.listdir(real_dataset_path)])
    real_images = [np.array(Image.open(path).convert("RGB")) for path in real_image_paths]
    print('Real Images Loaded')

    generated_image_paths = sorted([os.path.join(generated_dataset_path, x) for x in os.listdir(generated_dataset_path)])
    generated_images = [np.array(Image.open(path).convert("RGB")) for path in generated_image_paths]
    print('Generated Images Loaded')

    def preprocess_image(image):
        image = torch.tensor(image).unsqueeze(0)
        image = image.permute(0, 3, 1, 2) / 255.0
        return F.center_crop(image, (256, 256))

    ssim_score = 0
    for real_image, generated_image in tqdm(zip(real_images, generated_images)):
        ssim_score += ms_ssim(preprocess_image(generated_image), preprocess_image(real_image))
    print(ssim_score/len(real_images))    

def test_finetuned_gazediff_reflacx(prompt=None, image_name=None):
    base_model_path = "/data04/shared/moibhattacha/model_weights/diffusers_finetuning/stable_diffusion_v1_5_egc1_15k/"

    focal_path = '/data04/shared/moibhattacha/model_weights/diffusers_finetuning/controlnet_egc1_sdv1-5_focal/'
    global_path = '/data04/shared/moibhattacha/model_weights/diffusers_finetuning/controlnet_egc1_sdv1-5_global/'

    controlnet_focal = ControlNetModel.from_pretrained(focal_path, torch_dtype=torch.float16).to("cuda")
    controlnet_global = ControlNetModel.from_pretrained(global_path, torch_dtype=torch.float16).to("cuda")

    pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=[controlnet_focal, controlnet_global], safety_checker=None,  torch_dtype=torch.float16)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    pipe = pipe.to("cuda:0")

    # for i in tqdm(os.listdir('/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_focal_1/conditioning_images/')):
    image = load_image("/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_cxr_2/images/{}".format(image_name))

    resized_image = np.array(image)
    resized_image = cv2.resize(resized_image, (512, 512))
    
    control_image_focal = load_image("/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_{}_2/conditioning_images/{}".format('focal', image_name))
    control_image_global = load_image("/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_{}_2/conditioning_images/{}".format('global', image_name))

    control_image_focal = np.array(control_image_focal)
    control_image_focal = cv2.resize(control_image_focal, (512, 512))
    control_image_focal = Image.fromarray(control_image_focal)

    control_image_global = np.array(control_image_focal)
    control_image_global = cv2.resize(control_image_global, (512, 512))
    control_image_global = Image.fromarray(control_image_global)

    control_image = [control_image_focal, control_image_global]

    # prompt = "A photo of chext x-ray with tuberculosis"

    generator = torch.manual_seed(3407)
    image = pipe(
        prompt, num_inference_steps=40, controlnet_conditioning_scale=2.5, image=control_image, generator=generator,#=resized_image, control_image=control_image,
        negative_prompt="low quality",
    ).images[0]
    image.save("/home/moibhattacha/diffusion-classifier/images/gazediff/qualitative_temp/{}".format(image_name))
 
if __name__ == '__main__':
    flag = 0

    # test_stable_diffusion()
    # test_controlnet()

    # test_finetuned_stable_diffusion()
    # test_finetuned_controlnet()
    # test_roentgen()
    # test_finetuned_gazediff()


    # test_finetuned_controlnet_gaze(type='focal')
    # test_finetuned_controlnet_gaze(type='global')

    # compute_metrics_sd_generate()
    # compute_metrics_controlnet_generate()
    # compute_metrics_roentgen_generate()
    # compute_metrics_gazediff_generate()

    # calculate_fid(real_dataset_path="/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_cxr_2/images/", generated_dataset_path="/home/moibhattacha/diffusion-classifier/images/sd/")
    # calculate_fid(real_dataset_path="/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_cxr_2/images/", generated_dataset_path="/home/moibhattacha/diffusion-classifier/images/roentgen/")
    # calculate_fid(real_dataset_path="/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_cxr_2/images/", generated_dataset_path="/home/moibhattacha/diffusion-classifier/images/controlnet/")
    # calculate_fid(real_dataset_path="/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_cxr_2/images/", generated_dataset_path="/home/moibhattacha/diffusion-classifier/images/gazediff_temp/")

    # calculate_clip_score(path="/home/moibhattacha/diffusion-classifier/images/sd/")
    # calculate_clip_score(path="/home/moibhattacha/diffusion-classifier/images/roentgen/")
    # calculate_clip_score(path="/home/moibhattacha/diffusion-classifier/images/controlnet/")
    # calculate_clip_score(path="/home/moibhattacha/diffusion-classifier/images/gazediff_temp/")

    # calculate_ssim(real_dataset_path="/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_cxr_2/images/", generated_dataset_path="/home/moibhattacha/diffusion-classifier/images/sd/")
    # calculate_ssim(real_dataset_path="/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_cxr_2/images/", generated_dataset_path="/home/moibhattacha/diffusion-classifier/images/roentgen/")
    # calculate_ssim(real_dataset_path="/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_cxr_2/images/", generated_dataset_path="/home/moibhattacha/diffusion-classifier/images/controlnet/")
    # calculate_ssim(real_dataset_path="/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_cxr_2/images/", generated_dataset_path="/home/moibhattacha/diffusion-classifier/images/gazediff_temp/")

    # calculate_ms_ssim(real_dataset_path="/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_cxr_2/images/", generated_dataset_path="/home/moibhattacha/diffusion-classifier/images/sd/")
    # calculate_ms_ssim(real_dataset_path="/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_cxr_2/images/", generated_dataset_path="/home/moibhattacha/diffusion-classifier/images/roentgen/")
    # calculate_ms_ssim(real_dataset_path="/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_cxr_2/images/", generated_dataset_path="/home/moibhattacha/diffusion-classifier/images/controlnet/")
    # calculate_ms_ssim(real_dataset_path="/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_cxr_2/images/", generated_dataset_path="/home/moibhattacha/diffusion-classifier/images/gazediff_temp/")

    ####-####

    # test_finetuned_gazediff_reflacx(prompt="left ij catheter is in the svc. right picc line is at the cavoatrial junction. there is a left ventricular assist device. cardiac pacer is noted with leads in the right ventricle and coronary sinus. there is moderate to severe cardiomegaly. mild diffuse hazy opacity likely represents pulmonary edema. there is left basilar atelectasis or consolidation. no pneumothorax. no acute osseous abnormality.", image_name="c9e5f35b-c6a65cbe-adb2ca51-428ee51d-f086523a.png")
    # test_finetuned_gazediff_reflacx(prompt="support apparatus. no pneumothorax. enlarged cardiac silhouette. possible mild right basilar consolidation.", image_name="dcdc4bd9-4301b111-2a65a814-ee8e7bc5-7f0b9a5a.png")
    # test_finetuned_gazediff_reflacx(prompt="no pneumothorax. cardiac silhouette normal. clear lungs.", image_name="888c19bd-5c440de3-201cedb1-bbb8d0c8-dfe5d659.png")
    # test_finetuned_gazediff_reflacx(prompt="faint opacities in the left lung base may represent ground-glass opacities or atelectasis. no pleural effusion or pneumothorax. normal cardiomediastinal silhouette. no acute osseous abnormality.", image_name="638e38a7-14b085fe-c65bf685-7d3ee265-7c420d3e.png")
    
    # #### consolidation ####
    # test_finetuned_gazediff_reflacx(prompt="cardiac silhouette is enlarged. peribronchial cuffing consistent with pulmonary edema. patchy consolidation at the right lung base and small right pleural effusion. no pneumothorax. no acute osseous abnormality.", image_name="f255b3ef-c0bf0f6b-4e6f2c68-0e68aa39-65913951.png")
    # test_finetuned_gazediff_reflacx(prompt="large right pleural effusion with patchy ground-glass abnormality in the right upper lobe and right basilar consolidation versus multisegmental atelectasis, mass not excluded. left basilar subsegmental atelectasis. heart size appears within normal limits although the right cardiac border is obscured. no pneumothorax. no acute osseous abnormality.", image_name="d471efcd-b9883de0-61154002-0ed78c74-1fe5a5e5.png")
    # test_finetuned_gazediff_reflacx(prompt="probable external lead overlying the right hemithorax. cardiac silhouette is enlarged. left basilar opacity consistent with consolidation, multisegmental atelectasis and / or pleural effusion. no pneumothorax is visualized. no acute osseous abnormality. mild mediastinal shift to the left.", image_name="2e0e4c28-77275236-9695b5f1-bedff2d6-f816e6f7.png")
    # test_finetuned_gazediff_reflacx(prompt="right chest port is at the cavoatrial junction. two right-sided chest tubes are seen. there is consolidation versus mass in the right upper and possibly lower lung. moderate right pleural effusion. no pneumothorax. probable emphysema. some trace left pleural effusion. the heart is normal size. no acute osseous abnormality.", image_name="1f0fa741-0c5e5a37-92744e13-2920cf39-ec6d6a86.png")

    # "d471efcd-b9883de0-61154002-0ed78c74-1fe5a5e5.png"

    compute_metrics_sd_generate("d471efcd-b9883de0-61154002-0ed78c74-1fe5a5e5")

    # #### teaser ####
    # compute_metrics_sd_generate()
    # test_finetuned_gazediff_reflacx(prompt="left picc line tip is in the left brachiocephalic vein. there are bilateral pleural effusions and likely atelectasis versus consolidation. no pneumothorax. the heart is normal size. pulmonary vessels are normal. no acute osseous abnormality.", image_name="fa771fa1-d9571d07-bff8f655-327734a7-6e10b29d.png")

    # import pandas as pd
    # df_atelectesis = pd.read_csv('/home/moibhattacha/temp_csv/atelectesis.csv')
    # df_edema = pd.read_csv('/home/moibhattacha/temp_csv/edema.csv')
    # df_fracture = pd.read_csv('/home/moibhattacha/temp_csv/fracture.csv')
    # df_opacity = pd.read_csv('/home/moibhattacha/temp_csv/opacity.csv')
    # df_pneumothorax = pd.read_csv('/home/moibhattacha/temp_csv/pneumothorax.csv')
    # df_consolidation = pd.read_csv('/home/moibhattacha/temp_csv/consolidation.csv')
    # df_emphysema = pd.read_csv('/home/moibhattacha/temp_csv/emphysema.csv')
    # df_mass = pd.read_csv('/home/moibhattacha/temp_csv/mass.csv')
    # df_pleural_abnormality = pd.read_csv('/home/moibhattacha/temp_csv/pleural_abnormality.csv')
    # df_support_devices = pd.read_csv('/home/moibhattacha/temp_csv/support_devices.csv')

    # compute_metrics_controlnet_generate(df_atelectesis[:20]['dicom_id'].tolist(), 'atelectesis')
    # compute_metrics_gazediff_generate(df_atelectesis[:20]['dicom_id'].tolist(), 'atelectesis')
    # compute_metrics_controlnet_generate(df_edema[:20]['dicom_id'].tolist(), 'edema')
    # compute_metrics_gazediff_generate(df_edema[:20]['dicom_id'].tolist(), 'edema')
    # compute_metrics_controlnet_generate(df_fracture[:20]['dicom_id'].tolist(), 'fracture')
    # compute_metrics_gazediff_generate(df_fracture[:20]['dicom_id'].tolist(), 'fracture')
    # compute_metrics_controlnet_generate(df_opacity[:20]['dicom_id'].tolist(), 'opacity')
    # compute_metrics_gazediff_generate(df_opacity[:20]['dicom_id'].tolist(), 'opacity')
    # compute_metrics_controlnet_generate(df_pneumothorax[:20]['dicom_id'].tolist(), 'pneumothorax')
    # compute_metrics_gazediff_generate(df_pneumothorax[:20]['dicom_id'].tolist(), 'pneumothorax')
    # compute_metrics_controlnet_generate(df_consolidation[:20]['dicom_id'].tolist(), 'consolidation')
    # compute_metrics_gazediff_generate(df_consolidation[:20]['dicom_id'].tolist(), 'consolidation')
    # compute_metrics_controlnet_generate(df_emphysema[:20]['dicom_id'].tolist(), 'emphysema')
    # compute_metrics_gazediff_generate(df_emphysema[:20]['dicom_id'].tolist(), 'emphysema')
    # compute_metrics_controlnet_generate(df_mass[:20]['dicom_id'].tolist(), 'mass')
    # compute_metrics_gazediff_generate(df_mass[:20]['dicom_id'].tolist(), 'mass')
    # compute_metrics_controlnet_generate(df_pleural_abnormality[:20]['dicom_id'].tolist(), 'pleural_abnormality')
    # compute_metrics_gazediff_generate(df_pleural_abnormality[:20]['dicom_id'].tolist(), 'pleural_abnormality')
    # compute_metrics_controlnet_generate(df_support_devices[:20]['dicom_id'].tolist(), 'support_devices')
    # compute_metrics_gazediff_generate(df_support_devices[:20]['dicom_id'].tolist(), 'support_devices')

    # compute_metrics_sd_generate(df_atelectesis[:20]['dicom_id'].tolist(), 'atelectesis')
    # compute_metrics_roentgen_generate(df_atelectesis[:20]['dicom_id'].tolist(), 'atelectesis')
    # compute_metrics_sd_generate(df_edema[:20]['dicom_id'].tolist(), 'edema')
    # compute_metrics_roentgen_generate(df_edema[:20]['dicom_id'].tolist(), 'edema')
    # compute_metrics_sd_generate(df_fracture[:20]['dicom_id'].tolist(), 'fracture')
    # compute_metrics_roentgen_generate(df_fracture[:20]['dicom_id'].tolist(), 'fracture')
    # compute_metrics_sd_generate(df_opacity[:20]['dicom_id'].tolist(), 'opacity')
    # compute_metrics_roentgen_generate(df_opacity[:20]['dicom_id'].tolist(), 'opacity')
    # compute_metrics_sd_generate(df_pneumothorax[:20]['dicom_id'].tolist(), 'pneumothorax')
    # compute_metrics_roentgen_generate(df_pneumothorax[:20]['dicom_id'].tolist(), 'pneumothorax')
    # compute_metrics_sd_generate(df_consolidation[:20]['dicom_id'].tolist(), 'consolidation')
    # compute_metrics_roentgen_generate(df_consolidation[:20]['dicom_id'].tolist(), 'consolidation')
    # compute_metrics_sd_generate(df_emphysema[:20]['dicom_id'].tolist(), 'emphysema')
    # compute_metrics_roentgen_generate(df_emphysema[:20]['dicom_id'].tolist(), 'emphysema')
    # compute_metrics_sd_generate(df_mass[:20]['dicom_id'].tolist(), 'mass')
    # compute_metrics_roentgen_generate(df_mass[:20]['dicom_id'].tolist(), 'mass')
    # compute_metrics_sd_generate(df_pleural_abnormality[:20]['dicom_id'].tolist(), 'pleural_abnormality')
    # compute_metrics_roentgen_generate(df_pleural_abnormality[:20]['dicom_id'].tolist(), 'pleural_abnormality')
    # compute_metrics_sd_generate(df_support_devices[:20]['dicom_id'].tolist(), 'support_devices')
    # compute_metrics_roentgen_generate(df_support_devices[:20]['dicom_id'].tolist(), 'support_devices')
    
    # disease_name = [
    #     'atelectesis', 
    #     'consolidation', 
    #     'emphysema', 
    #     'fracture',
    #     'opacity',
    #     'pneumothorax',
    #     'edema',
    #     'pleural_abnormality',
    #     'mass',
    #     'support_devices'
    # ]

    # # mkdir atelectesis consolidation emphysema fracture opacity pneumothorax edema pleural_abnormality mass support_devices

    # for i in disease_name:
    #     if i == 'support_devices':
    #         print(i)
    #         calculate_fid(real_dataset_path="/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_cxr_2/images/", generated_dataset_path="/home/moibhattacha/diffusion-classifier/images/sd/{}/".format(i))
    #         calculate_fid(real_dataset_path="/data04/shared/moibhattacha/3_project_eyegaze/controlnet_eye_gaze_cxr_2/images/", generated_dataset_path="/home/moibhattacha/diffusion-classifier/images/roentgen/{}/".format(i))

    #         calculate_clip_score(path="/home/moibhattacha/diffusion-classifier/images/sd/{}/".format(i))
    #         calculate_clip_score(path="/home/moibhattacha/diffusion-classifier/images/roentgen/{}/".format(i))
