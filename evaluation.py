from torchvision.transforms import functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure

def calculate_fid(real_dataset_path="/path/to/images/", generated_dataset_path="/path/to/generated_images/"):
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
    dataset = load_dataset("json", data_files= str('/path/to/metadata_reflacx.jsonl'))

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

    images = np.array(images)

    sd_clip_score = calculate_clip_score_x(images, prompts)
    print(f"CLIP score: {sd_clip_score}")

def calculate_ssim(real_dataset_path="/path/to/images/", generated_dataset_path="/path/to/generated_images/"):
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

if __name__ == '__main__':
    flag = 0

 
