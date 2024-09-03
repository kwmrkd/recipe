import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import AutoTokenizer
from torch.optim import AdamW
from tqdm import tqdm
import time

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Custom Dataset
class CIFAR10ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = sorted(os.listdir(img_dir))  # Sorting to maintain order

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert("RGB")
        class_index = int(self.img_files[idx][3:5])  # Extract class from the filename
        prompt = f"a photo of a {CIFAR10_CLASSES[class_index]}"
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "prompt": prompt
        }

transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10 image size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = CIFAR10ImageDataset(img_dir='./datasets/embedded/cifar10/images/', transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


model_id = "CompVis/stable-diffusion-v1-4" 
pipeline = StableDiffusionPipeline.from_pretrained(model_id)
pipeline = pipeline.to("cuda")  # Assuming you have a GPU



# Tokenizer for text prompts
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# Fine-tuning setup
optimizer = AdamW(pipeline.unet.parameters(), lr=1e-5)

num_epochs = 5

# Training loop
pipeline.unet.train()
for epoch in range(num_epochs):
    for batch in tqdm(dataloader):
        images = batch['image'].to(pipeline.device)
        prompts = batch['prompt']
        
        # Tokenize the prompts
        text_inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
        text_input_ids = text_inputs.input_ids.to(pipeline.device)

        # Encode text prompts to get the text embeddings
        encoder_hidden_states = pipeline.text_encoder(text_input_ids)[0]

        # Encode images to latents
        latents = pipeline.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215  # scaling factor

        # Sample a random timestep for each image in the batch
        timesteps = torch.randint(0, pipeline.scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device).long()

        # Generate random noise
        noise = torch.randn_like(latents).to(pipeline.device)

        # Add noise to the latents based on the current timestep
        noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

        # Predict noise using U-Net
        noise_pred = pipeline.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Compute the loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    print(f"Epoch {epoch + 1}/{num_epochs} completed, Loss: {loss.item()}")

# Save the fine-tuned model
pipeline.save_pretrained("./fine-tuned-stable-diffusion")
