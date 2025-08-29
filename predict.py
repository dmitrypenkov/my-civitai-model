import torch
import os
from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import subprocess
import time

MODEL_URL = "https://civitai.com/api/download/models/1883050?type=Model&format=SafeTensor&size=pruned&fp=fp16"  # Замените на вашу ссылку
MODEL_CACHE = "model_cache"
MODEL_FILE = "model.safetensors"

class Predictor(BasePredictor):
    def setup(self):
        """Загрузка модели при старте"""
        print("Начинаем загрузку модели...")
        
        # Создаем папку для кеша
        os.makedirs(MODEL_CACHE, exist_ok=True)
        model_path = os.path.join(MODEL_CACHE, MODEL_FILE)
        
        # Скачиваем модель если её нет
        if not os.path.exists(model_path):
            print(f"Скачиваем модель с {MODEL_URL}")
            # Используем wget для загрузки
            subprocess.run([
                "wget", 
                "-O", model_path,
                "--content-disposition",
                MODEL_URL
            ], check=True)
            print("Модель скачана!")
        
        print("Загружаем модель в память...")
        
        # Загружаем как SDXL модель
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        
        # Оптимизация
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        # Переносим на GPU
        self.pipe = self.pipe.to("cuda")
        
        # Включаем оптимизации
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.enable_vae_slicing()
        
        print("Модель готова к работе!")

    def predict(
        self,
        prompt: str = Input(
            description="Описание изображения",
            default="beautiful landscape, high quality"
        ),
        negative_prompt: str = Input(
            description="Что НЕ должно быть на изображении",
            default="ugly, blurry, low quality, distorted"
        ),
        width: int = Input(
            description="Ширина изображения",
            default=1024,
            choices=[512, 768, 1024, 1280, 1536]
        ),
        height: int = Input(
            description="Высота изображения",
            default=1024,
            choices=[512, 768, 1024, 1280, 1536]
        ),
        num_inference_steps: int = Input(
            description="Количество шагов (больше = качественнее, но медленнее)",
            default=30,
            ge=1,
            le=100
        ),
        guidance_scale: float = Input(
            description="Насколько точно следовать промпту (7-12 обычно лучше)",
            default=7.5,
            ge=1.0,
            le=20.0
        ),
        seed: int = Input(
            description="Seed для воспроизводимости (-1 для случайного)",
            default=-1
        ),
    ) -> Path:
        """Генерация изображения"""
        
        # Генерируем seed если не указан
        if seed == -1:
            seed = int(time.time())
        
        generator = torch.Generator("cuda").manual_seed(seed)
        
        print(f"Генерируем с seed: {seed}")
        
        # Генерируем изображение
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        # Сохраняем результат
        output_path = "/tmp/output.png"
        image.save(output_path)
        
        print(f"Изображение сохранено: {output_path}")
        return Path(output_path)
