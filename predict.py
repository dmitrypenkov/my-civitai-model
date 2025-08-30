import torch
import os
from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import time

class Predictor(BasePredictor):
    def setup(self):
        """Загрузка модели при старте"""
        print("Загружаем модель в память...")
        
        # Загружаем локальную модель
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            "model.safetensors",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        
        # Оптимизация планировщика
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        # Переносим на GPU
        self.pipe = self.pipe.to("cuda")
        
        # Включаем оптимизации памяти
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.enable_vae_slicing()
        
        print("Модель готова к работе!")

    def predict(
        self,
        prompt: str = Input(
            description="Описание изображения",
            default="beautiful landscape, high quality, detailed"
        ),
        negative_prompt: str = Input(
            description="Что НЕ должно быть на изображении",
            default="ugly, blurry, low quality, distorted, disfigured"
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
            default=-1,
            ge=-1,
            le=2147483647
        ),
        num_images: int = Input(
            description="Количество изображений",
            default=1,
            ge=1,
            le=4
        ),
    ) -> list[Path]:
        """Генерация изображения"""
        
        # Генерируем seed если не указан
        if seed == -1:
            seed = int(time.time())
        
        generator = torch.Generator("cuda").manual_seed(seed)
        
        print(f"Генерируем {num_images} изображений с seed: {seed}")
        print(f"Промпт: {prompt}")
        
        # Генерируем изображения
        images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=num_images
        ).images
        
        # Сохраняем результаты
        output_paths = []
        for i, image in enumerate(images):
            output_path = f"/tmp/output_{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))
            print(f"Изображение {i+1} сохранено: {output_path}")
        
        return output_paths if len(output_paths) > 1 else output_paths[0]
