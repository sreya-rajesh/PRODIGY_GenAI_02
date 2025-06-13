# PRODIGY_GenAI_02
# Text-to-Image Generator

A Python-based text-to-image generator using Stable Diffusion v1.5 that transforms text descriptions into high-quality images with GPU acceleration support.

## Description

This project implements a text-to-image generator using the Stable Diffusion v1.5 model from RunwayML. Originally developed in Google Colab, it provides an interactive command-line interface for generating images from text prompts. The application automatically detects available hardware (GPU/CPU) and optimizes performance accordingly, making it accessible for both high-end and modest computing environments.

## Features

- **Stable Diffusion v1.5 Integration**: Uses the powerful RunwayML Stable Diffusion model
- **Automatic Hardware Detection**: Optimizes for GPU when available, gracefully falls back to CPU
- **Memory Optimization**: Implements multiple memory-saving techniques including:
  - Half-precision (float16) for GPU
  - XFormers memory efficient attention
  - Model CPU offloading
- **Interactive CLI**: Simple command-line interface for prompt input
- **Customizable Parameters**: Control generation with negative prompts, inference steps, and guidance scale
- **Seed Support**: Reproducible results with optional seed parameter
- **Visual Display**: Automatic image display using matplotlib
- **Error Handling**: Comprehensive error management with helpful troubleshooting tips
- **Google Colab Ready**: Optimized for Colab environment with GPU setup detection

## Tech Stack

- **Python 3.7+**: Core programming language
- **PyTorch**: Deep learning framework with CUDA support
- **Diffusers**: Hugging Face library for diffusion models
- **Transformers**: Pre-trained model management
- **Matplotlib**: Image visualization and display
- **Pillow (PIL)**: Image processing
- **Google Colab**: Development and execution environment

## Installation

### Prerequisites
- Python 3.7 or higher
- CUDA-compatible GPU (recommended) or CPU
- 4GB+ RAM (8GB+ recommended for GPU)
- Stable internet connection for model download

### For Google Colab (Recommended)
1. Open the notebook in Google Colab:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreya-rajesh/PRODIGY_GenAI_02/blob/main/imagegeneration.ipynb)

2. Enable GPU acceleration:
   - Runtime → Change runtime type → Hardware accelerator → GPU

3. Run all cells in sequence

### For Local Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sreya-rajesh/text-to-image-generator.git
   cd text-to-image-generator
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install diffusers transformers accelerate
   pip install matplotlib pillow
   ```

4. For XFormers optimization (optional but recommended):
   ```bash
   pip install xformers
   ```

## Usage

### Running the Application
1. Execute the Python script:
   ```bash
   python imagegeneration.py
   ```

2. The application will:
   - Automatically detect your hardware (GPU/CPU)
   - Load the Stable Diffusion v1.5 model (this may take several minutes on first run)
   - Display system information and optimizations applied

3. When prompted, enter your image description:
   ```
   Enter your image prompt: A serene mountain landscape at sunset with purple clouds
   ```

4. Optionally, enter a negative prompt (or press Enter for default):
   ```
   Enter the negative prompt (by default:"blurry and distorted"): low quality, pixelated
   ```

5. Wait for generation to complete and view the result

### Example Prompts

**Landscape:**
```
Enter your image prompt: A mystical forest with glowing mushrooms and fairy lights, ethereal atmosphere
```

**Portrait:**
```
Enter your image prompt: Portrait of a wise old wizard with long white beard, detailed, fantasy art style
```

**Abstract:**
```
Enter your image prompt: Abstract digital art with flowing neon colors, cyberpunk aesthetic
```

**Architecture:**
```
Enter your image prompt: Modern glass skyscraper reflecting sunset clouds, architectural photography
```

### Advanced Usage

You can modify the generation parameters in the code:

```python
image = generate_image(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,  # Higher = better quality, slower
    guidance_scale=7.5,      # Higher = closer to prompt
    seed=42                  # For reproducible results
)
```

## System Requirements

### Minimum Requirements (CPU)
- 8GB RAM
- 10GB free disk space
- Generation time: 2-10 minutes per image

### Recommended Requirements (GPU)
- NVIDIA GPU with 4GB+ VRAM
- 8GB+ RAM
- 10GB free disk space
- Generation time: 10-30 seconds per image

### Supported GPUs
- NVIDIA GTX 1060 6GB or better
- NVIDIA RTX series (all models)
- Tesla/Quadro series with 4GB+ VRAM

## Configuration

### Memory Optimization Settings

The application automatically applies optimizations based on your hardware:

**GPU Optimizations:**
```python
torch_dtype=torch.float16          # Half precision
pipe.enable_xformers_memory_efficient_attention()  # Memory efficient attention
pipe.enable_model_cpu_offload()    # CPU offloading
```

**CPU Configuration:**
```python
torch_dtype=torch.float32          # Full precision for stability
```

### Troubleshooting Common Issues

**GPU Out of Memory:**
- Reduce `num_inference_steps` to 15-20
- Restart runtime and try again
- Use CPU mode if GPU memory is insufficient

**Model Loading Errors:**
- Check internet connection
- Verify sufficient disk space (10GB+)
- Restart Python kernel/runtime

**Slow Generation:**
- Ensure GPU is properly enabled
- Check CUDA installation
- Consider using Google Colab Pro for better GPUs

## Model Information

- **Model**: RunwayML Stable Diffusion v1.5
- **Resolution**: 512x512 pixels
- **License**: CreativeML Open RAIL-M License
- **Model Size**: ~4GB download
- **Architecture**: Latent Diffusion Model

## Limitations

- **Fixed Resolution**: Currently generates 512x512 images only
- **Single Image Generation**: Generates one image per prompt
- **Memory Requirements**: Requires significant RAM/VRAM
- **Model Download**: Initial setup requires downloading 4GB+ of model data
- **Generation Time**: CPU generation can be very slow (2-10 minutes)
- **Content Filtering**: Some prompts may be filtered by the model's safety checker

## Performance Metrics

**Google Colab (T4 GPU):**
- Generation time: ~15-25 seconds
- Memory usage: ~3-4GB VRAM

**Local RTX 3070:**
- Generation time: ~8-12 seconds
- Memory usage: ~4-5GB VRAM

**CPU (Intel i7):**
- Generation time: ~3-8 minutes
- Memory usage: ~6-8GB RAM

## Future Enhancements

- [ ] Support for different image resolutions
- [ ] Batch generation capabilities
- [ ] Web interface using Streamlit/Gradio
- [ ] Image-to-image generation
- [ ] Multiple model support (DALL-E, Midjourney-style)
- [ ] Advanced parameter tuning interface
- [ ] Image saving with metadata

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The Stable Diffusion model is licensed under the CreativeML Open RAIL-M License.

## Credits

- **[RunwayML](https://runwayml.com/)**: Stable Diffusion v1.5 model
- **[Hugging Face](https://huggingface.co/)**: Diffusers library and model hosting
- **[Stability AI](https://stability.ai/)**: Original Stable Diffusion research
- **[Google Colab](https://colab.research.google.com/)**: Development and testing environment

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Ensure your system meets the minimum requirements
3. Verify GPU drivers are up to date (for GPU usage)
4. Create an issue on GitHub with detailed error logs

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

**⭐ Star this repository if you found it helpful!**

*Made with ❤️ for the AI art generation community*
