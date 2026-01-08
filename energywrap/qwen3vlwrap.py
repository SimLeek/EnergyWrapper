"""
Load Qwen3-VL-2B-Instruct with Energy Dynamics wrapper.

This script loads the Qwen3-VL-2B-Instruct model and wraps it with energy dynamics
to apply biological neural dynamics to all hidden neurons.
"""

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from typing import Optional
import logging

# Import the energy wrapper (assumes it's in the same directory or installed)
#from automatic_energy_wrapper_qwen3 import wrap_qwen3vl_energy_dynamics
import manual_qwen3_vl
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_qwen3vl_with_energy_dynamics(
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.float32,
        delta: Optional[float] = None,
        gamma: float = 0.05,
        lambda_kl: float = 0.01,
        lambda_l1: float = 0.005,
        beta: float = 0.05,
):
    """
    Load Qwen3-VL-2B-Instruct model with energy dynamics wrapper.

    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on ('cuda', 'cpu', 'mps')
        torch_dtype: Data type for model weights (torch.float32, torch.bfloat16, torch.float16)
        delta: Energy increment per step (None = auto: 1/hidden_dim)
        gamma: Energy drain factor [0.01, 0.5]
        lambda_kl: KL-divergence weight [0.001, 0.1]
        lambda_l1: L1 regularization weight [0.001, 0.1]
        beta: Target sparsity (~5% active neurons)
        load_in_8bit: Use 8-bit quantization (requires bitsandbytes)
        load_in_4bit: Use 4-bit quantization (requires bitsandbytes)

    Returns:
        wrapped_model: Energy-wrapped Qwen3VL model
        processor: Qwen3VL processor for input preprocessing
    """

    logger.info(f"Loading {model_name}...")

    # Prepare loading kwargs
    load_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto" if device == "cuda" else None,
    }

    # Load the base model
    try:
        model = manual_qwen3_vl.Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            **load_kwargs
        )
        logger.info(f"Wrapped model loaded successfully on {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Load processor
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        logger.info("Processor loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load processor: {e}")
        raise

    # Identify output layers (lm_head is the final output layer)
    output_layers = [model.lm_head]
    logger.info(f"Identified {len(output_layers)} output layer(s)")

    # Move to device if not using device_map
    if device != "cuda" or not load_kwargs.get("device_map"):
        model = model.to(device)
        logger.info(f"Model moved to {device}")

    return model, processor


def example_usage():
    """Example of how to use the loaded model."""

    # Load model with energy dynamics
    model, processor = load_qwen3vl_with_energy_dynamics(
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
        gamma=0.05,
        lambda_kl=0.01,
        lambda_l1=0.005,
        beta=0.05
    )

    # Example: Process an image with text
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Prepare inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Generate
    logger.info("Generating response...")
    generated_ids = model.generate(**inputs, max_new_tokens=128)

    # Get auxiliary loss from energy dynamics
    aux_loss = model.aux_loss
    logger.info(f"Auxiliary energy loss: {aux_loss.item():.4f}")

    # Decode output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    logger.info(f"Generated text: {output_text[0]}")

    return model, processor, output_text


def process_vision_info(messages):
    """Helper to extract images and videos from messages."""
    image_inputs, video_inputs = None, None
    for message in messages:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "image":
                    image_inputs = [ele["image"]]
                elif ele["type"] == "video":
                    video_inputs = [ele["video"]]
    return image_inputs, video_inputs


def train_with_energy_dynamics(
        model,
        processor,
        train_data,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        aux_loss_weight: float = 0.1,
):
    """
    Example training loop with energy dynamics auxiliary loss.

    Args:
        model: Energy-wrapped Qwen3VL model
        processor: Qwen3VL processor
        train_data: Training dataset
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        aux_loss_weight: Weight for auxiliary energy loss
    """
    from torch.optim import AdamW

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        total_aux_loss = 0

        for batch in train_data:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(**batch)

            # Main loss (language modeling)
            main_loss = outputs.loss

            # Auxiliary energy loss
            aux_loss = model.aux_loss

            # Combined loss
            total_loss_batch = main_loss + aux_loss_weight * aux_loss

            # Backward pass
            total_loss_batch.backward()
            optimizer.step()

            total_loss += main_loss.item()
            total_aux_loss += aux_loss.item()

        avg_loss = total_loss / len(train_data)
        avg_aux_loss = total_aux_loss / len(train_data)

        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"  Average main loss: {avg_loss:.4f}")
        logger.info(f"  Average auxiliary loss: {avg_aux_loss:.4f}")

    return model


if __name__ == "__main__":
    # Load model with energy dynamics
    logger.info("Starting Qwen3-VL with Energy Dynamics loader...")

    try:
        model, processor = load_qwen3vl_with_energy_dynamics(
            model_name="Qwen/Qwen3-VL-2B-Instruct",
            #device="cuda" if torch.cuda.is_available() else "cpu",
            device = "cpu",
            torch_dtype=torch.bfloat16,
            gamma=0.05,
            lambda_kl=0.01,
            lambda_l1=0.005,
            beta=0.05
        )

        logger.info("Model loaded and wrapped successfully!")
        logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        logger.info(f"Energy buffers: {len(model.energy_buffers)}")

        # Print some statistics about wrapped layers
        logger.info("\nWrapped layer statistics:")
        for i, (name, module) in enumerate(model.hidden_layers[:5]):  # Show first 5
            logger.info(f"  Layer {i}: {name} ({type(module).__name__}) - {model.hidden_dims[i]} neurons")
        if len(model.hidden_layers) > 5:
            logger.info(f"  ... and {len(model.hidden_layers) - 5} more layers")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    logger.info("\nModel ready for inference or training!")
    logger.info("Use model.aux_loss to access the auxiliary energy dynamics loss during training.")