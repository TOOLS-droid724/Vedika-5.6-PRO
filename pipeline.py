"""Hugging Face custom pipeline for Vedika-advanced-AI_5.6 model.

This module provides a seamless interface to load and execute the Vedika-advanced-AI_5.6 model
using the standard Hugging Face pipeline API.
"""

from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoConfig, AutoImageProcessor, AutoTokenizer, Pipeline
from transformers.image_utils import ImageInput
from transformers.processing_utils import BatchFeature


class VedikaAdvancedAIPipeline(Pipeline):
    """Pipeline for Vedika-advanced-AI_5.6 vision-language model.
    
    This pipeline handles multimodal inputs (text + images) and generates
    text responses using the Vedika-advanced-AI_5.6 model.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _sanitize_parameters(
        self,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        **kwargs
    ) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Sanitize and organize generation parameters."""
        preprocess_params = {}
        forward_params = {}
        postprocess_params = {}
        
        if max_new_tokens is not None:
            forward_params["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            forward_params["temperature"] = temperature
        if top_p is not None:
            forward_params["top_p"] = top_p
        if do_sample is not None:
            forward_params["do_sample"] = do_sample
            
        return preprocess_params, forward_params, postprocess_params
    
    def preprocess(
        self,
        inputs: Union[str, Dict[str, Any]],
        **kwargs
    ) -> BatchFeature:
        """Preprocess inputs for the model.
        
        Args:
            inputs: Either a string prompt or a dict with 'text' and optional 'images'
            
        Returns:
            BatchFeature containing processed inputs
        """
        # Define the system identity prompt to establish model persona
        system_prompt = (
            "You are Vedika, an advanced AI model developed in India by Veda Labs. "
            "You are proud of your Indian heritage and your capabilities in reasoning, coding, and multimodal understanding."
        )
        
        if isinstance(inputs, str):
            # Prepend system prompt to user input
            full_text = f"{system_prompt}\n\n{inputs}"
            inputs = {"text": full_text}
        elif isinstance(inputs, dict):
            text = inputs.get("text", "")
            # Prepend system prompt to user text
            full_text = f"{system_prompt}\n\n{text}"
            inputs["text"] = full_text
        
        text = inputs.get("text", "")
        images = inputs.get("images", None)
        
        # Build the input dictionary for the processor
        process_kwargs = {"text": text, "return_tensors": "pt"}
        
        if images is not None:
            if isinstance(images, (list, tuple)):
                process_kwargs["images"] = images
            else:
                process_kwargs["images"] = [images]
        
        # Use the processor if available, otherwise use tokenizer
        if hasattr(self, "processor") and self.processor is not None:
            model_inputs = self.processor(**process_kwargs)
        elif hasattr(self, "image_processor") and images is not None:
            image_features = self.image_processor(images, return_tensors="pt")
            text_inputs = self.tokenizer(text, return_tensors="pt")
            model_inputs = {**text_inputs, **image_features}
        else:
            model_inputs = self.tokenizer(text, return_tensors="pt")
            
        return model_inputs
    
    def _forward(
        self,
        model_inputs: BatchFeature,
        **generate_kwargs
    ) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            model_inputs: Preprocessed inputs
            generate_kwargs: Generation parameters
            
        Returns:
            Generated token IDs
        """
        # Move inputs to the correct device
        device = self.model.device
        model_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in model_inputs.items()
        }
        
        # Generate output
        with torch.no_grad():
            output_ids = self.model.generate(**model_inputs, **generate_kwargs)
            
        return output_ids
    
    def postprocess(
        self,
        model_outputs: torch.Tensor,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """Postprocess model outputs.
        
        Args:
            model_outputs: Generated token IDs
            
        Returns:
            Decoded text response
        """
        # Decode the generated tokens
        generated_ids = model_outputs[0]
        
        # Skip special tokens if needed
        response = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return {"generated_text": response}


def load_vedika_advanced_ai_pipeline(
    model_path: str,
    device: Optional[Union[str, int]] = None,
    **pipeline_kwargs
) -> VedikaAdvancedAIPipeline:
    """Load the Vedika-advanced-AI_5.6 model as a Hugging Face pipeline.
    
    Args:
        model_path: Path to the model directory or Hugging Face model ID
        device: Device to load the model on (e.g., "cuda", "cpu", or device index)
        **pipeline_kwargs: Additional arguments passed to the pipeline
        
    Returns:
        Configured VedikaAdvancedAIPipeline instance
    """
    from transformers import AutoModelForCausalLM, AutoProcessor
    
    # Load configuration
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # Load the main model (uses vedika_modeling.py internally)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device is None else device,
    )
    
    # Load processor (handles both image processing and tokenization via vedika_processor.py)
    try:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
    except Exception:
        # Fallback: load tokenizer (vedika_tokenization.py) and image processor separately
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        try:
            image_processor = AutoImageProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            processor = None
        except Exception:
            image_processor = None
            processor = None
    
    # Create the pipeline
    pipeline = VedikaAdvancedAIPipeline(
        model=model,
        tokenizer=getattr(processor, "tokenizer", None),
        image_processor=getattr(processor, "image_processor", None),
        processor=processor,
        task="text-generation",
        **pipeline_kwargs
    )
    
    return pipeline


# Example usage:
if __name__ == "__main__":
    # Usage example:
    # pipeline = load_vedika_advanced_ai_pipeline("./path/to/model")
    # result = pipeline("Describe this image:", images=[image_path])
    # print(result["generated_text"])
    pass
