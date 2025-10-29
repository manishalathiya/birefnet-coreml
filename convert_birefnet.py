import torch
import coremltools as ct
from transformers import AutoModelForImageSegmentation

MODEL_ID = "ZhengPeng7/BiRefNet_1K_Base"

print("🚀 Loading model:", MODEL_ID)
model = AutoModelForImageSegmentation.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

model.eval()

# Example input
example_input = torch.rand(1, 3, 512, 512)

print("⚡ Tracing model...")
traced = torch.jit.trace(model, example_input)
traced.save("birefnet_lite_traced.pt")

print("🧠 Converting to CoreML with optimization...")
mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(name="input_image", shape=example_input.shape, scale=1/255.0)],
    convert_to="mlprogram",
    compute_units=ct.ComputeUnit.ALL,
)

# Optimize model precision for iOS
print("📦 Optimizing for iOS (FP16 precision)...")
mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(
    mlmodel, nbits=16
)

mlmodel.save("BiRefNetBackgroundRemoval.mlmodel")

print("✅ Conversion complete — saved as BiRefNetBackgroundRemoval.mlmodel")
