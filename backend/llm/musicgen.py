import scipy
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

prompt = ["upbeat vocaloid song with energetic beats and catchy melody, english lyrics about friendship and adventure"]
inputs = processor(text=prompt, return_tensors="pt")

with torch.no_grad():
    audio_values = model.generate(**inputs, max_new_tokens=512)

sampling_rate = model.config.audio_encoder.sampling_rate
scipy.io.wavfile.write("musicgen_output.wav", rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())