import torch
import torch._dynamo
import sys
import os


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.1'
import folder_paths
from pathlib import Path
import json
import random
import numpy as np
import comfy.utils
from tqdm import tqdm

# Add the correct moshi source directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "moshi_src"))

from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import TTSModel

# Monkey-patch the problematic function in the moshi library
# This prevents a PyTorch compilation error on Windows by disabling
# the JIT compiler for this specific function.
try:
    import moshi.modules.rope
    torch._dynamo.disable(moshi.modules.rope.apply_rope)
except (ImportError, AttributeError) as e:
    print(f"KyutaiTTS Node: Could not patch moshi.modules.rope.apply_rope. If you encounter an OverflowError, this may be the cause. Error: {e}")

class KyutaiTTS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hey there! How are you?"}),
                "model_path": ("STRING", {"default": "", "multiline": False, "folder_input": True}),
                "voice_model": (folder_paths.get_filename_list("loras"), ),
                "device": (["cuda", "cpu"],),
                "n_q": ("INT", {"default": 32}),
                "temp": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.1}),
                "cfg_coef": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "padding_between": ("INT", {"default": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("AUDIO", )
    FUNCTION = "generate"
    CATEGORY = "Kyutai"

    def generate(self, text, model_path, voice_model, device, n_q, temp, cfg_coef, padding_between, seed):


        def seed_all(seed):
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        seed_all(seed)

        device = torch.device(device)
        
        # Create CheckpointInfo from the local model path
        full_model_path = model_path
        if not full_model_path or not os.path.isdir(full_model_path):
            raise FileNotFoundError(f"Model directory not found: {full_model_path}")

        # Define expected file names within the model directory
        moshi_weights_path = os.path.join(full_model_path, "dsm_tts_1e68beda@240.safetensors")
        if not os.path.exists(moshi_weights_path):
            raise FileNotFoundError(f"Moshi weights (dsm_tts_1e68beda@240.safetensors) not found in {full_model_path}")

        mimi_weights_path = os.path.join(full_model_path, "tokenizer-e351c8d8-checkpoint125.safetensors")
        if not os.path.exists(mimi_weights_path):
            raise FileNotFoundError(f"Mimi weights (tokenizer-e351c8d8-checkpoint125.safetensors) not found in {full_model_path}")

        tokenizer_path = os.path.join(full_model_path, "tokenizer_spm_8k_en_fr_audio.model")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer (tokenizer_spm_8k_en_fr_audio.model) not found in {full_model_path}")
            
        config_path = os.path.join(full_model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {full_model_path}")

        with open(config_path, 'r') as f:
            raw_config = json.load(f)

        # Extract specific configs for CheckpointInfo and remove them from lm_config
        tts_config = raw_config.get("tts_config", {})
        stt_config = raw_config.get("stt_config", {})
        lm_gen_config = raw_config.get("lm_gen_config", {})
        model_id = raw_config.get("model_id", {})
        model_type = raw_config.get("model_type", "moshi") # Extract model_type

        lm_config = dict(raw_config) # Create a copy for lm_config

        # Remove keys not meant for LMModel from lm_config
        lm_config.pop("tts_config", None)
        lm_config.pop("stt_config", None)
        lm_config.pop("lm_gen_config", None)
        lm_config.pop("model_id", None)
        lm_config.pop("moshi_name", None)
        lm_config.pop("mimi_name", None)
        lm_config.pop("tokenizer_name", None)
        lm_config.pop("lora_name", None)
        lm_config.pop("model_type", None) # Remove model_type from lm_config

        checkpoint_info = CheckpointInfo(
            moshi_weights=Path(moshi_weights_path),
            mimi_weights=Path(mimi_weights_path),
            tokenizer=Path(tokenizer_path),
            lm_config=lm_config,
            raw_config=raw_config,
            tts_config=tts_config,
            stt_config=stt_config,
            lm_gen_config=lm_gen_config,
            model_id=model_id,
            model_type=model_type # Pass model_type to CheckpointInfo
        )

        tts_model = TTSModel.from_checkpoint_info(
            checkpoint_info, n_q=n_q, temp=temp, device=device
        )

        entries = tts_model.prepare_script([text], padding_between=padding_between)
        
        voice_path = folder_paths.get_full_path("loras", voice_model)
        if not voice_path or not os.path.exists(voice_path):
            raise FileNotFoundError(f"Voice model not found: {voice_model}")

        condition_attributes = tts_model.make_condition_attributes(
            [voice_path], cfg_coef=cfg_coef
        )

        # --- Step 1: Generate audio token frames ---
        frames_list = []
        # A more accurate estimation including initial/final padding, model delays, and an empirical fudge factor.
        initial_padding = tts_model.machine.initial_padding
        final_padding = tts_model.final_padding
        delay_steps = tts_model.delay_steps
        # Add a small fudge factor for each word to account for un-predictable discretionary padding.
        FUDGE_FACTOR_PER_WORD = 1
        word_steps = sum(len(entry.tokens) + entry.padding + FUDGE_FACTOR_PER_WORD for entry in entries)
        total_steps = initial_padding + word_steps + delay_steps + final_padding
        gen_pbar = comfy.utils.ProgressBar(total_steps)
        with tqdm(total=total_steps, desc="Generating Tokens") as pbar_cmd_gen:
            def _on_frame_collect(frame):
                if (frame != -1).all():
                    frames_list.append(frame.clone())
                    # Update by 1 for each frame generated.
                    gen_pbar.update(1)
                    pbar_cmd_gen.update(1)

            all_entries = [entries]
            all_condition_attributes = [condition_attributes]
            with tts_model.mimi.streaming(len(all_entries)):
                tts_model.generate(all_entries, all_condition_attributes, on_frame=_on_frame_collect)

        # --- Step 2: Decode frames to PCM audio ---
        pcms = []
        if frames_list:
            decode_pbar = comfy.utils.ProgressBar(len(frames_list))
            with tqdm(total=len(frames_list), desc="Decoding Audio") as pbar_cmd_decode:
                for frame in frames_list:
                    pcm = tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                    pcms.append(np.clip(pcm[0], -1, 1)[np.newaxis, :])
                    decode_pbar.update(1)
                    pbar_cmd_decode.update(1)

        # --- Step 3: Concatenate audio chunks ---
        audio = np.concatenate(pcms, axis=-1) if pcms else np.array([])
        
        print(f"KyutaiTTS Node: Outputting audio with sample rate: {tts_model.mimi.sample_rate}")
        # Return audio in the format expected by ComfyUI's AUDIO type
        return ({"waveform": torch.from_numpy(audio), "sample_rate": tts_model.mimi.sample_rate},)

NODE_CLASS_MAPPINGS = {
    "KyutaiTTS": KyutaiTTS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KyutaiTTS": "KyutaiTTS",
}