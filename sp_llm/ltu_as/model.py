import torch.nn as nn
from sp_llm.ltu_as.predict import load_audio_trans
import torch
from transformers.models.llama.modeling_llama import ATModel, Linear
import os.path as osp


def load_audio_tokenizer():
    audio_encoder = torch.load(osp.join(osp.dirname(__file__), "../../pretrained_mdls/audio_encoder.pt"))
    audio_proj = torch.load(osp.join(osp.dirname(__file__), "../../pretrained_mdls/audio_proj.pt"))
    return AudioTokenizer(audio_encoder, audio_proj)


class AudioTokenizer(nn.Module):
    def __init__(self, audio_encoder, audio_proj):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.audio_proj = audio_proj
        self.load_audio_trans = load_audio_trans

    def forward(self, audio_file):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        audio_input, _ = self.load_audio_trans(audio_file, device=device)
        audio_input = audio_input.unsqueeze(0)
        self.audio_encoder.to(audio_input.device)
        audio_input = self.audio_encoder(audio_input)  # [B, 25, 1280]
        assert audio_input.shape[1] == 25
        self.audio_proj.to(audio_input.device)
        audio_input = self.audio_proj(audio_input)  # [B, 25, 4096]
        return audio_input
