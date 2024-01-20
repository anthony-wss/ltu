import sp_llm.whisper as ori_whisper
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


def test_encoder():
    model = ori_whisper.load_model("large-v2", device=device)
    audio = ori_whisper.load_audio(os.path.join(os.path.dirname(__file__), "typing.wav"))
    audio = ori_whisper.pad_or_trim(audio)
    mel = ori_whisper.log_mel_spectrogram(audio).to(model.device)
    emb = model.embed_audio(mel.unsqueeze(0))
    assert emb.shape == torch.Size([1, 1500, 1280])
