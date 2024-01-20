from sp_llm import whisper
import torch
import os
device = "cpu"


def test_encoder():
    model = whisper.load_model("tiny", device=device)
    audio = whisper.load_audio(os.path.join(os.path.dirname(__file__), "typing.wav"))
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    emb = model.embed_audio(mel.unsqueeze(0))
    assert emb.shape == torch.Size([1, 1500, 384])
