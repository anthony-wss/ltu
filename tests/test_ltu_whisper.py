import sp_llm.ltu_as.whisper as ltu_whisper
import os.path as osp
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def test_ltu_whisper():
    audio_file = osp.join(osp.dirname(__file__), "typing.wav")
    # audio = ltu_whisper.load_audio()
    # audio = ltu_whisper.pad_or_trim(audio)

    model = ltu_whisper.load_model(device=device)
    mel, audio_rep = model.transcribe_audio(audio_file)
    assert audio_rep.shape == torch.Size([1, 500, 1280, 33])
