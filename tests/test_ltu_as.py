import sp_llm.ltu_as as ltu_as
from sp_llm.ltu_as.predict import predict, predict_with_prefix
import os.path as osp
import torch
from sp_llm.ltu_as.model import load_audio_tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"


def test_predict(capsys):
    model, tokenizer = ltu_as.load_model_with_tknzer(device=device)
    audio_path = osp.join(osp.dirname(__file__), "typing.wav")
    question = "What is this sound?"
    ans = predict(model, tokenizer, audio_path, question, prompt_template="alpaca_short", device=device)
    with capsys.disabled():
        print("Output:", ans)
    assert len(ans) > 0
    del model


def test_predict_with_prefix(capsys):
    model, tokenizer = ltu_as.load_model_with_tknzer(device=device)
    audio_path = osp.join(osp.dirname(__file__), "typing.wav")
    prefix_inputs = '### Instruction:\nRepeat this sentence "'
    inputs = '"\nYou must NOT output any other words.\n\n### Response:\n'

    ans = predict_with_prefix(
        model=model,
        tokenizer=tokenizer,
        audio_path=audio_path,
        prefix_inputs=prefix_inputs,
        inputs=inputs,
        prompt_template="alpaca_short",
        device="cuda"
    )
    with capsys.disabled():
        print(ans)
    assert len(ans) > 0
    del model


def test_audio_encoder():
    audio_encoder = load_audio_tokenizer()
    audio_path = osp.join(osp.dirname(__file__), "typing.wav")
    out = audio_encoder(audio_path)
    assert out.shape == torch.Size([1, 25, 4096]), f"output size: {out.shape} is wrong"
    del audio_encoder
