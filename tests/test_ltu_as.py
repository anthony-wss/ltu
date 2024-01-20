import sp_llm.ltu_as as ltu_as
from sp_llm.ltu_as.predict import predict
import os.path as osp
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def test_predict(capsys):

    model, tokenizer = ltu_as.load_model_with_tknzer(device="cuda")

    audio_path = osp.join(osp.dirname(__file__), "typing.wav")
    question = "What is this sound?"
    ans = predict(model, tokenizer, audio_path, question, prompt_template="alpaca_short", device="cuda")
    with capsys.disabled():
        print("Output:", ans)
    assert len(ans) > 0
