from sp_llm.whisper import load_model
from sp_llm.ltu_as.predict import predict_with_prefix
import sp_llm.ltu_as as ltu_as


if __name__ == "__main__":
    model, tokenizer = ltu_as.load_model_with_tknzer(device="cuda")
    audio_path = "./tests/typing.wav"
    prefix = 'Repeat this sentence "'
    inputs = '"'
    ans = predict_with_prefix(
        model=model,
        tokenizer=tokenizer,
        audio_path=audio_path,
        prefix_inputs=prefix,
        question=inputs,
        prompt_template="alpaca_short",
        device="cuda"
    )
