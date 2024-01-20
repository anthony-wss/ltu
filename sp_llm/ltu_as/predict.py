import json
import os
import torch
import time
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from sp_llm.ltu_as.utils.prompter import Prompter
import numpy as np
import datetime
import re
import skimage.measure
import sp_llm.whisper as whisper
import sp_llm.ltu_as.whisper as ltu_whisper


def trim_string(a):
    separator = "### Response:\n"
    trimmed_string = a.partition(separator)[-1]
    trimmed_string = trimmed_string.strip()
    return trimmed_string


text_cache = {}
def load_audio_trans(filename, device="cpu"):
    global text_cache
    whisper_text_model = whisper.load_model("tiny", device=device)
    whisper_feat_model = ltu_whisper.load_model(device=device)
    if filename not in text_cache:
        result = whisper_text_model.transcribe(filename)
        text = result["text"].lstrip()
        # text = remove_thanks_for_watching(result["text"].lstrip())
        text_cache[filename] = text
    else:
        text = text_cache[filename]
        print('using asr cache')
    _, audio_feat = whisper_feat_model.transcribe_audio(filename)
    audio_feat = audio_feat[0]
    audio_feat = torch.permute(audio_feat, (2, 0, 1)).detach().cpu().numpy()
    audio_feat = skimage.measure.block_reduce(audio_feat, (1, 20, 1), np.mean)
    audio_feat = audio_feat[1:]  # skip the first layer
    audio_feat = torch.FloatTensor(audio_feat)
    return audio_feat, text


def predict(model, tokenizer, audio_path, question, prompt_template="alpaca_short", device="cpu"):
    print('audio path, ', audio_path)
    begin_time = time.time()

    if audio_path is not None:
        cur_audio_input, cur_input = load_audio_trans(audio_path, device=device)
        if not torch.cuda.is_available():
            pass
        else:
            cur_audio_input = cur_audio_input.unsqueeze(0).half().to(device)

    instruction = question
    prompter = Prompter(prompt_template)
    prompt = prompter.generate_prompt(instruction, cur_input)
    print('Input prompt: ', prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    temp, top_p, top_k = 0.1, 0.95, 500

    generation_config = GenerationConfig(
        do_sample=True,
        temperature=temp,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=1.1,
        max_new_tokens=500,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
        num_return_sequences=1
    )

    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            audio_input=cur_audio_input,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=500,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    output = output[5:-4]
    end_time = time.time()
    print(trim_string(output))
    cur_res = {'audio_id': audio_path, 'instruction': instruction, 'input': cur_input, 'output': trim_string(output)}
    # eval_log.append(cur_res)
    # with open(log_save_path, 'w') as outfile:
    #     json.dump(eval_log, outfile, indent=1)
    print('eclipse time: ', end_time-begin_time, ' seconds.')
    return trim_string(output)