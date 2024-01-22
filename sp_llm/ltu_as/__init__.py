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
from .utils.prompter import Prompter
import numpy as np
import datetime
import re
import os.path as osp
import warnings


def convert_params_to_float32(model):
    for name, param in model.named_parameters():
        if "audio_encoder" in name and "ln" in name:
            if param.dtype == torch.float16:
                print(f"Converting parameter '{name}' to float32")
                param.data = param.data.float()
        if "audio_proj" in name:
            if param.dtype == torch.float16:
                print(f"Converting parameter '{name}' to float32")
                param.data = param.data.float()


def load_model_with_tknzer(device="cpu"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # do not change this, this will load llm
        base_model = osp.join(osp.dirname(__file__), "../../pretrained_mdls/vicuna_ltuas/")
        prompt_template = "alpaca_short"
        # change this to your checkpoint
        eval_mdl_path = osp.join(osp.dirname(__file__), "../../pretrained_mdls/ltuas_long_noqa_a6.bin")
        eval_mode = 'joint'
        prompter = Prompter(prompt_template)
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
        if device == 'cuda':
            model = LlamaForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.float16)
        else:
            model = LlamaForCausalLM.from_pretrained(base_model)
        convert_params_to_float32(model)

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, config)

        # temp, top_p, top_k = 0.1, 0.95, 500

        state_dict = torch.load(eval_mdl_path, map_location='cpu')
        miss, unexpect = model.load_state_dict(state_dict, strict=False)

        model.is_parallelizable = True
        model.model_parallel = True

        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        return model, tokenizer

    # model.eval()

    # eval_log = []
    # cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # log_save_path = './inference_log/'
    # if os.path.exists(log_save_path) == False:
    #     os.mkdir(log_save_path)
    # log_save_path = log_save_path + cur_time + '.json'
