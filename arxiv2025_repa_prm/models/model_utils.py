from functools import partial

import numpy as np
import httpx
import openai
from accelerate import Accelerator, infer_auto_device_map, init_empty_weights
from accelerate.utils import is_xpu_available
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    LlamaTokenizer,
    LlamaTokenizerFast,
)

from .templates import llama_chat_template


ERROR_SAMPLE = {
    openai.RateLimitError: openai.RateLimitError(
        message="Test RateLimitError",
        response=httpx.Response(
            status_code=httpx.codes.TOO_MANY_REQUESTS,
            request=httpx.Request(
                method="GET",
                url="www.dummy-test.com",
            ),
        ),
        body=None,
    ),
    openai.BadRequestError: openai.BadRequestError(
        message="Test BadRequestError",
        response=httpx.Response(
            status_code=httpx.codes.BAD_REQUEST,
            request=httpx.Request(
                method="GET",
                url="www.dummy-test.com",
            ),
        ),
        body=None,
    ),
    openai.APIConnectionError: openai.APIConnectionError(request=None),
}


def prepare_pad_token(tokenizer, model=None, pad_token="unk_token"):
    pad_token = tokenizer.special_tokens_map.get(pad_token, pad_token)
    if pad_token:
        tokenizer.add_special_tokens({"pad_token": pad_token})

        if model:
            if model.config.vocab_size != len(tokenizer):
                model.resize_token_embeddings(len(tokenizer))

            model.config.pad_token_id = tokenizer.pad_token_id
            if model.generation_config is not None:
                model.generation_config.pad_token_id = tokenizer.pad_token_id

        print(f"Pad token added:")
        print(
            f"pad_token: {tokenizer.pad_token}\n"
            f"pad_token_id: {tokenizer.pad_token_id}"
        )

    return tokenizer, model


def prepare_chat_template(tokenizer, chat_template=None):
    if chat_template:
        tokenizer.chat_template = chat_template
    elif not tokenizer.chat_template:
        tokenizer.chat_template = getattr(tokenizer, "default_chat_template", None)
        if isinstance(tokenizer, (LlamaTokenizer, LlamaTokenizerFast)):
            # rectify incorrect spacing between eos and bos
            tokenizer.chat_template = partial(llama_chat_template, tokenizer)()
    return tokenizer


def process_device_map(
    device_map="auto", model=None, modelclass=AutoModelForCausalLM, **kwargs
):
    r"""
    Currently supports only Causal LM when model is specified for device_map=infer
    if distributed data parallel, then copy the model to each device
    taken from trl's sft example script:
    https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py
    """

    if device_map == "ddp":

        device_map = (
            {"": f"xpu:{Accelerator().local_process_index}"}
            if is_xpu_available()
            else {"": Accelerator().local_process_index}
        )

    elif device_map == "infer" and model:
        config = AutoConfig.from_pretrained(model)
        with init_empty_weights():
            model = modelclass.from_config(config)

        device_map = infer_auto_device_map(
            model, no_split_module_classes=model._no_split_modules, **kwargs
        )

    return device_map


def retry(func, num_tries=5, error_handlers={}, **kwargs):

    if callable(error_handlers):
        error_handlers = {"default": error_handlers}

    num_tries = max(num_tries, 1)  # minimum once
    for try_idx in range(num_tries):

        try:
            output = func(**kwargs)
            can_return = True

        except Exception as err:
            raise err
            # handler = error_handlers.get(
            #     err.__class__, error_handlers.get("default")
            # )

            # if handler:
            #     output, can_return = handler(err, try_idx=try_idx)
            # else:
            #     raise err

        if not can_return:
            print(f"Attempted {try_idx+1} out of {num_tries} times already.")
        else:
            if try_idx == num_tries - 1:
                print(f"Exhausted num_tries={num_tries}.")
            return output


def get_generation_count(total_generation=1, num_return_sequences=1):

    if total_generation < num_return_sequences:
        print(
            f"`total_generation` < `num_return_sequences`."
            "Setting it equal to `num_return_sequences`"
        )
        total_generation = num_return_sequences

    generation_count = int(np.ceil(total_generation / num_return_sequences))

    return generation_count


def generate_and_combine(
    gen_fn,
    batch,
    generation_count,
    total_generation,
    **gen_kwargs,
):
    output = [[] for i in range(len(batch))]

    for i in range(generation_count):
        cur_out = gen_fn(batch=batch, **gen_kwargs)

        # append subsequent regenerations
        output = [all_out + out for all_out, out in zip(output, cur_out)]

    # discard extra generation in the last generation_count
    output = [i[:total_generation] for i in output]

    return output


def postprocess_output_texts(
    output_texts,
    remove_start_texts=None,  # pass if needs to be removed from output
    n_sequence_per_input=1,
    process_text=lambda x: x.strip(),
):
    num_inputs = len(output_texts) // n_sequence_per_input
    processed = [[] for _ in range(num_inputs)]

    for i, t in enumerate(output_texts):
        if remove_start_texts:
            text_to_remove = remove_start_texts[i // n_sequence_per_input]
            t = t[len(text_to_remove) :]
        t = process_text(t)
        processed[i // n_sequence_per_input].append(t)

    return processed
