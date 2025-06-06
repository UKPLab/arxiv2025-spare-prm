import re
import os
import json
import time
from functools import partial
from copy import deepcopy

import openai

from .model_utils import (
    retry,
    get_generation_count,
    generate_and_combine,
    postprocess_output_texts,
)


RETRY_SEC_PAT = re.compile(r"(?<=retry after)\s*\d+\s*(?=second)")

OPENAI_CLIENT_CONFIG = {
    "api_version": "2023-05-15",
    "api_key": os.getenv("AZURE_OPENAI_KEY"),
    "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
}

"""
https://platform.openai.com/docs/api-reference/chat
Recommended to either alter top_p or temperature but not both.
top_p: default=1, min=0, max=1, lesser the value, greedier the strategy
temperature: default=1, min=0, max=2, lesser the value, greedier the strategy
"""
OPENAI_GENERATION_CONFIG = dict(
    model="text-davinci-003",
    n=1,  # number of completions
    max_tokens=64,  # tokens in model response, high for cot, low otherwise e.g. 32
    top_p=1,  # sample from tokens percentile
    temperature=1,  # modifies proba values
    frequency_penalty=0,
    presence_penalty=0,
    stop=None,  # or ["\n"] in case of code-*-* models e.g. code-davinci-002
)


class OpenAILM:
    def __init__(self, client="AzureOpenAI", config=OPENAI_CLIENT_CONFIG):

        self.model = getattr(openai, client)(**config)
        self.reset()

        self.blank_chat_output = {
            "choices": [
                {
                    "finish_reason": "",
                    "index": 0,
                    "message": {
                        "content": "",
                        "role": "assistant",
                        "function_call": None,
                        "tool_calls": None,
                    },
                    "logprobs": None,
                },
            ],
            "created": 0,
            "id": "",
            "model": "",
            "object": "",
            "usage": {
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "total_tokens": 0,
            },
            "system_fingerprint": None,
        }

    def reset(self):
        self.stored_output = []
        self.error_handlers = {}

    def _error_as_output_handler(
        self,
        error,
        model="gpt-4",
        n=1,
        try_idx=0,
    ):
        can_return = True  # provides signal to retrying function to break and return
        error_str = repr(error)

        print(f"{error_str}\nContinuing by omitting and outputting error as content.")

        output = deepcopy(self.blank_chat_output)
        output["model"] = model

        choice = output["choices"][0]
        choice["finish_reason"] = "error"
        choice["message"]["content"] = error_str

        output["choices"] = [choice for _ in range(n)]

        return output, can_return

    def _multiple_tries_handler(
        self,
        error,
        error_as_output_handler=False,
        sleep_seconds=60,
        num_tries=5,
        try_idx=0,
    ):
        r"""Keep sleep_seconds >= 60 to retry openai api"""

        # can_return provides signal to retrying function to break and return
        output, can_return = deepcopy(self.blank_chat_output), False

        if error_as_output_handler in (None, True):
            error_as_output_handler = self._error_as_output_handler

        if try_idx == num_tries - 1:
            if error_as_output_handler:
                output, can_return = error_as_output_handler(error)
            else:
                raise error

        else:
            error_str = repr(error)

            suggested_sleep = RETRY_SEC_PAT.search(error_str)
            if suggested_sleep:
                # 1 sec additional buffer
                suggested_sleep = int(suggested_sleep.group(0).strip()) + 1
            else:
                suggested_sleep = float("inf")

            sleep_seconds = min(sleep_seconds, suggested_sleep)

            print(f"{error_str}\nWaiting for {sleep_seconds} seconds and resuming...")
            time.sleep(sleep_seconds)

        return output, can_return

    def _get_error_handlers(self, model="gpt-4", n=1, sleep_seconds=60, num_tries=5):

        error_as_output_handler = partial(
            self._error_as_output_handler, model=model, n=n
        )

        # provide error_as_output_handler or mark as None to handle tries exhaustion
        multiple_tries_handler = partial(
            self._multiple_tries_handler,
            # error_as_output_handler=error_as_output_handler,
            sleep_seconds=sleep_seconds,
            num_tries=num_tries,
        )

        handlers = {
            openai.BadRequestError: error_as_output_handler,
            openai.RateLimitError: multiple_tries_handler,
        }

        # # currently just multiple tries and then logged pass but not throwing error
        # # might need to rerun for those examples which were passed and logged
        # handlers = multiple_tries_handler

        return handlers

    def _complete_chat(
        self,
        messages,
        generation_config=OPENAI_GENERATION_CONFIG,
        dump_json=True,
    ):
        output = self.model.chat.completions.create(
            messages=messages, **generation_config
        )

        if dump_json:
            output = json.loads(output.model_dump_json())
        return output

    def _complete_text(
        self,
        batch,
        generation_config=OPENAI_GENERATION_CONFIG,
        dump_json=True,
    ):
        output = self.model.completions.create(prompt=batch, **generation_config)
        if dump_json:
            output = json.loads(output.model_dump_json())
        return output

    def _generate(
        self,
        batch,
        is_chat_completion=True,
        generation_config=OPENAI_GENERATION_CONFIG,
        process_text=lambda x: x.strip(),
        store_output=True,
        num_tries=5,
        error_handlers={},
    ):

        if is_chat_completion:
            output = []

            for messages in batch:

                out = retry(
                    self._complete_chat,
                    num_tries=num_tries,
                    error_handlers=error_handlers,
                    messages=messages,
                    generation_config=generation_config,
                )

                output.append(out)

            output_texts = [
                o["message"].get("content", "")
                for out in output
                for o in out["choices"]
            ]

        else:
            output = retry(
                self._complete_text,
                num_tries=num_tries,
                error_handlers=error_handlers,
                batch=batch,
                generation_config=generation_config,
            )

            output_texts = [o["text"] for o in output["choices"]]

        if store_output:
            self.stored_output.extend(output)

        output = postprocess_output_texts(
            output_texts=output_texts,
            remove_start_texts=None,  # pass if needs to be removed from output
            n_sequence_per_input=generation_config.get("n", 1),
            process_text=process_text,
        )

        return output

    def generate(
        self,
        batch,
        total_generation=1,
        generation_config=OPENAI_GENERATION_CONFIG,
        process_text=lambda x: x.strip(),
        store_output=True,
        num_tries=5,
        retry_interval_seconds=60,
        cache_error_handlers=False,
    ):

        if isinstance(batch, str) or (
            isinstance(batch, list) and isinstance(batch[0], dict)
        ):
            batch = [batch]

        model = generation_config.get("model", "gpt-35-turbo")
        n = generation_config.get("n", 1)
        is_chat_completion = "gpt-35-turbo" in model or "gpt-4" in model

        error_handlers = self.error_handlers
        if not error_handlers:

            error_handlers = self._get_error_handlers(
                model=model,
                n=n,
                sleep_seconds=retry_interval_seconds,
                num_tries=num_tries,
            )

            if cache_error_handlers:
                self.error_handlers = error_handlers

        generation_count = get_generation_count(
            total_generation=total_generation, num_return_sequences=n
        )

        generation_config_copy = deepcopy(generation_config)

        # for try_idx in range(num_tries):
        #     try:
        #         # output = generate_and_combine(
        #         #     self._generate,
        #         #     batch,
        #         #     generation_count=generation_count,
        #         #     total_generation=total_generation,
        #         #     is_chat_completion=is_chat_completion,
        #         #     # generation_config=generation_config,
        #         #     generation_config=generation_config_copy,
        #         #     process_text=process_text,
        #         #     store_output=store_output,
        #         #     num_tries=num_tries,
        #         #     error_handlers=error_handlers,
        #         # )

        #     except Exception as err:
        #         if try_idx < num_tries - 1 and isinstance(err, openai.RateLimitError):

        #             if n > 1:
        #                 error_handlers[openai.RateLimitError](err, try_idx=try_idx)

        #                 print(
        #                     f"RateLimitError occured. "
        #                     f"Trying with `n={n//2}` instead of `n={n}`"
        #                 )

        #                 print(
        #                     f"prev args:\n"
        #                     f"total_generation={total_generation}\n"
        #                     f"generation_config={generation_config_copy}"
        #                 )

        #                 n = n // 2

        #                 generation_count = get_generation_count(
        #                     total_generation=total_generation,
        #                     num_return_sequences=n,
        #                 )

        #                 generation_config_copy["n"] = n

        #                 print(
        #                     f"prev args:\n"
        #                     f"total_generation={total_generation}\n"
        #                     f"generation_config={generation_config_copy}"
        #                 )
        #             else:
        #                 raise err
        #         else:
        #             raise err

        output = generate_and_combine(
            self._generate,
            batch,
            generation_count=generation_count,
            total_generation=total_generation,
            is_chat_completion=is_chat_completion,
            # generation_config=generation_config,
            generation_config=generation_config_copy,
            process_text=process_text,
            store_output=store_output,
            num_tries=num_tries,
            error_handlers=error_handlers,
        )

        return output
