import yaml
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime

import peft
import numpy as np
from datasets import concatenate_datasets

from data.utils import (
    load_data,
    shuffle_select_data,
    create_chat_templated_text_data,
)
from data.preference_generator import (
    load_preference_data, 
    create_chat_templated_preference_prompt,
    process_before_train,
)
import models as GenerativeLM
from models.train import train_clm
from utils.log import init_neptune, get_gpu_info, save_and_upload_output


SEED = 42


def main(exp, run=None):

    if exp["data"]["load"].get("name") and exp["data"]["load"].get("source"):
        exp["data"]["load"]["source"]["name"] = exp["data"]["load"].get("name")
    
    source_data = load_data(**exp["data"]["load"]["source"])

    seed = exp.get("seed", SEED)
    rng = np.random.default_rng(seed=seed)
    train_output_path = exp["data"]["load"]["preference"].pop("train_output_path")

    preference_data = load_preference_data(
        train_output_path=train_output_path,
        source_data=source_data,
        data_name=exp["data"]["load"].get("name", []),
        seed=seed,
        rng=rng,
        **exp["data"]["load"]["preference"],
    )

    preference_data = {
        name: shuffle_select_data(
            data,
            seed=seed,
            shuffle=exp["data"].get("shuffle"),
            select=exp["data"].get("select"),  # None for whole dataset
        )[0]  # get the dataset but leave out the selection info
        for name, data in preference_data.items()
    }

    # ################## Write for Debugging ##################
    # for name, data in preference_data.items():
    #     for split, split_data in data.items():
    #         split_data.to_json(f"{exp['output_dir']}/{split}-{name}.jsonl")
    # ################## Write for Debugging ##################

    print(f"\nGPU info before model load:\n{get_gpu_info()}\n")

    model = getattr(GenerativeLM, exp["model"]["model_class"])(
        **exp["model"]["init_args"]
    )

    print(f"\nGPU info after model load:\n{get_gpu_info()}\n")

    preference_data = create_chat_templated_preference_prompt(
        preference_data=preference_data,
        chat_template_func=model.tokenizer.apply_chat_template,
        **exp["data"]["chat_template_args"],
    )

    preference_data = {
        split: concatenate_datasets([data[split] for data in preference_data.values()])
        for split in exp["data"]["chat_template_args"].get("prepare_splits", ["train"])
    }

    if fn_kwargs_list := exp["data"].get("process_before_train", []):
        preference_data = {
            split: process_before_train(split_data, fn_kwargs_list)
            for split, split_data in preference_data.items()
        }
    # import pdb; pdb.set_trace()

    training_args = exp["train"]["training"]
    training_args["output_dir"] = exp["output_dir"]

    trainer_args = exp["train"].get("trainer", {})
    if peft_config := exp.get("peft"):
        peft_config = getattr(peft, peft_config["config"])(**peft_config["config_args"])
    trainer_args["peft_config"] = peft_config

    trainer = train_clm(
        model=model.model,
        tokenizer=model.tokenizer,
        train_dataset=preference_data["train"],
        trainer_class=exp["train"]["class"],
        args_class=exp["train"]["args_class"],
        eval_dataset=preference_data.get("validation"),
        save_model=exp["output_dir"] / "final",
        callbacks=exp["train"].get("callbacks", []),
        training_args=training_args,
        trainer_args=trainer_args,
        run=run,
    )


def process_config_for_log(config):
    save_dir = Path(config["exp"]["output_dir"])

    model_name = config["exp"]["model"]["model_class"]
    if model_name == "OpenAILM":
        model_name = "openai"
    else:
        model_name = config["exp"]["model"]["init_args"]["model"]

    model_name = model_name.replace("/", "-")
    if model_name == "openai":
        model_name += "-" + config["exp"]["model"]["generation_config"]["model"]

    save_dir = save_dir / f"train-{model_name}-{str(datetime.now())}"
    config["exp"]["output_dir"] = save_dir
    return config


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Supervised finetuning of LLMs on spatial reasoning datasets."
    )

    parser.add_argument(
        "--config",
        default="config/train-po-config.yaml",
        help="path to config file specifying parameters of training.",
    )

    parser.add_argument(
        "--private",
        default="config/private-config.yaml",
        help="path to private file specifying api keys and tokens.",
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    with open(args.private) as f:
        private = yaml.safe_load(f)

    run = init_neptune(config.get("neptune"), private.get("neptune-api-token"))
    if run:
        run["reference/config.yaml"].upload(args.config)

    config = process_config_for_log(config)
    main(exp=config["exp"], run=run)
