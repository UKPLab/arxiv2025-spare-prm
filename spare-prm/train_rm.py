import yaml
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
from functools import partial
from collections import defaultdict

import peft
from datasets import Dataset, DatasetDict, concatenate_datasets

from data.utils import (
    load_data,
    load_model_output, 
    flatten_model_output,
    shuffle_select_data,
    field_stratified_select_data,
    create_chat_templated_text_data,
    merge_details_into_model_output,
)
from data.rm_utils import ground_truth_as_graded, create_step_labels
from utils.answer import extract_ans_from_output_files
from evaluation.evaluate import batch_compute_metrics
import models as GenerativeLM
from models.train import train_clm
from utils.log import init_neptune, get_gpu_info, save_and_upload_output


SEED = 42


def main(exp, run=None):

    src_dataset = load_data(**exp["data"]["load"])

    merge_args = exp["data"]["merge_output"]
    dataset = defaultdict(DatasetDict)

    # explicit generator function as Dataset.from_generator doesn't work with native generator
    # and Dataset.from_list exhausts memory
    def ex_gen(d):
        for i in d:
            yield i

    for split in ("train", "validation", "test"):
        if output_dir := merge_args.get(f"{split}_model_output"):

            if extract_args := merge_args.get("extract_args"):
                split_output = extract_ans_from_output_files(
                    input_dir=output_dir,
                    seed=exp.get("seed", SEED),
                    **extract_args
                )

            else:
                split_output = load_model_output(output_dir)

            if ground_truth_kwargs := merge_args.get("ground_truth_as_graded"):
                id_keys = merge_args["id_keys"]
                if isinstance(merge_args["id_keys"], dict):
                    id_keys = list(merge_args["id_keys"].values())
                
                split_output = ground_truth_as_graded(
                    split_output, id_keys=id_keys,** ground_truth_kwargs
                )

            split_output = merge_details_into_model_output(
                data=src_dataset, 
                model_output=split_output, 
                model_output_split=split,
                merge_keys=merge_args["merge_keys"],
                id_keys=merge_args["id_keys"],
            )

            split_output = {i: j[:100] for i,j in split_output.items()}
            # import pdb; pdb.set_trace()
            if metric_args := merge_args.get("compute_metrics_kwargs"):
                split_output = batch_compute_metrics(split_output, **metric_args)

            for name, data in split_output.items():
                
                if isinstance(model_output := data[0].get("model_output"), list):
                    data = flatten_model_output(
                        {"grouped": data}, flatten_key="model_output"
                    )["grouped"]
                
                data = Dataset.from_generator(partial(ex_gen, data))
                dataset[name][split] = data

    if step_tag_args := exp["data"].get("step_tag_args"):
        process_step_tag = partial(create_step_labels, **step_tag_args)
        
        dataset = {
            name: {
                split: split_data.map(process_step_tag) 
                for split, split_data in data.items()
            }
            for name, data in dataset.items()
        }

    if select_info := exp["data"].get("select"):
        for name in dataset.keys():
            if select_info["select_fn"] == "suffle_select":
                dataset[name], _ = shuffle_select_data(
                    dataset[name],
                    seed=exp.get("seed", SEED),
                    shuffle=exp["data"].get("shuffle"),
                    select=select_info.get("select_kwargs"),  # None for whole dataset
                )

            elif select_info["select_fn"] == "field_stratified_select":
                dataset[name], _ = field_stratified_select_data(
                    dataset[name],
                    seed=exp.get("seed", SEED),
                    shuffle=exp["data"].get("shuffle"),
                    **select_info.get("select_kwargs", {})
                )

    print(f"\nGPU info before model load:\n{get_gpu_info()}\n")

    model = getattr(GenerativeLM, exp["model"]["model_class"])(
        **exp["model"]["init_kwargs"]
    )

    print(f"\nGPU info after model load:\n{get_gpu_info()}\n")

    # import pdb; pdb.set_trace()
    dataset = create_chat_templated_text_data(
        dataset,
        chat_template_func=model.tokenizer.apply_chat_template,
        system_as_separate=exp["data"]["chat_template_args"].pop("system_as_separate", True),
        system_user_sep=exp["data"]["chat_template_args"].pop("system_user_sep", "\n\n"),
        **exp["data"]["chat_template_args"],
    )

    dataset = {
        split: concatenate_datasets([data[split] for data in dataset.values()]).shuffle(seed=exp.get("seed", SEED))
        for split in exp["data"]["chat_template_args"].get("prepare_splits", ["train"])
    }
    # import pdb; pdb.set_trace()
    
    dataset = {
        split: data.map(
            lambda ex: model.tokenizer(ex["chat_text"]), 
            remove_columns="chat_text",
            batched=True,
        )
        for split, data in dataset.items()
    }

    training_args = exp["train"]["training"]
    training_args["output_dir"] = exp["output_dir"]

    trainer_args = exp["train"].get("trainer", {})
    if peft_config := exp.get("peft"):
        peft_config = getattr(peft, peft_config["config"])(**peft_config["config_args"])
    trainer_args["peft_config"] = peft_config

    trainer = train_clm(
        model=model.model,
        tokenizer=model.tokenizer,
        train_dataset=dataset["train"],
        trainer_class=exp["train"]["class"],
        args_class=exp["train"]["args_class"],
        eval_dataset=dataset.get("validation"),
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
        model_name = config["exp"]["model"]["init_kwargs"]["model"]

    model_name = model_name.replace("/", "-")
    if model_name == "openai":
        model_name += "-" + config["exp"]["model"]["generation_config"]["model"]

    save_dir = save_dir / f"train-{model_name}-{str(datetime.now())}"
    config["exp"]["output_dir"] = save_dir
    return config


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Training HF models as Process or Outcome Rewrad Models."
    )

    parser.add_argument(
        "--config",
        default="config/train-tc-rm-config.yaml",
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
