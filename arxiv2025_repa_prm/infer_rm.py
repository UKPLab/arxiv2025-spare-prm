import yaml
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
from functools import partial
from datetime import datetime

import pandas as pd

import torch
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader, RandomSampler
from transformers.trainer_utils import seed_worker

from utils.log import init_neptune, get_gpu_info, save_and_upload_output
from data.utils import (
    load_data,
    load_model_output, ##
    flatten_model_output, ##
    shuffle_select_data,
    field_stratified_select_data, ##
    create_chat_templated_text_data, ##
    merge_details_into_model_output, ##
    # example_to_content, # replaced by custom code below
    filter_model_out_idcs_from_data,
    field_filter,
)

from data.rm_utils import prepare_text_with_step_tag

from evaluation.evaluate import batch_compute_metrics
import models as GenerativeLM
from utils.answer import extract_ans_from_output_files
from models.RewardModel.token_classifier_rmutils import collate_logits


SEED = 42


def main(exp, run=None):

    datasets = load_data(**exp["data"]["load"])

    if merge_args := exp["data"].get("merge_output"):

        if extract_args := merge_args.get("extract_args"):
            model_output = extract_ans_from_output_files(
                input_dir=merge_args["model_output"],
                seed=exp.get("seed", SEED),
                **extract_args
            )

        else:
            model_output = load_model_output(merge_args["model_output"])

        datasets = merge_details_into_model_output(
            data=datasets, 
            model_output={k:v for k,v in model_output.items() if k in datasets}, 
            model_output_split=exp.get("infer_split", "test"),
            merge_keys=merge_args["merge_keys"],
            id_keys=merge_args["id_keys"],
        )

        if (filter_kwargs := exp["data"].get("filter")) and not filter_kwargs.get("model_output"):
            filter_fn = partial(field_filter, **filter_kwargs)
            datasets = {
                name: [ex for ex in named_data if filter_fn(ex)]
                for name, named_data in datasets.items()
            }

        datasets = batch_compute_metrics(
            datasets, **merge_args.get("compute_metrics_kwargs", {})
        )

    # explicit generator function as Dataset.from_generator doesn't work with native generator
    # and Dataset.from_list exhausts memory
    def ex_gen(d):
        for i in d:
            yield i
    
    datasets = {
        name: Dataset.from_generator(partial(ex_gen, data))
        for name, data in datasets.items()
    }

    if select_info := exp["data"].get("select"):
        for name in datasets.keys():
            if select_info["select_fn"] == "shuffle_select":
                datasets[name], selection = shuffle_select_data(
                    datasets[name],
                    seed=exp.get("seed", SEED),
                    shuffle=exp["data"].get("shuffle"),
                    select=select_info.get("select_kwargs"),  # None for whole dataset
                )

            elif select_info["select_fn"] == "field_stratified_select":
                datasets[name], selection = field_stratified_select_data(
                    datasets[name],
                    seed=exp.get("seed", SEED),
                    shuffle=exp["data"].get("shuffle"),
                    **select_info.get("select_kwargs", {})
                )

    datasets = flatten_model_output(datasets)
    datasets = {
        name: {exp.get("infer_split", "test"): Dataset.from_generator(partial(ex_gen, data))}
        for name, data in datasets.items()
    }

    if step_tag_args := exp["data"].get("step_tag_args"):
        def process_step_tag(ex, **kwargs):
            response_field = kwargs.pop("response_field", "response")
            ex["step_tagged_response"] = prepare_text_with_step_tag(
                ex[response_field], **kwargs
            )
            return ex

        datasets = {
            name: {
                split: split_data.map(partial(process_step_tag, **step_tag_args)) 
                for split, split_data in data.items()
            }
            for name, data in datasets.items()
        }

    if (filter_kwargs := exp["data"].get("filter")) and filter_kwargs.get("model_output"):
        datasets = filter_model_out_idcs_from_data(datasets, **filter_kwargs)

    print(f"\nGPU info before model load:\n{get_gpu_info()}\n")

    model = getattr(GenerativeLM, exp["model"]["model_class"])(
        **exp["model"]["init_args"]
    )

    print(f"\nGPU info after model load:\n{get_gpu_info()}\n")

    datasets = create_chat_templated_text_data(
        datasets,
        chat_template_func=model.tokenizer.apply_chat_template,
        **exp["data"]["chat_template_args"],
    )

    infer_split = exp.get("infer_split", "test")

    infer_args = exp.get("infer_args", {})
    if step_tag := infer_args.get("step_tag"):
        infer_args["step_tag_id"] = model.tokenizer.encode(step_tag)[1]
        infer_args["step_target_ids"] = model.tokenizer.convert_tokens_to_ids(infer_args["step_targets"])
        step_proba_idx = infer_args["step_targets"].index(infer_args["step_proba_tag"])

    for name, dataset in tqdm(datasets.items()):

        output = []

        dataset = dataset[infer_split]
        batch_size = exp.get("batch_size", 1)

        for i, batch_start_idx in enumerate(tqdm(range(0, len(dataset), batch_size))):
            # if i == 5:
            #     break

            batch_end_idx = min(batch_start_idx + batch_size, len(dataset))
            batch = dataset[batch_start_idx:batch_end_idx]
            inputs = model.tokenizer(
                batch.pop("chat_text"), padding=True, return_tensors="pt"
            )
            
            # import pdb; pdb.set_trace()
            for i in ("combined_paragraphs", "instruction"):
                batch.pop(i, None)

            with torch.no_grad():
                outputs = model.model(
                    input_ids=inputs["input_ids"].to(model.model.device), 
                    attention_mask=inputs["attention_mask"].to(model.model.device),
                    return_dict=True,
                )

            if i == 0:
                print(f"\nGPU info with 1st batch of {name}:\n{get_gpu_info()}\n")

            if step_tag:
                logits, tag_pos = collate_logits(
                    logits=outputs["logits"], 
                    input_ids=inputs["input_ids"],
                    step_tag_id=infer_args["step_tag_id"], 
                    step_target_ids=infer_args["step_target_ids"], 
                    padding_side=model.tokenizer.padding_side,
                )

                scores = logits.softmax(dim=-1)[:,:,step_proba_idx]

                batch["scores"] = [
                    [s.item() for s,p in zip(sc, tp) if p != 0] # tag_pos = 0 was a fill_value
                    for sc, tp in zip(scores, tag_pos)
                ]

            else:
                batch["scores"] = [[s.item()] for s in outputs["logits"].sigmoid()]

            batch = pd.DataFrame(batch).to_dict(orient="records")
            output.extend(batch)

        output_file = exp["output_dir"] / f"{name}.json"
        save_and_upload_output(output, run=run, save_file=output_file)


def process_config_for_log(config):
    save_dir = Path(config["exp"]["output_dir"])

    model_name = config["exp"]["model"]["model_class"]
    if model_name == "OpenAILM":
        model_name = "openai"
    else:
        model_name = config["exp"]["model"]["init_args"]["model"]

    model_name = model_name.replace("/", "-")
    if model_name == "openai":
        model_name += (
            "-" + config["exp"]["model"]["generate_args"]["generation_config"]["model"]
        )

    save_dir = save_dir / f"infer-{model_name}-{str(datetime.now())}"
    config["exp"]["output_dir"] = save_dir
    return config


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Arguments for inferencing LLM-based textual spatial reasoning."
    )

    parser.add_argument(
        "--config",
        default="config/infer-rm-config.yaml",
        help="path to config file specifying parameters of experiments.",
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

    if config["exp"]["model"]["model_class"] == "OpenAILM":
        config["exp"]["model"]["init_args"]["config"].update(private.get("openai", {}))

    config = process_config_for_log(config)

    main(exp=config["exp"], run=run)
