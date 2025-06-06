import re
import json
from pathlib import Path
from functools import partial
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, DatasetDict

from .utils import example_to_content

import sys

root_dir = Path(__file__).parents[1]

sys.path.append(str(root_dir))

from evaluation.sparp.metrics import averaged_em_f1
from utils.answer import extract_ans_per_output
from data.utils import (
    load_model_output,
    merge_outputs_by_ids,
    merge_details_into_model_output,
)
from data.rm_utils import (
    prepare_step_tag_and_scores,
    decompose_and_aggregate_prm_scores,
    decompose_and_aggregate_grading_scores,
)
from prompting.utils import (
    comma_separated_formatter,
    form_query_text,
    form_response_text,
    create_prompt_messages,
)


BOS_TOKEN_LEN = {
    "llama-2": len("<s> "),
    "llama-3": len("<|begin_of_text|>"),
}

PROMPT_POSTPROCESSOR = {
    # specific to llama-2 to remove "<s> " in the beginning (as this is added
    # later by preference trainer) and add space in the end
    "llama-2": lambda x: x[BOS_TOKEN_LEN["llama-2"] :] + " ",
    "llama-3": lambda x: x[BOS_TOKEN_LEN["llama-3"] :],
    "default": lambda x: x,  # no postprocessing
}

COMPARE_FN = {
    "greater_than": lambda x,y: x > y,
    "pareto_comparator": None, # replace by implementation below
}


def outcome_scorer(
    output,
    targets,
    target_choices=[],
    metric="em",
    ans_trigger="Hence, the answer is",
    regex_flags=re.MULTILINE,
    **kwargs, # kwargs just a placeholder
):

    extracted = extract_ans_per_output(
        output if isinstance(output, str) else output["response"],
        ans_choices=target_choices,
        ans_trigger=ans_trigger,
        regex_flags=regex_flags,
    )

    em, f1, _ = averaged_em_f1(
        y_true=targets, y_pred=extracted, as_set=True, create_lb=True
    )

    score = em if metric == "em" else f1
    
    return score


def reasoning_scorer(
    output, 
    use_scores=["answer", "sequence_mean"], 
    mean=False, 
    fill_none_with=None, 
    **kwargs, # kwargs just a placeholder
):
    r"""Dictionary of individual reasoning scores are pre-computed"""

    if isinstance(use_scores, str):
        use_scores = [use_scores]
    if fill_none_with is None:
        fill_none_with = np.nan
    
    score = []
    for score_name in use_scores:
        score_val = output["agg_evals"].get(score_name) # can be 0
        score_val = score_val if score_val is not None else fill_none_with
        score.append(score_val)

    score = tuple(score)
    if len(score) == 1:
        score = score[0]
    elif mean:
        score = np.nanmean(score)

    return score


def sequential_reasoning_scorer(
    output, 
    use_scores=["em", "mean_step"], 
    score_key="step_score", 
    **kwargs, # kwargs just a placeholder
):
    
    if isinstance(use_scores, str):
        use_scores = [use_scores]

    score = []

    for score_name in use_scores:
        score_val = 0
        
        if score_name == "em":
            for i in output[score_key]:
                if i["scored"] == "answer":
                    score_val = int(i["score"] == 1)
                    break
        
        elif score_name == "mean_step":
            score_val = np.mean([i["score"] for i in output[score_key]])
        
        score.append(score_val)
    
    score = tuple(score)
    if len(score) == 1:
        score = score[0]

    return score


def decomposed_reasoning_scorer(
    output, use_scores=["qa_ans_em"], mean=False, fill_none_with=None, **kwargs # kwargs just a placeholder
):
    r"""Dictionary of individual reasoning scores are pre-computed"""

    if isinstance(use_scores, str):
        use_scores = [use_scores]
    if fill_none_with is None:
        fill_none_with = np.nan
    
    score = []
    for score_name in use_scores:
        score_val = output["score"].get(score_name) # can be 0
        if score_val is None:
            score_val = output["aggregated_evals"].get(score_name)
        # even after this score_val is None, which means
        # this evaluation can be marked as 1 for comparison
        score_val = score_val if score_val is not None else fill_none_with
        score.append(score_val)

    score = tuple(score)
    if len(score) == 1:
        score = score[0]
    elif mean:
        score = np.nanmean(score)

    return score


def get_preference_scorer(preference_scorer=None, **kwargs):
    if preference_scorer == "outcome_scorer":
        preference_scorer = partial(outcome_scorer, **kwargs)
    elif preference_scorer == "reasoning_scorer":
        preference_scorer = partial(reasoning_scorer, **kwargs)
    elif preference_scorer == "sequential_reasoning_scorer":
        preference_scorer = partial(sequential_reasoning_scorer, **kwargs)
    elif preference_scorer == "decomposed_reasoning_scorer":
        preference_scorer = partial(decomposed_reasoning_scorer, **kwargs)
    else:
        # no preference
        preference_scorer = lambda x: 1

    return preference_scorer

# deprecated, use comparison_preference_sampler instead
def binarized_preference_sampler(
    samples,
    preference_score,
    pair_count=1,
    # samples only unique pairs and hence can be less that pair_count request 
    # else oversample to match pair_count
    unique_pairs=True,
    threshold=0.5,
    seed=42,
    rng=None,
):
    if not rng:
        rng = np.random.default_rng(seed=seed)

    binarized = [0 if sc <= threshold else 1 for sc in preference_score]
    samples_with_score = list(zip(samples, preference_score))
    choose_samples = [s for i, s in enumerate(samples_with_score) if binarized[i] == 1]
    reject_samples = [s for i, s in enumerate(samples_with_score) if binarized[i] == 0]

    chosen, chosen_score = [], []
    if choose_samples:
        choose_idx = rng.choice(len(choose_samples), size=pair_count)
        chosen = [choose_samples[i][0] for i in choose_idx]
        chosen_score = [choose_samples[i][1] for i in choose_idx]

    rejected, rejected_score = [], []
    if reject_samples:
        reject_idx = rng.choice(len(reject_samples), size=pair_count)
        rejected = [reject_samples[i][0] for i in reject_idx]
        rejected_score = [reject_samples[i][1] for i in reject_idx]

    return chosen, rejected, chosen_score, rejected_score


def pareto_comparator(x, y, strict=False):
    comparison = []
    eq_count = 0

    for i,j in zip(x,y):
        if not np.isnan(i) and not np.isnan(j):
            gte = (i > j if strict else i >= j)
            eq = i == j
            comparison.append(int(gte))
            if eq:
                eq_count += 1

    preferred = True if all(comparison) and eq_count != len(comparison) else False
    return preferred

COMPARE_FN["pareto_comparator"] = pareto_comparator


def comparison_preference_sampler(
    samples, 
    preference_score, 
    pair_count=1, 
    # samples only unique pairs and hence can be less that pair_count request 
    # else oversample to match pair_count
    unique_pairs=True,
    seed=42, 
    compare_fn=COMPARE_FN["greater_than"],
    compare_fn_kwargs={},
    rng=None, 
):
    if not rng:
        rng = np.random.default_rng(seed=seed)

    if isinstance(compare_fn, str):
        compare_fn = COMPARE_FN[compare_fn]

    if compare_fn_kwargs is None:
        compare_fn_kwargs = {}

    sorted_score_idx_pairs = []
    for i in range(len(preference_score)):
        for j in range(i+1, len(preference_score)):
            i_score, j_score = preference_score[i], preference_score[j]

            if compare_fn(i_score, j_score, **compare_fn_kwargs):
                sorted_score_idx_pairs.append((i,j))
            elif compare_fn(j_score, i_score, **compare_fn_kwargs):
                sorted_score_idx_pairs.append((j,i))

    # no prefrence pairs is possible if all entries have same score, hence check
    if sorted_score_idx_pairs:
        if unique_pairs:
            pair_count = min(pair_count, len(sorted_score_idx_pairs))
        sorted_score_idx_pairs = rng.choice(
            sorted_score_idx_pairs, size=pair_count, replace=not unique_pairs
        )
    
    chosen = [samples[idx[0]] for idx in sorted_score_idx_pairs]
    chosen_score = [preference_score[idx[0]] for idx in sorted_score_idx_pairs]
    
    rejected = [samples[idx[1]] for idx in sorted_score_idx_pairs]
    rejected_score = [preference_score[idx[1]] for idx in sorted_score_idx_pairs]

    return chosen, rejected, chosen_score, rejected_score


def get_preference_sampler(preference_sampler=None, seed=42, rng=None, **kwargs):
    if preference_sampler == "comparison_preference_sampler":
        preference_sampler = partial(
            comparison_preference_sampler, seed=seed, rng=rng, **kwargs
        )
    else:
        preference_sampler = partial(
            binarized_preference_sampler, seed=seed, rng=rng, **kwargs
        )

    return preference_sampler


def generate_preference_data(
    model_output,
    preference_scorer="outcome_scorer",
    preference_scorer_kwargs={
        "metric": "em", 
        "ans_trigger": "Hence, the answer is"
    },
    preference_sampler="binarized_preference_sampler",
    preference_sampler_kwargs={"pair_count": 1},
    seed=42,
    rng=None,
):

    if not rng:
        rng = np.random.default_rng(seed=seed)

    preference_scorer = get_preference_scorer(
        preference_scorer, **preference_scorer_kwargs
    )

    preference_sampler = get_preference_sampler(
        preference_sampler, seed, rng, **preference_sampler_kwargs
    )

    for dataset in model_output.values():
        for ex in tqdm(dataset):
            
            # remove duplicates
            if isinstance(ex["model_output"][0], str):
                ex["model_output"] = list(set(ex["model_output"]))
            else:
                unique_output = dict()
                for ex_out in ex["model_output"]:
                    if ex_out["response"] not in unique_output:
                        unique_output[ex_out["response"]] = ex_out
                ex["model_output"] = list(unique_output.values())

            ex["preference_score"] = [
                preference_scorer(
                    out, targets=ex["targets"], target_choices=ex["target_choices"]
                )
                for out in ex["model_output"]
            ]

            (
                ex["chosen"],
                ex["rejected"],
                ex["chosen_score"],
                ex["rejected_score"],
            ) = preference_sampler(
                samples=ex["model_output"], preference_score=ex["preference_score"]
            )

            if ex["chosen"] and not isinstance(ex["chosen"][0], str):
                if ex["chosen"][0].get("score_sequence"):
                    ex["chosen_score_sequence"] = [i["score_sequence"] for i in ex["chosen"]]
                ex["chosen"] = [i["response"] for i in ex["chosen"]]

            if ex["rejected"] and not isinstance(ex["rejected"][0], str):
                if ex["rejected"][0].get("score_sequence"):
                    ex["rejected_score_sequence"] = [i["score_sequence"] for i in ex["rejected"]]
                ex["rejected"] = [i["response"] for i in ex["rejected"]]

    return model_output


def insert_ground_truth_as_preference(
    model_output, 
    also_vs_chosen=False, 
    ex_to_content_kwargs=dict(),
    step_tag_and_scores_kwargs={},
):

    # check tuple score and length
    tuple_score_len = 1
    for dataset in model_output.values():
        for ex in dataset:
            if ex["chosen_score"] and isinstance(ex["chosen_score"][0], (tuple, list)):
                tuple_score_len = len(ex["chosen_score"][0])
                break
            elif ex["rejected_score"] and isinstance(ex["rejected_score"][0], (tuple, list)):
                tuple_score_len = len(ex["rejected_score"][0])
                break
            # otherwise continue to next example
    
    for dataset in model_output.values():
        for ex in tqdm(dataset):
            if ex["chosen"] and ex["rejected"]:
                ground_truth = example_to_content(ex, **ex_to_content_kwargs)["response"]
                if step_tag_and_scores_kwargs:
                    ground_truth, score_sequence = prepare_step_tag_and_scores(
                        response=ground_truth, grading=None, **step_tag_and_scores_kwargs
                    )

                # rejected = list(set(ex["rejected"]))
                # dictionary to deduplicate
                rejected_with_score = {
                    i: (sc, ss) 
                    for i,sc,ss in zip(
                        ex["rejected"], ex["rejected_score"], ex["rejected_score_sequence"]
                    )
                }
                rejected = list(rejected_with_score.keys())
                rejected_score = [i[0] for i in rejected_with_score.values()]
                if step_tag_and_scores_kwargs:
                    rejected_score_sequence = [i[1] for i in rejected_with_score.values()]

                chosen = [ground_truth] * len(rejected)
                chosen_score = [
                    tuple([1]*tuple_score_len) if tuple_score_len > 1 else 1
                ] * len(rejected)
                if step_tag_and_scores_kwargs:
                    chosen_score_sequence = [score_sequence] * len(rejected)

                if also_vs_chosen:
                    
                    if step_tag_and_scores_kwargs:
                        ex_chosen_with_score = {
                            i: (sc, ss) 
                            for i,sc,ss in zip(
                                ex["chosen"], ex["chosen_score"], ex["chosen_score_sequence"]
                            )
                        }
                    else:
                        ex_chosen_with_score = {
                            i: (sc,) for i,sc in zip(ex["chosen"], ex["chosen_score"])
                        }

                    ex_chosen = list(ex_chosen_with_score.keys())
                    ex_chosen_score = [i[0] for i in ex_chosen_with_score.values()]
                    
                    if step_tag_and_scores_kwargs:
                        ex_chosen_score_sequence = [i[1] for i in ex_chosen_with_score.values()]

                    chosen = [ground_truth] * len(ex_chosen) + chosen
                    chosen_score = [
                        tuple([1]*tuple_score_len) if tuple_score_len > 1 else 1
                    ] * len(ex_chosen) + chosen_score
                    if step_tag_and_scores_kwargs:
                        chosen_score_sequence = [score_sequence] * len(ex_chosen) + chosen_score_sequence

                    rejected = ex_chosen + rejected
                    rejected_score = ex_chosen_score + rejected_score
                    if step_tag_and_scores_kwargs:
                        rejected_score_sequence = ex_chosen_score_sequence + rejected_score_sequence

                if ex["chosen"] and ex["rejected"]:
                    chosen += ex["chosen"]
                    chosen_score += ex["chosen_score"]
                    rejected += ex["rejected"]
                    rejected_score += ex["rejected_score"]

                    if step_tag_and_scores_kwargs:
                        chosen_score_sequence += ex["chosen_score_sequence"]
                        rejected_score_sequence += ex["rejected_score_sequence"]

                ex["chosen"] = chosen
                ex["chosen_score"] = chosen_score
                ex["rejected"] = rejected
                ex["rejected_score"] = rejected_score

                if step_tag_and_scores_kwargs:
                    ex["chosen_score_sequence"] = chosen_score_sequence
                    ex["rejected_score_sequence"] = rejected_score_sequence

    return model_output


def sample_preference_per_example(
    model_output,
    sample_count=1,
    seed=42,
    rng=None,
):
    r"""call after preference count normalization i.e. both chosen and rejected
    per example are of same count."""

    if not rng:
        rng = np.random.default_rng(seed=seed)

    for dataset in model_output.values():
        for ex in tqdm(dataset):
            if ex["chosen"] and ex["rejected"]:
                choice_idx = rng.choice(
                    len(ex["chosen"]), size=min(sample_count, len(ex["chosen"])), replace=False,
                )

                ex["chosen"] = [ex["chosen"][i] for i in choice_idx]
                ex["chosen_score"] = [ex["chosen_score"][i] for i in choice_idx]
                ex["rejected"] = [ex["rejected"][i] for i in choice_idx]
                ex["rejected_score"] = [ex["rejected_score"][i] for i in choice_idx]

                try:
                    ex["chosen_score_sequence"] = [ex["chosen_score_sequence"][i] for i in choice_idx]
                    ex["rejected_score_sequence"] = [ex["rejected_score_sequence"][i] for i in choice_idx]
                except IndexError as e:
                    pass

    return model_output


def preference_output_to_hf_dataset(
    model_output,
    prompt_col="question",
    chosen_col="chosen",
    rejected_col="rejected",
    instruction_col="instruction",  # mark None if not present
    context_col="context",  # mark None if not present
    context_id_col="context_id",
    question_id_col="question_id",
    chosen_score_col="chosen_score",
    rejected_score_col="rejected_score",
    chosen_score_sequence_col=None,
    rejected_score_sequence_col=None,
):

    converted = dict()

    all_cols = [
        context_id_col, 
        question_id_col, 
        prompt_col, 
        chosen_col, 
        rejected_col,
        chosen_score_col,
        rejected_score_col,
        chosen_score_sequence_col,
        rejected_score_sequence_col,
    ]

    all_cols = [i for i in all_cols if i] # keep only not None or not empty ones

    if instruction_col:
        all_cols.append(instruction_col)

    if context_col:
        all_cols.append(context_col)

    for name, dataset in model_output.items():
        cur_dict = {i: [] for i in all_cols}

        for ex in tqdm(dataset):
            if ex[chosen_score_col] and ex[rejected_score_col]:
                for i, (chosen, rejected, chosen_score, rejected_score) in enumerate(
                    zip(
                        ex[chosen_col], 
                        ex[rejected_col], 
                        ex[chosen_score_col], 
                        ex[rejected_score_col],
                    )
                ):
                    cur_dict[chosen_col].append(chosen)
                    cur_dict[rejected_col].append(rejected)
                    cur_dict[chosen_score_col].append(chosen_score)
                    cur_dict[rejected_score_col].append(rejected_score)

                    for col in cur_dict.keys():
                        if col not in (
                            chosen_col, rejected_col, chosen_score_col, rejected_score_col
                        ):
                            if col in (chosen_score_sequence_col, rejected_score_sequence_col):
                                cur_dict[col].append(ex[col][i])
                            else:
                                cur_dict[col].append(ex[col])

        converted[name] = Dataset.from_dict(cur_dict)

    return converted


def get_score_creator(score_creator=None, **kwargs):
    if isinstance(score_creator, str):
        if score_creator == "decompose_and_aggregate_prm_scores":
            score_creator = partial(decompose_and_aggregate_prm_scores, **kwargs)
        elif score_creator == "decompose_and_aggregate_grading_scores":
            score_creator = partial(decompose_and_aggregate_grading_scores, **kwargs)
        else:
            raise NotImplementedError

    return score_creator


def load_split_preference_data(
    model_output_path,
    data_name=[],
    merge_by_id_kwargs=dict(), # merge within output if these kwargs are provided
    score_creator=None, # in case score values e.g. metrics needs to be obtained from model output
    score_creator_kwargs=dict(),
    source_data=None,  # contains ground truth and other details
    merge_source_data_kwargs=dict(),
    ex_to_content_kwargs=dict(),
    split="train",
    preference_scorer="outcome_scorer",
    preference_scorer_kwargs={
        "metric": "em", 
        "ans_trigger": "Hence, the answer is"
    },
    preference_sampler="binarized_preference_sampler",
    preference_sampler_kwargs={"pair_count": 1},
    ground_truth_as_preference=True,
    ground_truth_vs_chosen=False,
    step_tag_and_scores_kwargs={},
    pair_count_post_ground_truth=1,
    output_to_hf_dataset_kwargs=dict(),
    seed=42,
    rng=None,
):

    if Path(model_output_path).is_file():
        with open(model_output_path, "r") as f:
            name = data_name if data_name else "default"
            model_output = {name: json.load(f)}

    else:
        model_output = load_model_output(
            input_dir=model_output_path, data_name=data_name
        )

    if merge_by_id_kwargs:
        model_output = merge_outputs_by_ids(model_output, **merge_by_id_kwargs)

    if score_creator := get_score_creator(score_creator, **score_creator_kwargs):
        for out in model_output.values():
            for ex in tqdm(out):
                ex["model_output"] = [score_creator(i) for i in ex["model_output"]]

    if not rng:
        rng = np.random.default_rng(seed=seed)

    # ################
    # for name, data in model_output.items():
    #     with open(f"output/sparp/grading/scored-llama-3-8B-1I-on-train-20-gen-grading/{name}.json", "w+") as f:
    #         json.dump(data, f, indent=2)
    # import pdb; pdb.set_trace()
    # ################

    model_output = generate_preference_data(
        model_output,
        preference_scorer=preference_scorer,
        preference_scorer_kwargs=preference_scorer_kwargs,
        preference_sampler=preference_sampler,
        preference_sampler_kwargs=preference_sampler_kwargs,
        rng=rng,
    )

    # to copy ground truth details
    if source_data:
        model_output = merge_details_into_model_output(
            source_data, model_output, model_output_split=split, **merge_source_data_kwargs
        )

        if ground_truth_as_preference:
            model_output = insert_ground_truth_as_preference(
                model_output, 
                also_vs_chosen=ground_truth_vs_chosen, 
                ex_to_content_kwargs=ex_to_content_kwargs,
                step_tag_and_scores_kwargs=step_tag_and_scores_kwargs,
            )

            model_output = sample_preference_per_example(
                model_output, sample_count=pair_count_post_ground_truth, rng=rng
            )

    preference_dataset = preference_output_to_hf_dataset(
        model_output, **output_to_hf_dataset_kwargs
    )

    return preference_dataset


def load_preference_data(
    train_output_path,
    validation_output_path=None,
    data_name=[],
    merge_by_id_kwargs=dict(), # merge within output if these kwargs are provided
    score_creator=None, # in case score values e.g. metrics needs to be obtained from model output
    score_creator_kwargs=dict(),
    source_data=None, # contains ground truth and other details
    merge_source_data_kwargs=dict(), 
    ex_to_content_kwargs=dict(),
    preference_scorer="outcome_scorer",
    preference_scorer_kwargs={
        "metric": "em", 
        "ans_trigger": "Hence, the answer is"
    },
    preference_sampler="binarized_preference_sampler",
    sampler_common_kwargs={"unique_pairs": True},
    train_preference_sampler_kwargs={"pair_count": 1},
    validation_preference_sampler_kwargs={"pair_count": 1},
    ground_truth_as_preference=True,
    ground_truth_vs_chosen=False,
    step_tag_and_scores_kwargs={},
    train_pair_count_post_ground_truth=1,
    validation_pair_count_post_ground_truth=1,
    output_to_hf_dataset_kwargs=dict(),
    seed=42,
    rng=None,
):

    if not rng:
        rng = np.random.default_rng(seed=seed)

    preference_data = {
        i: {"train": j}
        for i, j in load_split_preference_data(
            model_output_path=train_output_path,
            data_name=data_name,
            merge_by_id_kwargs=merge_by_id_kwargs, 
            score_creator=score_creator, 
            score_creator_kwargs=score_creator_kwargs,
            source_data=source_data,
            merge_source_data_kwargs=merge_source_data_kwargs,
            ex_to_content_kwargs=ex_to_content_kwargs,
            split="train",
            preference_scorer=preference_scorer,
            preference_scorer_kwargs=preference_scorer_kwargs,
            preference_sampler=preference_sampler,
            preference_sampler_kwargs={
                **sampler_common_kwargs, **train_preference_sampler_kwargs
            },
            ground_truth_as_preference=ground_truth_as_preference,
            ground_truth_vs_chosen=ground_truth_vs_chosen,
            step_tag_and_scores_kwargs=step_tag_and_scores_kwargs,
            pair_count_post_ground_truth=train_pair_count_post_ground_truth,
            output_to_hf_dataset_kwargs=output_to_hf_dataset_kwargs,
            seed=seed,
            rng=rng,
        ).items()
    }

    if validation_output_path:
        for i, j in load_split_preference_data(
            model_output_path=validation_output_path,
            data_name=data_name,
            merge_by_id_kwargs=merge_by_id_kwargs, 
            score_creator=score_creator, 
            score_creator_kwargs=score_creator_kwargs,
            source_data=source_data,
            merge_source_data_kwargs=merge_source_data_kwargs,
            ex_to_content_kwargs=ex_to_content_kwargs,
            split="validation",
            preference_scorer=preference_scorer,
            preference_scorer_kwargs=preference_scorer_kwargs,
            preference_sampler=preference_sampler,
            preference_sampler_kwargs={
                **sampler_common_kwargs, **validation_preference_sampler_kwargs
            },
            ground_truth_as_preference=ground_truth_as_preference,
            ground_truth_vs_chosen=ground_truth_vs_chosen,
            step_tag_and_scores_kwargs=step_tag_and_scores_kwargs,
            pair_count_post_ground_truth=validation_pair_count_post_ground_truth,
            output_to_hf_dataset_kwargs=output_to_hf_dataset_kwargs,
            seed=seed,
            rng=rng,
        ).items():

            preference_data[i]["validation"] = j

    return preference_data


def create_chat_templated_preference_prompt(
    preference_data,
    chat_template_func,
    prepare_splits=["train", "validation"],
    prompt_col="question",
    rename_prompt_col="prompt",  # None if final col same as prompt_col
    instruction_col="instruction",  # mark None if not present
    context_col="context",  # mark None if not present
    ctx_ques_sep=" ",
    response_cols=["chosen", "rejected"],
    implicit_prompt=False, # set True e.g. for preference based RewardModels
    prompt_postprocessor="default",  # or model/tokenizer specific function
):
    r"""adapted from create_chat_templated_text_data of data.utils"""

    prompt_postprocessor = PROMPT_POSTPROCESSOR.get(
        prompt_postprocessor, PROMPT_POSTPROCESSOR["default"]
    )

    def ex_to_query_content(ex):
        content = {
            "query": form_query_text(
                # space as separator between context and question
                context=ex[context_col] + ctx_ques_sep if context_col else "",
                question=ex[prompt_col],
            ),
        }
        return content

    def create_prompt_text(example):
        msgs = create_prompt_messages(
            contents=[ex_to_query_content(example)],
            system_content=example[instruction_col] if instruction_col else "",
        )
        prompt_text = chat_template_func(msgs, tokenize=False)
        prompt_text = prompt_postprocessor(prompt_text)
        return {prompt_col: prompt_text}

    def ex_to_response_content(ex, resp_col="chosen"):
        return {"role": "assistant", "content": ex[resp_col]}

    def create_response_text(example, resp_col="chosen"):
        response_text = chat_template_func(
            [ex_to_response_content(example, resp_col=resp_col)], tokenize=False
        )
        response_text = prompt_postprocessor(response_text)
        return {resp_col: response_text}

    def ex_to_content(ex, resp_col="chosen"):
        content = {
            "query": form_query_text(
                # space as separator between context and question
                context=ex[context_col] + ctx_ques_sep if context_col else "",
                question=ex[prompt_col],
            ),
            "response": ex[resp_col]
        }
        return content

    def create_implicit_response_text(example, resp_col="chosen"):
        msgs = create_prompt_messages(
            contents=[ex_to_content(example, resp_col=resp_col)],
            system_content=example[instruction_col] if instruction_col else "",
        )
        response_text = chat_template_func(msgs, tokenize=False)
        # # postprocessing not needed as implicit response is used in 
        # # RewardModelling which applies tokenization as is.
        # response_text = prompt_postprocessor(response_text)
        return {resp_col: response_text}

    if implicit_prompt:
        map_fn = {k: partial(create_implicit_response_text, resp_col=k) for k in response_cols}
    else:
        map_fn = {k: partial(create_response_text, resp_col=k) for k in response_cols}
        map_fn[rename_prompt_col if rename_prompt_col else prompt_col] = create_prompt_text

    preference_datasets = {}
    for name, data in preference_data.items():
        preference_datasets[name] = DatasetDict()

        for split in prepare_splits:
            split_data = data[split]
            
            for col, fmap in map_fn.items():
                split_data = split_data.map(fmap)
                
                if col == rename_prompt_col and prompt_col != rename_prompt_col:
                    split_data = split_data.rename_column(prompt_col, rename_prompt_col)

            preference_datasets[name][split] = split_data

    return preference_datasets


def add_margin(
    ex, 
    margin_col="margin",
    chosen_score_col="chosen_score", 
    rejected_score_col="rejected_score"
):
    chosen_score = ex[chosen_score_col]
    if isinstance(chosen_score, (list, tuple)):
        chosen_score = np.sum(chosen_score)

    rejected_score = ex[rejected_score_col]
    if isinstance(rejected_score, (list, tuple)):
        rejected_score = np.sum(rejected_score)

    return {margin_col: chosen_score - rejected_score}


def process_before_train(hf_data, fn_kwargs_list):
    for fn_kwargs in fn_kwargs_list:
        if fn_kwargs["name"] == "add_margin":
            hf_data = hf_data.map(partial(add_margin, **fn_kwargs["kwargs"]))

    return hf_data
