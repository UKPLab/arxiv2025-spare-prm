import re
import json
from pathlib import Path
from collections import Counter
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

import sys

root_dir = Path(__file__).parents[1]

sys.path.append(str(root_dir))
from data.utils import load_model_output

from . import (
    numeric_output, 
    text_output,
    qa_output, 
    sparp_output,
)


ALL_EXTRACTORS = [numeric_output, text_output, qa_output, sparp_output]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def extract_ans_per_output(
    output,
    ans_choices=[],
    ans_aliases=[],
    sort_choices=False,
    sort_aliases=False,
    ans_trigger="Hence, the answer is",
    trigger_as_regex=False,
    take_trigger_pos=-1, # 0 for first, -1 for last
    extractor_fn="extract_sparp_relations",
    extractor_kwargs={},
    regex_flags=re.MULTILINE,
    ignore_case=True,
):
    if sort_choices:
        # sorting by length is required to match the largest strings first and
        # remove them progressively to avoid multiple counting of small strings
        ans_choices = sorted(ans_choices, key=lambda x: len(x), reverse=True)

    if sort_aliases:
        # sorting by length is required to match the largest strings first and 
        # remove them progressively to avoid multiple counting of small strings 
        # if required
        ans_aliases = sorted(ans_aliases, key=lambda x: len(x), reverse=True)

    if ans_trigger:
        if isinstance(ans_trigger, str):
            ans_trigger = [ans_trigger]

        # anything between trigger till the end.
        found = ""
        for trigger in ans_trigger:
            
            if trigger_as_regex:
                pattern = re.compile(trigger, flags=regex_flags)
            else:
                pattern = re.compile(f"(?<={trigger}).+$", flags=regex_flags)
            
            found = pattern.findall(output)
            if found:
                break

        output = found[take_trigger_pos] if found else ""

    for e in ALL_EXTRACTORS:
        if extractor_fn and (extractor := getattr(e, extractor_fn, None)):
            
            if ignore_case:
                output = output.lower()
            
            output = extractor(
                output, 
                choices=ans_choices, 
                aliases=ans_aliases, 
                **extractor_kwargs
            )

            break

    return output


def count_extractions(extracted):

    empty_placeholder = ""
    hash_inv_fn = None
    
    if extracted:
        if not isinstance(extracted[0], (str, list)) and (np.isnan(extracted[0]) or isinstance(extracted[0], (float, int))):
            empty_placeholder = np.nan
        
        elif isinstance(extracted[0], list):
            # adapt its entries to form a dict key i.e. hashable
            empty_placeholder = "" # since ", ".join([]) is just ""
            extracted = [", ".join(i) for i in extracted]
            hash_inv_fn = lambda x: i if (i := x.split(", "))[0] else []

    extracted = Counter(extracted)

    if empty_placeholder in extracted:
        # convert its count to negative as a special case
        # useful in sorting and avoiding empty values
        extracted[empty_placeholder] *= -1

    return extracted, hash_inv_fn


def set_majority_voting(
    extracted,
    hash_inv_fn=None, # inverse of hashing in count_extractions
    rng=np.random.default_rng(seed=42),
):

    # identify max counts as max votes
    max_count = max(extracted.values())

    # identify the subset with highest counts
    candidates = [k for k, v in extracted.items() if v == max_count]

    # break the tie randomly
    predicted = rng.choice(candidates)

    if hash_inv_fn:
        predicted = hash_inv_fn(predicted)

    return predicted


def extract_ans(
    response,
    ans_choices=[],
    ans_aliases=[],
    consolidate_strategy="set_majority_voting",
    sort_choices=False,
    sort_aliases=False,
    ans_trigger="Hence, the answer is",
    trigger_as_regex=False,
    take_trigger_pos=-1, # 0 for first, -1 for last
    extractor_fn="extract_sparp_relations",
    extractor_kwargs={},
    ignore_case=True,
    rng=np.random.default_rng(seed=42),
):

    if sort_choices:
        # sorting by length is required to match the largest strings first and
        # remove them progressively to avoid multiple counting of small strings
        ans_choices = sorted(ans_choices, key=lambda x: len(x), reverse=True)

    if sort_aliases:
        # sorting by length is required to match the largest strings first and 
        # remove them progressively to avoid multiple counting of small strings 
        # if required
        ans_aliases = sorted(ans_aliases, key=lambda x: len(x), reverse=True)

    if isinstance(ans_trigger, str):
        ans_trigger = [ans_trigger]

    extracted = [
        extract_ans_per_output(
            r, 
            ans_choices=ans_choices, 
            ans_aliases=ans_aliases, 
            sort_choices=False, 
            sort_aliases=False, 
            ans_trigger=ans_trigger,
            trigger_as_regex=trigger_as_regex,
            take_trigger_pos=take_trigger_pos, 
            extractor_fn=extractor_fn,
            extractor_kwargs=extractor_kwargs,
            ignore_case=ignore_case,
        ) 
        for r in response
    ]

    predicted = extracted
    if consolidate_strategy == "set_majority_voting":
        extracted, hash_inv_fn = count_extractions(extracted)
        predicted = set_majority_voting(extracted, hash_inv_fn=hash_inv_fn, rng=rng)

    return extracted, predicted


def extract_ans_from_output_files(
    input_dir,
    seed=42,
    ans_trigger="Hence, the answer is",
    trigger_as_regex=False,
    self_consistency_sample_count=None,
    consolidate_strategy="set_majority_voting",
    target_choices_field="target_choices",
    target_aliases_field="target_aliases",
    take_trigger_pos=-1, # 0 for first, -1 for last
    extractor_fn="extract_sparp_relations",
    extractor_kwargs={},
    ignore_case=True,
):
    output = load_model_output(input_dir)

    rng = np.random.default_rng(seed=seed)
    if isinstance(ans_trigger, str):
        ans_trigger = [ans_trigger]

    for _, model_out in tqdm(output.items()):

        # sorting by length is required to match the largest strings first and
        # remove them progressively to avoid multiple counting of small strings
        target_choices = sorted(
            model_out[0].get(target_choices_field, []), key=lambda x: len(x), reverse=True
        )

        for ex in tqdm(model_out):

            response = ex["model_output"]
            
            # sorting by length is required to match the largest strings first 
            # and remove them progressively to avoid multiple counting of small 
            # strings if required
            target_aliases = sorted(
                ex.get(target_aliases_field, []), key=lambda x: len(x), reverse=True
            )

            if self_consistency_sample_count and self_consistency_sample_count < len(
                response
            ):

                response = rng.choice(
                    response, size=self_consistency_sample_count, replace=False
                )

            ex["extracted"], ex["predicted"] = extract_ans(
                response=response,
                ans_choices=target_choices,
                ans_aliases=target_aliases,
                consolidate_strategy=consolidate_strategy,
                sort_choices=False,
                sort_aliases=False,
                ans_trigger=ans_trigger,
                trigger_as_regex=trigger_as_regex,
                take_trigger_pos=take_trigger_pos, 
                extractor_fn=extractor_fn,
                extractor_kwargs=extractor_kwargs,
                ignore_case=ignore_case,
                rng=rng,
            )

    return output