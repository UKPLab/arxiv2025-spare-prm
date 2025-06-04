import re
from copy import deepcopy
from functools import partial

import numpy as np
import json_repair
from datasets import concatenate_datasets


def regex_substitute(
    text, 
    repl, 
    pattern="$", # "$" for only at the end of text i.e. outcome step
    omit_at_start=True,
    insert_at_end=True,
):
    text = re.sub(pattern, repl, text)

    # find effective repl length since double backslashes are replaced 
    # with single backslashes during sub
    repl_len = len(repl) - len(re.findall("\\\\", repl))
    
    if text and omit_at_start and re.fullmatch(repl, text[:repl_len]):
        text = text[repl_len:]
    
    if text and insert_at_end and text[-repl_len:] != repl:
        text += repl
    
    return text


INSERTER_FN = {
    "regex_substitute": regex_substitute
}


def prepare_text_with_step_tag(
    text, 
    step_tag=" ки\n", # ensure uncommon but single token upon tokenization
    step_pattern="$", # "$" for only at the end of text i.e. outcome step
    step_tag_inserter=None, # callable or keys supported in STEP_TAG_INSERTER
    step_tag_inserter_kwargs=dict(),
    preprocessor=None,
    preprocessor_kwargs=dict(),
):

    preprocessor = INSERTER_FN.get(preprocessor, preprocessor)
    if callable(preprocessor):
        text = preprocessor(text, **preprocessor_kwargs)
    
    if step_tag != step_pattern:
        if was_str := isinstance(text, str):
            text = [text]
        
        step_tag_inserter = INSERTER_FN.get(step_tag_inserter, step_tag_inserter)
        if callable(step_tag_inserter):
            text = [
                step_tag_inserter(
                    t, 
                    repl=step_tag, 
                    pattern=step_pattern, 
                    **step_tag_inserter_kwargs
                )
                for t in text
            ]

        if was_str:
            text = text[0]

    return text


def prepare_step_tag_and_scores(
    response,
    grading,
    empty_grading_as_gt=True,
    outcome_score=None,
    step_tag=" ки\n", # ensure uncommon but single token upon tokenization
    step_pattern="$", # "$" for only at the end of text i.e. outcome step
    reference_ans="",
    preprocessor=None,
    preprocessor_kwargs=dict(),
    step_tag_inserter=None, # callable or keys supported in STEP_TAG_INSERTER
    step_tag_inserter_kwargs=dict(),
    step_targets=dict(INCORRECT=0, CORRECT=1),
    missed_grading_as_outcome=False, # False is strict i.e. treat any missed grading as incorrect, else treat same as outcome
):

    preprocessor = INSERTER_FN.get(preprocessor, preprocessor)
    if callable(preprocessor):
        response = preprocessor(response, **preprocessor_kwargs)

    response = prepare_text_with_step_tag(
        response, 
        step_tag=step_tag, 
        step_pattern=step_pattern, 
        step_tag_inserter=step_tag_inserter,
        step_tag_inserter_kwargs=step_tag_inserter_kwargs,
    )
    # import pdb; pdb.set_trace()

    num_student_steps = len(re.findall(step_tag, response))

    if not grading and empty_grading_as_gt:
        grading = {i: step_targets["CORRECT"] for i in range(num_student_steps)}

    else:
        grading = json_repair.loads(grading)

        grading = {
            s["student_step"]-1: step_targets.get(
                s.get("label"), step_targets["INCORRECT"]
            )
            for s in grading if isinstance(s, dict) and 
            isinstance(s.get("student_step"), int) and
            isinstance(s.get("label"), str)
        }

    if outcome_score is not None:
        if isinstance(outcome_score, (int, float)):
            if outcome_score == 1:
                outcome_score = step_targets["CORRECT"]
            else:
                outcome_score = step_targets["INCORRECT"]

    if missed_grading_as_outcome and outcome_score is not None:
        missed_grade = outcome_score
    else:
        missed_grade = step_targets["INCORRECT"]

    for s in range(num_student_steps):
        if s not in grading:
            grading[s] = missed_grade

    if outcome_score is not None:
        grading[num_student_steps-1] = outcome_score

    scores = [grading[i] for i in range(num_student_steps)]

    # # only outcome tagging
    # if step_pattern == "$":
    #     try:
    #         assert len(scores) == 1, "there should be only 1 score for outcome based step pattern"
    #     except AssertionError:
    #         import pdb; pdb.set_trace()
    #     scores = scores[0]

    return response, scores


def create_step_labels(
    ex,
    response_field="grade_ans",
    label_field="grading",
    outcome_field="em",
    reference_field="",
    step_tag=" ки\n", # ensure uncommon but single token upon tokenization
    step_pattern="$", # "$" for only at the end of text i.e. outcome step
    preprocessor=None,
    preprocessor_kwargs=dict(),
    step_tag_inserter=None, # callable or keys supported in STEP_TAG_INSERTER
    step_tag_inserter_kwargs=dict(),
    step_targets=dict(INCORRECT=0, CORRECT=1),
    missed_grading_as_outcome=False, # False is strict i.e. treat any missed grading as incorrect, else treat same as outcome
):

    resp, scores = prepare_step_tag_and_scores(
        response=ex[response_field],
        grading=ex.get(label_field),
        outcome_score=ex[outcome_field],
        reference_ans=ex.get(reference_field, ""),
        step_tag=step_tag, 
        step_pattern=step_pattern,
        preprocessor=preprocessor,
        preprocessor_kwargs=preprocessor_kwargs,
        step_tag_inserter=step_tag_inserter, 
        step_tag_inserter_kwargs=step_tag_inserter_kwargs,
        step_targets=step_targets,
        missed_grading_as_outcome=missed_grading_as_outcome,
    )

    ex["step_tagged_response"] = resp
    ex["labels"] = scores
    return ex


def preference_to_instance_dataset(
    dataset,
    chosen_col="chosen",
    rejected_col="rejected",
    chosen_score_sequence_col="chosen_score_sequence",
    rejected_score_sequence_col="rejected_score_sequence",
    shuffle=True,
    seed=42,
):
    
    for split in dataset.keys():
        
        base_columns = [
            i 
            for i in dataset[split].column_names 
            if i not in (
                chosen_col, 
                rejected_col, 
                chosen_score_sequence_col, 
                rejected_score_sequence_col
            )
        ]

        chosen_dataset = dataset[split].select_columns(
            base_columns+[chosen_col, chosen_score_sequence_col]
        )
        chosen_dataset = chosen_dataset.rename_columns(
            {chosen_col: "step_tagged_response", chosen_score_sequence_col: "labels"}
        )

        rejected_dataset = dataset[split].select_columns(
            base_columns+[rejected_col, rejected_score_sequence_col]
        )
        rejected_dataset = rejected_dataset.rename_columns(
            {rejected_col: "step_tagged_response", rejected_score_sequence_col: "labels"}
        )

        instance_dataset = concatenate_datasets((chosen_dataset, rejected_dataset))
        if shuffle:
            instance_dataset = instance_dataset.shuffle(seed=seed)

        dataset[split] = instance_dataset

    return dataset


def ground_truth_as_graded(
    graded_output,
    id_keys=["context_id", "question_id"],
    reference_field="reference_ans",
    grade_field="grade_ans",
    label_field="grading",
    outcome_field="em",
):
    
    for name in graded_output.keys():
        gt_output = dict()
        for ex in graded_output[name]:
            ex_id = tuple(ex[k] for k in id_keys) if isinstance(id_keys, list) else ex[id_keys]
            if ex_id not in gt_output:
                ex = deepcopy(ex)
                
                ex[grade_field] = ex[reference_field]
                ex[label_field] = "" # empty placeholder, filled later while creating step_labels
                ex[outcome_field] = 1
                gt_output[ex_id] = ex

        graded_output[name].extend(gt_output.values())

    return graded_output


def decompose_and_aggregate_prm_scores(
    ex_out,
    step_tagged_field="step_tagged_response",
    score_field="scores",
    outcome_field="em",
    step_tag=" ки\n",
):
    cot_trigger = 0
    if step_tagged_field := ex_out.pop(step_tagged_field, None):
        steps = re.split(step_tag, step_tagged_field)
    
        ex_out["step_score"] = [
            dict(text=s, score=sc, scored="") 
            for s, sc in zip(steps, ex_out[score_field])
        ]
        
        ex_out["step_score"][0]["scored"] = "cot_trigger"
        ex_out["step_score"][-1]["scored"] = "answer"

        cot_trigger = ex_out["step_score"][0]["score"]

    ex_out["agg_evals"] = dict(
        cot_trigger=cot_trigger,
        answer=ex_out[outcome_field],
        sequence_mean=np.mean(ex_out.pop(score_field)),
    )

    return ex_out


def decompose_and_aggregate_grading_scores(
    ex_out,
    grade_ans_field="grade_ans",
    graded_field="grading",
    outcome_field="em",
    step_pattern="\\n*\\[\\d+\\]\\s*",
    step_joiner="\n",
    insert_joiner_at_end=False,
    step_targets=dict(INCORRECT=-1, CORRECT=1),
    missed_grading_as_outcome=False, # False is strict i.e. treat any missed grading as incorrect, else treat same as outcome
):
    steps = re.split(step_pattern, ex_out.pop(grade_ans_field))
    if steps and not steps[0]:
        steps = steps[1:]
    ex_out["response"] = step_joiner.join(steps)
    if insert_joiner_at_end:
        ex_out["response"] += step_joiner

    grading = json_repair.loads(ex_out.pop(graded_field))

    grading = {
        s["student_step"]-1: (
            step_targets.get(label, step_targets["INCORRECT"])
            if isinstance(label := s.get("label"), str) 
            else step_targets["INCORRECT"]
        )
        for s in grading if isinstance(s, dict) and 
        isinstance(s.get("student_step"), int)
    }

    if ex_out[outcome_field] == 1:
        outcome_score = step_targets["CORRECT"]
    else:
        outcome_score = step_targets["INCORRECT"]

    if missed_grading_as_outcome:
        missed_grade = outcome_score
    else:
        missed_grade = step_targets["INCORRECT"]

    for s in range(len(steps)):
        if s not in grading:
            grading[s] = missed_grade

    grading[len(steps)-1] = outcome_score

    scores = [grading[i] for i in range(len(steps))]

    ex_out["step_score"] = [
        dict(text=s, score=sc, scored="") 
        for s, sc in zip(steps, scores)
    ]
    
    ex_out["step_score"][0]["scored"] = "cot_trigger"
    ex_out["step_score"][-1]["scored"] = "answer"
    ex_out["score_sequence"] = scores

    ex_out["agg_evals"] = dict(
        cot_trigger=ex_out["step_score"][0]["score"],
        answer=ex_out[outcome_field],
        sequence_mean=np.mean(scores),
    )

    return ex_out