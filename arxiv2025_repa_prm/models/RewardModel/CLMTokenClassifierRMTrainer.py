import torch
from torch import nn
from datasets import Dataset
from transformers import DataCollatorForTokenClassification
from transformers.trainer_pt_utils import nested_detach
from trl import RewardTrainer

from .token_classifier_rmutils import collate_logits, collate_labels


class CLMTokenClassifierRMTrainer(RewardTrainer):

    def __init__(
        self,
        step_tag="ки", # ensure uncommon but single token upon tokenization
        step_targets=["-", "+"], # ensure values are single token upon tokenization
        targets_label_pad_id=-100,
        **reward_trainer_kwargs,
    ):

        train_dataset = reward_trainer_kwargs.pop("train_dataset", None)
        
        if "tokenizer" in reward_trainer_kwargs:
            reward_trainer_kwargs["processing_class"] = reward_trainer_kwargs.pop("tokenizer")
        
        dummy_dataset = Dataset.from_dict({"input_ids_chosen": [1]})
        super().__init__(train_dataset=dummy_dataset, **reward_trainer_kwargs)

        self.data_collator = DataCollatorForTokenClassification(self.processing_class, padding=True)
        self.train_dataset = train_dataset
        self.compute_metrics = None

        self.step_tag = step_tag
        self.step_tag_id = self.processing_class.encode(step_tag)[-1]

        self.step_targets = step_targets
        self.step_target_ids = self.processing_class.convert_tokens_to_ids(step_targets)

        self.targets_label_pad_id = targets_label_pad_id


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(
            input_ids=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        )
        
        logits = collate_logits(
            logits=outputs["logits"], 
            input_ids=inputs["input_ids"],
            step_tag_id=self.step_tag_id, 
            step_target_ids=self.step_target_ids, 
            padding_side=self.processing_class.padding_side,
        )

        labels = collate_labels(
            labels=inputs["labels"], 
            targets_label_pad_id=self.targets_label_pad_id, 
            padding_side=self.processing_class.padding_side,
        ).to(logits.device)

        # cross_entropy expects input in (N,C,d1,d2,...,dk) format
        logits = logits.transpose(1,2)

        # spurious / extra step token that is part of the text itself should be 
        # rare if the step token is chosen not to be common e.g. "ки" can occur 
        # in Russian text and can cause this issue since it is caught as label 
        # during label creation but not as step token id during tokenization. 
        # Workaround: skip their handling for now assuming such samples will be 
        # extremely rare.
        if labels.size(1) > logits.size(2):
            labels = labels[:, :logits.size(2)]
        elif labels.size(1) < logits.size(2):
            logits = logits[:, :, :labels.size(1)]

        loss = nn.functional.cross_entropy(
            input=logits, target=labels, ignore_index=self.targets_label_pad_id
        )
        
        return (loss, logits) if return_outputs else loss


    def prediction_step(
        self, model, inputs, prediction_loss_only=True, ignore_keys=None
    ):
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, logits = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        loss = loss.detach()
        logits = nested_detach(logits)
        labels = inputs["labels"]

        return loss, logits, labels


    def evaluate(self, *args, **kwargs):
        # Skip Reward Trainer's evaluate method which just does visualization
        # and call the Trainer evaluate method directly
        return super(RewardTrainer, self).evaluate(*args, **kwargs)
