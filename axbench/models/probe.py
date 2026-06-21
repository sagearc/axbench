from .model import Model
import torch, transformers, datasets
from tqdm.auto import tqdm
from pyvene import (
    IntervenableConfig,
    IntervenableModel
)
from dataclasses import dataclass
from typing import Dict, Sequence
from torch.utils.data import DataLoader
from .interventions import (
    AdditionIntervention,
    SubspaceIntervention,
    ProbeIntervention,
    SparseProbeIntervention
)
from ..utils.model_utils import (
    gather_residual_activations, 
)
from ..utils.probe_training import train_probe_direction, unit_norm
from ..utils.realizable_basis import resolve_or_build_projector

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


@dataclass
class DataCollator(object):
    """Collate examples for ReFT."""
    
    tokenizer: transformers.AutoTokenizer
    data_collator: transformers.DataCollator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        max_intervention_len = max([len(inst["intervention_locations"][0]) for inst in instances])
        max_seq_len = max([len(inst["input_ids"]) for inst in instances])
        
        for inst in instances:
            non_pad_len = len(inst["input_ids"])

            _intervention_mask = torch.ones_like(inst["intervention_locations"][0])
            _intervention_location_paddings = torch.tensor(
                [[len(inst["input_ids"]) for _ in range(max_intervention_len - len(inst["intervention_locations"][0]))]])
            _intervention_mask_paddings = torch.tensor(
                [0 for _ in range(max_intervention_len - len(inst["intervention_locations"][0]))])
            inst["intervention_locations"] = torch.cat([inst["intervention_locations"], _intervention_location_paddings], dim=-1).int()
            inst["intervention_masks"] = torch.cat([_intervention_mask, _intervention_mask_paddings], dim=-1).int()

            _input_id_paddings = torch.tensor(
                [self.tokenizer.pad_token_id for _ in range(max_seq_len - non_pad_len)])
            inst["input_ids"] = torch.cat((inst["input_ids"], torch.tensor([self.tokenizer.pad_token_id]), _input_id_paddings)).int()
            inst["attention_mask"] = (inst["input_ids"] != self.tokenizer.pad_token_id).int()
            inst["labels"] = inst["labels"].int()
        batch_inputs = self.data_collator(instances)
        return batch_inputs


def make_data_module(
    tokenizer: transformers.PreTrainedTokenizer, model, df, prefix_length=1
):
    all_input_ids, all_labels, all_intervention_locations = [], [], []
    for _, row in df.iterrows():
        input_ids = tokenizer(
            row["input"], max_length=1024, truncation=True, return_tensors="pt")["input_ids"][0]
        base_length = len(input_ids)
        intervention_locations = torch.tensor([[i for i in range(prefix_length, base_length)]])
        all_input_ids.append(input_ids)
        all_labels.append(row["labels"])
        all_intervention_locations.append(intervention_locations)

    train_dataset = datasets.Dataset.from_dict({
        "input_ids": all_input_ids,
        "labels": all_labels,
        "intervention_locations": all_intervention_locations
    })
    train_dataset.set_format(type='torch', columns=['input_ids', 'labels', 'intervention_locations'])

    data_collator_fn = transformers.DefaultDataCollator(
        return_tensors="pt"
    )
    data_collator = DataCollator(tokenizer=tokenizer, data_collator=data_collator_fn)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


class LinearProbe(Model):
    def __str__(self):
        return 'LinearProbe'

    def _training_arg(self, name, default=None):
        return getattr(self.training_args, name, default) if self.training_args is not None else default

    def _probe_transform(self, **kwargs):
        return kwargs.get(
            "direction_transform",
            kwargs.get(
                "probe_transform",
                self._training_arg("direction_transform", self._training_arg("probe_transform", "none")),
            ),
        )

    @torch.no_grad()
    def _activation_dataset(self, train_dataloader, prefix_length: int):
        activations, labels = [], []
        for batch in train_dataloader:
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            acts = gather_residual_activations(
                self.model,
                self.layer,
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                },
            ).detach()
            mask = inputs["intervention_masks"].bool()
            token_acts = acts[:, prefix_length : prefix_length + mask.shape[1]]
            expanded_labels = inputs["labels"].unsqueeze(-1).expand_as(mask)
            activations.append(token_acts[mask].detach().cpu().float())
            labels.append(expanded_labels[mask].detach().cpu().float())
        if not activations:
            raise ValueError("No probe activations were collected.")
        return torch.cat(activations, dim=0), torch.cat(labels, dim=0)

    def make_model(self, **kwargs):
        mode = kwargs.get("mode", "latent")
        if mode == "steering":
            intervention_type = kwargs.get("intervention_type", "addition")
            if intervention_type == "addition":
                ax = AdditionIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            elif intervention_type == "clamping":
                ax = SubspaceIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
        else:
            ax = ProbeIntervention(
                embed_dim=self.model.config.hidden_size, 
                low_rank_dimension=kwargs.get("low_rank_dimension", 1),
            )
        layers = self.steering_layers if self.steering_layers else [self.layer]
        self.ax = ax.to(self.device)
        self.ax.train()
        ax_config = IntervenableConfig(representations=[{
            "layer": l,
            "component": f"model.layers[{l}].output",
            "low_rank_dimension": kwargs.get("low_rank_dimension", 1),
            "intervention": self.ax} for l in layers])
        ax_model = IntervenableModel(ax_config, self.model)
        ax_model.set_device(self.device)
        self.ax_model = ax_model
    
    def make_dataloader(self, examples, **kwargs):
        data_module = make_data_module(self.tokenizer, self.model, examples)
        train_dataloader = DataLoader(
            data_module["train_dataset"], shuffle=True, batch_size=self.training_args.batch_size, 
            collate_fn=data_module["data_collator"])
        return train_dataloader

    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples, **kwargs)
        torch.cuda.empty_cache()
        transform = self._probe_transform(**kwargs)
        if transform not in {"none", "projected_gradient", "projected_activations"}:
            raise ValueError(f"Unknown probe transform: {transform!r}")

        projector = resolve_or_build_projector(self, examples, kwargs) if transform != "none" else None
        if transform in {"projected_gradient", "projected_activations"} and projector is None:
            raise ValueError(f"Probe transform {transform!r} requires a realizable basis/projector.")

        prefix_length = int(kwargs.get("prefix_length", 1))
        X, y = self._activation_dataset(train_dataloader, prefix_length)
        if transform == "projected_activations":
            X = projector.project(X)

        cfg = {
            "device": self.device,
            "seed": self.seed,
            "epochs": self._training_arg("n_epochs", 50),
            "lr": self._training_arg("lr", 1e-4),
            "batch_size": self._training_arg("batch_size", 4096),
            "weight_decay": self._training_arg("weight_decay", 0.0),
            "project_gradients": self._training_arg("project_gradients", False),
        }
        probe_projector = projector if transform == "projected_gradient" else None
        direction = train_probe_direction(X, y, cfg, projector=probe_projector)
        if transform == "projected_activations":
            direction = unit_norm(projector.project(direction)).squeeze(0)

        self.ax.proj.weight.data[0].copy_(direction.to(self.ax.proj.weight.device, dtype=self.ax.proj.weight.dtype))
        if self.ax.proj.bias is not None:
            self.ax.proj.bias.data.zero_()
        logger.warning("Training finished.")

    @torch.no_grad()
    def predict_latent(self, examples, **kwargs):
        self.ax.eval()
        batch_size = kwargs.get('batch_size', 32)
        
        all_acts = []
        all_max_act = []
        all_max_act_idx = []
        all_max_token = []
        all_tokens = []
        # Process in batches
        for i in range(0, len(examples), batch_size):
            batch = examples.iloc[i:i + batch_size]
            # Batch encode all inputs
            inputs = self.tokenizer(
                batch["input"].tolist(), return_tensors="pt", 
                add_special_tokens=True, padding=True, truncation=True).to(self.device)
            
            gather_acts = gather_residual_activations(
                self.model, self.layer, inputs)
            outputs = self.ax(
                gather_acts[:, kwargs["prefix_length"]:],  # no bos token
                subspaces={
                    "subspaces": torch.tensor(batch["concept_id"].tolist()).to(self.device),
                    "k": 1
                })
            ax_acts = outputs.latent[-1]
            seq_lens = inputs["attention_mask"].sum(dim=1) - kwargs["prefix_length"] # no bos token
            # Process each sequence in the batch
            for seq_idx, ax_seq in enumerate(ax_acts):
                acts = ax_seq[:seq_lens[seq_idx]].flatten().data.float().cpu().numpy().tolist()
                acts = [round(x, 3) for x in acts]
                max_act = max(acts)
                max_act_indices = [i for i, x in enumerate(acts) if x == max_act]
                max_act_idx = max_act_indices[0]
                # Get tokens for this specific sequence
                tokens = self.tokenizer.tokenize(batch.iloc[seq_idx]["input"])[kwargs["prefix_length"]-1:] # -1 is because it does not prepend BOS token
                max_token = tokens[max_act_idx]
                all_acts.append(acts)
                all_max_act.append(max_act)
                all_max_act_idx.append(max_act_idx)
                all_max_token.append(max_token)
                all_tokens.append(tokens)
        return {
            "acts": all_acts,
            "max_act": all_max_act,
            "max_act_idx": all_max_act_idx,
            "max_token": all_max_token,
            "tokens": all_tokens
        }
    
    @torch.no_grad()
    def predict_latents(self, examples, **kwargs):
        self.ax.eval()
        batch_size = kwargs.get('batch_size', 32)
        
        all_acts = []
        all_max_act = []
        all_max_act_idx = []
        all_max_token = []
        all_tokens = []
        # Process in batches
        for i in range(0, len(examples), batch_size):
            batch = examples.iloc[i:i + batch_size]
            # Batch encode all inputs
            inputs = self.tokenizer(
                batch["input"].tolist(), return_tensors="pt", 
                add_special_tokens=True, padding=True, truncation=True).to(self.device)
            
            gather_acts = gather_residual_activations(
                self.model, self.layer, inputs)
            ax_acts_batch = torch.matmul(
                gather_acts[:, kwargs["prefix_length"]:], # bs, s, h
                self.ax.proj.weight.permute(1, 0) # h, d
            ).float().cpu().numpy()
            
            # Process each sequence in the batch
            seq_lens = inputs["attention_mask"].sum(dim=1) - kwargs["prefix_length"] # no bos token
            for seq_idx, row in enumerate(batch.itertuples()):
                # select acts with attention mask
                acts_batch = ax_acts_batch[
                    seq_idx, :seq_lens[seq_idx]]
                
                concept_acts = []
                concept_max_act = []
                concept_max_act_idx = []
                concept_max_token = []
                concept_tokens = []
                for row_idx in range(ax_acts_batch.shape[-1]):
                    # row_idx here is the concept id
                    acts = acts_batch[:, row_idx].flatten().tolist()
                    acts = [round(x, 3) for x in acts]
                    max_act = max(acts)
                    max_act_indices = [i for i, x in enumerate(acts) if x == max_act]
                    max_act_idx = max_act_indices[0]
                    # Get tokens for this specific sequence
                    tokens = self.tokenizer.tokenize(row.input)[kwargs["prefix_length"]-1:] # -1 is because it does not prepend BOS token
                    max_token = tokens[max_act_idx]
                    concept_acts.append(acts)
                    concept_max_act.append(max_act)
                    concept_max_act_idx.append(max_act_idx)
                    concept_max_token.append(max_token)
                    concept_tokens.append(tokens)
                all_acts.append(concept_acts)
                all_max_act.append(concept_max_act)
                all_max_act_idx.append(concept_max_act_idx)
                all_max_token.append(concept_max_token)
                all_tokens.append(concept_tokens)
        return {
            # "acts": all_acts,
            "max_act": all_max_act,
            # "max_act_idx": all_max_act_idx,
            # "max_token": all_max_token,
            # "tokens": all_tokens
        }
