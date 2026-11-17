"""Selector model.

A frozen Qwen3.5 body with LoRA layers, plus three small parts trained from scratch:

- ElementAdapter: turns one parsed screen element (box, type, text) into one input vector.
- pointer head:   scores every element against the intent and picks one, or "none of these".
- action head:    picks what to do with it (click, type, ...).

Sequence fed to the body:  [intent tokens] [one vector per element] [query vector]
The body is causal, so the query position has seen everything and each element position
has seen the intent. The pointer head compares the query's hidden state with each element's.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

BASE = "Qwen/Qwen3.5-0.8B-Base"

ELEMENT_TYPES = ["other", "button", "link", "text", "input", "icon", "image", "checkbox", "menu", "tab"]
ACTIONS = ["click", "double_click", "right_click", "type", "key", "scroll", "done"]

# Linear layers inside the body that get a LoRA update: attention projections of the
# full-attention layers, the projections of the recurrent (Gated DeltaNet) layers, and the MLPs.
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "out_proj",
    "gate_proj", "up_proj", "down_proj",
]


@dataclass
class SelectorConfig:
    base: str = BASE
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    box_freqs: int = 8
    type_dim: int = 64
    pointer_dim: int = 256
    n_types: int = len(ELEMENT_TYPES)
    n_actions: int = len(ACTIONS)


class LoRALinear(nn.Module):
    """A frozen Linear plus a trainable low-rank update: y = W x + (alpha / r) * B A x."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))  # zero: starts as a no-op
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def weight(self) -> torch.Tensor:  # some callers read .weight.dtype of the layer they call
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(self.dropout(x), self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return self.base(x) + update * self.scaling


def add_lora(module: nn.Module, targets: list[str], rank: int, alpha: int, dropout: float) -> int:
    """Wraps every nn.Linear whose attribute name is in `targets`. Returns how many were wrapped."""
    wrapped = 0
    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(parent, name, LoRALinear(child, rank, alpha, dropout))
                wrapped += 1
    return wrapped


class ElementAdapter(nn.Module):
    """(box, type, pooled text embedding) -> one vector in the body's input space."""

    def __init__(self, hidden: int, cfg: SelectorConfig, out_norm: float):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(cfg.box_freqs)) * math.pi, persistent=False)
        box_dim = 4 * 2 * cfg.box_freqs
        self.type_emb = nn.Embedding(cfg.n_types, cfg.type_dim)
        self.mlp = nn.Sequential(
            nn.Linear(box_dim + cfg.type_dim + hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.scale = nn.Parameter(torch.tensor(float(out_norm)))

    def forward(self, boxes: torch.Tensor, types: torch.Tensor, text_vecs: torch.Tensor) -> torch.Tensor:
        # boxes [B,N,4] as x1,y1,x2,y2 in 0..1 | types [B,N] | text_vecs [B,N,H]
        angles = boxes.unsqueeze(-1) * self.freqs
        box_feat = torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)
        x = self.mlp(torch.cat([box_feat, self.type_emb(types), text_vecs], dim=-1))
        # Residual: at the start of training an element already looks like "the average of
        # its words", which the pretrained body can read.
        x = x + text_vecs
        return F.normalize(x, dim=-1) * self.scale


class SelectorModel(nn.Module):
    def __init__(self, cfg: SelectorConfig | None = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        from transformers import AutoModelForCausalLM

        self.cfg = cfg or SelectorConfig()
        lm = AutoModelForCausalLM.from_pretrained(self.cfg.base, dtype=dtype)
        self.body = lm.model  # embeddings + layers + final norm; the vocabulary head is not used
        for p in self.body.parameters():
            p.requires_grad_(False)
        self.lora_layers = add_lora(self.body, LORA_TARGETS, self.cfg.lora_rank, self.cfg.lora_alpha, self.cfg.lora_dropout)
        if self.lora_layers == 0:
            raise RuntimeError("no layers matched LORA_TARGETS; the base model's layer names changed")
        for name, p in self.body.named_parameters():
            p.requires_grad_("lora_" in name)

        hidden = self.body.config.hidden_size
        emb = self.body.get_input_embeddings().weight
        with torch.no_grad():
            sample = emb[:: max(1, emb.shape[0] // 4096)].float()
            emb_norm = sample.norm(dim=-1).mean().item()
            emb_mean = sample.mean(dim=0)

        self.adapter = ElementAdapter(hidden, self.cfg, out_norm=emb_norm)
        self.query = nn.Parameter(emb_mean + 0.02 * torch.randn(hidden))
        self.ptr_query = nn.Linear(hidden, self.cfg.pointer_dim)
        self.ptr_elem = nn.Linear(hidden, self.cfg.pointer_dim)
        self.none = nn.Parameter(0.02 * torch.randn(self.cfg.pointer_dim))
        self.action_head = nn.Linear(hidden, self.cfg.n_actions)

    # ---- forward -------------------------------------------------------------------------

    def _pooled_text(self, text_ids: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        emb = self.body.get_input_embeddings()(text_ids).float()  # [B,N,T,H]
        mask = text_mask.unsqueeze(-1).float()
        return (emb * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embed = self.body.get_input_embeddings()
        tok = embed(batch["intent_ids"])  # [B,L,H]
        text_vecs = self._pooled_text(batch["elem_text_ids"], batch["elem_text_mask"])
        elems = self.adapter(batch["elem_boxes"].float(), batch["elem_types"], text_vecs).to(tok.dtype)

        intent_len = batch["intent_mask"].sum(dim=1)  # [B]
        n_elem = batch["elem_mask"].sum(dim=1)  # [B]
        B, n_max = elems.shape[0], elems.shape[1]
        seq_len = int((intent_len + n_elem).max().item()) + 1

        x = tok.new_zeros(B, seq_len, tok.shape[-1])
        mask = torch.zeros(B, seq_len, dtype=torch.long, device=tok.device)
        query = self.query.to(tok.dtype)
        for i in range(B):
            li, ni = int(intent_len[i]), int(n_elem[i])
            x[i, :li] = tok[i, :li]
            x[i, li : li + ni] = elems[i, :ni]
            x[i, li + ni] = query
            mask[i, : li + ni + 1] = 1

        hidden = self.body(inputs_embeds=x, attention_mask=mask).last_hidden_state.float()

        rows = torch.arange(B, device=hidden.device)
        h_query = hidden[rows, intent_len + n_elem]  # [B,H]
        elem_pos = intent_len.unsqueeze(1) + torch.arange(n_max, device=hidden.device).unsqueeze(0)
        elem_pos = elem_pos.clamp(max=seq_len - 1)
        h_elem = hidden[rows.unsqueeze(1), elem_pos]  # [B,N,H]

        q = self.ptr_query(h_query)  # [B,P]
        e = self.ptr_elem(h_elem)  # [B,N,P]
        scale = 1.0 / math.sqrt(q.shape[-1])
        elem_scores = torch.einsum("bp,bnp->bn", q, e) * scale
        elem_scores = elem_scores.masked_fill(batch["elem_mask"] == 0, float("-inf"))
        none_score = (q @ self.none).unsqueeze(1) * scale
        # index 0 = "none of these", index i+1 = element i
        pointer_logits = torch.cat([none_score, elem_scores], dim=1)
        return {"pointer_logits": pointer_logits, "action_logits": self.action_head(h_query)}

    @staticmethod
    def loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return F.cross_entropy(out["pointer_logits"], batch["target"]) + F.cross_entropy(
            out["action_logits"], batch["action"]
        )

    # ---- saving only what was trained ----------------------------------------------------

    def trainable_state(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.named_parameters() if v.requires_grad}

    def save_trainable(self, folder: str | Path) -> None:
        from safetensors.torch import save_file

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        save_file(self.trainable_state(), str(folder / "selector.safetensors"))
        (folder / "selector.json").write_text(json.dumps(asdict(self.cfg), indent=2))

    def load_trainable(self, folder: str | Path) -> None:
        from safetensors.torch import load_file

        state = load_file(str(Path(folder) / "selector.safetensors"))
        missing, unexpected = self.load_state_dict(state, strict=False)
        expected = set(self.trainable_state())
        not_loaded = expected - set(state)
        if not_loaded or unexpected:
            raise RuntimeError(f"checkpoint mismatch: not loaded {sorted(not_loaded)[:5]}, unexpected {unexpected[:5]}")

    def param_counts(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
