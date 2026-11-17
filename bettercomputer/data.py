"""Samples, batching, and a synthetic task used to prove the model can learn.

A sample is one decision: an intent, the elements on screen, which element is the target
(or none), and the action. Real data (parsed screenshots) will produce the same Sample type.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from .model import ACTIONS, ELEMENT_TYPES


@dataclass
class Element:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in 0..1
    type: str
    text: str


@dataclass
class Sample:
    intent: str
    elements: list[Element]
    target: int  # index into elements, or -1 when the target is not on screen
    action: str


class Collator:
    """Turns a list of Samples into the tensors SelectorModel.forward expects."""

    def __init__(self, tokenizer, max_intent_tokens: int = 64, max_text_tokens: int = 12, max_elements: int = 256):
        self.tok = tokenizer
        self.max_intent = max_intent_tokens
        self.max_text = max_text_tokens
        self.max_elements = max_elements

    def _ids(self, texts: list[str], limit: int) -> list[list[int]]:
        if not texts:
            return []
        return [ids[:limit] for ids in self.tok(texts, add_special_tokens=False)["input_ids"]]

    def __call__(self, samples: list[Sample]) -> dict[str, torch.Tensor]:
        B = len(samples)
        for s in samples:
            if len(s.elements) > self.max_elements:
                raise ValueError(f"sample has {len(s.elements)} elements, limit is {self.max_elements}")
        intents = self._ids([f"Task: {s.intent}\nScreen:" for s in samples], self.max_intent)
        L = max(len(x) for x in intents)
        N = max(1, max(len(s.elements) for s in samples))
        T = self.max_text

        intent_ids = torch.zeros(B, L, dtype=torch.long)
        intent_mask = torch.zeros(B, L, dtype=torch.long)
        boxes = torch.zeros(B, N, 4)
        types = torch.zeros(B, N, dtype=torch.long)
        text_ids = torch.zeros(B, N, T, dtype=torch.long)
        text_mask = torch.zeros(B, N, T, dtype=torch.long)
        elem_mask = torch.zeros(B, N, dtype=torch.long)

        flat_text = self._ids([e.text for s in samples for e in s.elements], T)
        k = 0
        for i, s in enumerate(samples):
            intent_ids[i, : len(intents[i])] = torch.tensor(intents[i])
            intent_mask[i, : len(intents[i])] = 1
            for j, e in enumerate(s.elements):
                boxes[i, j] = torch.tensor(e.box)
                types[i, j] = ELEMENT_TYPES.index(e.type) if e.type in ELEMENT_TYPES else 0
                ids = flat_text[k]
                k += 1
                if ids:
                    text_ids[i, j, : len(ids)] = torch.tensor(ids)
                    text_mask[i, j, : len(ids)] = 1
                elem_mask[i, j] = 1

        return {
            "intent_ids": intent_ids,
            "intent_mask": intent_mask,
            "elem_boxes": boxes,
            "elem_types": types,
            "elem_text_ids": text_ids,
            "elem_text_mask": text_mask,
            "elem_mask": elem_mask,
            "target": torch.tensor([s.target + 1 for s in samples], dtype=torch.long),  # 0 = none
            "action": torch.tensor([ACTIONS.index(s.action) for s in samples], dtype=torch.long),
        }


# ---- synthetic task ----------------------------------------------------------------------

_WORDS = """save open close cancel submit search settings profile home back next previous delete edit
copy paste download upload share print help login logout register cart checkout buy order payment
address shipping account message inbox send reply forward archive folder file view tools window new
refresh reload zoom filter sort export import preview rename move undo redo format insert table image
link comment review rating price total discount coupon continue confirm accept decline allow deny
notifications privacy security language theme password username email phone country city calendar
today tomorrow week month year start stop pause play record volume mute camera microphone screen""".split()

_VERBS = {
    "click": ["click", "press", "select", "left-click"],
    "double_click": ["double-click", "double click"],
    "right_click": ["right-click", "right click", "open the context menu of"],
}
_SYNTH_TYPES = ["button", "link", "text", "input", "icon", "checkbox", "menu", "tab"]


def _box(rng: random.Random, y_range: tuple[float, float] = (0.0, 1.0)) -> tuple[float, float, float, float]:
    w, h = rng.uniform(0.02, 0.2), rng.uniform(0.015, 0.06)
    x = rng.uniform(0.0, 1.0 - w)
    y = rng.uniform(y_range[0], max(y_range[0], y_range[1] - h))
    return (x, y, x + w, y + h)


def synthetic_sample(rng: random.Random, min_elements: int = 6, max_elements: int = 40, p_none: float = 0.15) -> Sample:
    """One made-up screen. Three kinds, so text, 'none' and box position all get tested:

    - by text:      'press the "save file" button'              -> the element with that text
    - none:         the named element is not on screen           -> target -1
    - by position:  two elements share a text; 'the upper one'   -> needs the boxes
    """
    n = rng.randint(min_elements, max_elements)
    used: set[str] = set()
    elements: list[Element] = []
    while len(elements) < n:
        text = " ".join(rng.sample(_WORDS, rng.randint(1, 3)))
        if text in used:
            continue
        used.add(text)
        elements.append(Element(_box(rng), rng.choice(_SYNTH_TYPES), text))

    action = rng.choice(list(_VERBS))
    verb = rng.choice(_VERBS[action])
    roll = rng.random()

    if roll < p_none:
        while True:
            text = " ".join(rng.sample(_WORDS, rng.randint(1, 3)))
            if text not in used:
                break
        return Sample(f'{verb} the "{text}" {rng.choice(_SYNTH_TYPES)}', elements, -1, action)

    target = rng.randrange(n)
    t = elements[target]
    if roll < p_none + 0.25:
        # Put a twin with the same text and type on the other half of the screen;
        # only vertical position tells the two apart.
        upper = rng.random() < 0.5
        top, bottom = (0.0, 0.45), (0.55, 1.0)
        me = Element(_box(rng, top if upper else bottom), t.type, t.text)
        twin = Element(_box(rng, bottom if upper else top), t.type, t.text)
        elements[target] = me
        elements.insert(rng.randrange(len(elements) + 1), twin)
        target = next(i for i, e in enumerate(elements) if e is me)
        return Sample(f'{verb} the {"upper" if upper else "lower"} "{t.text}" {t.type}', elements, target, action)

    template = rng.choice(['{v} the "{x}" {t}', '{v} "{x}"', "{v} the {t} labelled {x}", "{v} {x}"])
    return Sample(template.format(v=verb, x=t.text, t=t.type), elements, target, action)


def synthetic_batches(collator: Collator, batch_size: int, seed: int, **kwargs):
    """Endless stream of collated synthetic batches."""
    rng = random.Random(seed)
    while True:
        yield collator([synthetic_sample(rng, **kwargs) for _ in range(batch_size)])
