"""Exact execution codecs. The original target and its prior mass remain authoritative."""

from dataclasses import dataclass

from ocbf._values import freeze
from ocbf.errors import ValidationError


@dataclass(frozen=True)
class ReductionCodec:
    clamped: object
    choices: object
    domains: object

    def __post_init__(self):
        for name in ("clamped", "choices", "domains"):
            object.__setattr__(self, name, freeze(getattr(self, name)))

    def expand(self, state):
        expanded = dict(self.clamped)
        for key, value in state.items():
            if key in self.choices:
                if value not in self.choices[key]:
                    raise ValidationError("invalid exactly-one choice", key=key)
                expanded.update({original: original == value for original in self.choices[key]})
            else:
                expanded[key] = value
        return expanded

    def encoded_key(self, key):
        return next((choice for choice, originals in self.choices.items() if key in originals), key)


def reduction_codec(model):
    clamped = {k: d[0] for k, d in model.domains.items() if len(d) == 1}
    domains = {k: d for k, d in model.domains.items() if k not in clamped}
    choices = {}
    for factor in sorted(model.factors, key=lambda f: f.key):
        if (
            factor.family == "count"
            and factor.role == "support"
            and factor.parameters.get("lo") == factor.parameters.get("hi") == 1
            and factor.scope
            and all(k in domains and domains[k] == (False, True) for k in factor.scope)
        ):
            key = "codec:exactly-one:" + factor.key
            if key in domains or key in model.continuous:
                raise ValidationError("reduction key collision", key=key)
            choices[key] = factor.scope
            for original in factor.scope:
                del domains[original]
            domains[key] = factor.scope
    return ReductionCodec(clamped, choices, domains)
