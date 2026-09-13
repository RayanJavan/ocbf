# Models and constraints

A model describes possible histories and how the supplied evidence and assumptions
distinguish them. `ModelSpec` contains the inputs; compilation produces a
`CompiledModel` that inference engines can evaluate.

## Possible histories

In the synthetic assessment, a start candidate and two end candidates can be associated
with one Operation. Their endpoint times are supplied, while occurrence and association
remain uncertain.

| Candidate history | Meaning for the Operation |
|---|---|
| Start and short end are associated | A closed execution with the shorter duration. |
| Start and long end are associated | A closed execution with the longer duration. |
| Start is associated, but no end is associated | An open execution at the query horizon. |
| Both ends are associated | Excluded by the example's at-most-one-end support factor. |

These are illustrative cases within the model's possible assignments. Reports affect
their relative probability; a report does not simply select one row.

## Model inputs

| Part of `ModelSpec` | Responsibility |
|---|---|
| Context | Candidate objects, events, types, and assertion meanings. |
| Interpreted evidence | Admitted report content with lineage and information grouping. |
| Parameters | Resolved channel values, priors, and explicit assumptions. |
| Variables and factors | Additional uncertain quantities and relationships among them. |
| Scope and decoding | The assessment boundary and the conversion from assignments to process histories. |

Compilation checks these inputs and constructs factors. A factor is a local contribution
that relates a set of variables: for example, an observation likelihood, an event/link
consistency rule, or a count constraint.

## Constraints and references

Hard support excludes histories. A descriptive preference makes some allowed histories
more plausible. A normative reference defines the question asked of the resulting
histories.

The example's at-most-one-end constraint excludes double association. Its 30-minute
duration reference does something different: a 60-minute history remains possible and
counts as a violation. A model that excluded durations above 30 minutes would already
have ruled out the outcome the query is meant to assess.

Contradictory support can leave no possible history. Compilation or inference then raises
an incompatibility error; it does not manufacture a replacement distribution.

## Compilation and inference

Compilation does not choose the most likely history or calculate query answers.
Different admitted engines can operate on the same compiled model. Their representations
and numerical qualifications differ, while the supplied scientific target stays fixed.

Decoding turns supported assignments into bounded object-centric histories. It preserves
event identity, associations, and the distinction between inactive and missing values.
This representation is separate from standard OCEL file I/O.

Related procedures: [constraints](../how-to/configure-constraints.md) and
[inference selection](../how-to/choose-inference.md).
