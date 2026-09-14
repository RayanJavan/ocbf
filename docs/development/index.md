# Development

Contribute to OCBF while keeping source interpretation, probability semantics, numerical
execution, and process evaluation independently maintainable. To add an interpreter,
channel, engine, or evaluator in your own code, see [Extend OCBF](../how-to/extend.md).

## Set up

Work from the checkout with the `dev` extra installed:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Optional native validation additionally uses `.[oracles]`. Documentation uses the
[pinned documentation environment](documentation.md#documentation-environment). The default
development and documentation checks do not require a factory archive or credentials.

## Before opening a pull request

- Run the [routine checks](validation.md#routine-checks), and the
  [documentation checks](validation.md#documentation-checks) when docs, examples, or
  public docstrings change.
- Update the owning docstring or reference page, the affected guide, and meaningful
  validation in the same change; see [when behavior changes](documentation.md#when-behavior-changes).
- CI runs the test suite and the documentation build as separate required checks; Read the
  Docs builds a preview of the site.

## Guides

- [Architecture](architecture.md): responsibilities and dependency direction.
- [Validation](validation.md): scientific reference checks and documentation checks.
- [Documentation maintenance](documentation.md): content ownership and build gates.
- [Documentation hosting](hosting.md): Read the Docs settings, previews, and versions.
