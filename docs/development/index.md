# Development

Use the existing public contracts to keep source interpretation, probability semantics,
numerical execution, and process evaluation independently maintainable.

- [Architecture](architecture.md): responsibilities and dependency direction.
- [Validation](validation.md): scientific reference checks and reproducible workloads.
- [Documentation maintenance](documentation.md): content ownership and build gates.
- [Documentation hosting](hosting.md): Read the Docs settings, previews, and versions.

Work from the checkout with `.[dev]` installed for library tests. Documentation uses the
[pinned documentation environment](documentation.md#documentation-environment).
Optional native validation additionally uses `.[oracles]`. The default development and
documentation checks do not require a factory archive or credentials.
