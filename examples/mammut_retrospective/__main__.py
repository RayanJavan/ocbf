"""Inspect the retained archive and publish its explicit initial admission gate."""

import argparse

from .admission import initial_gate
from .archive import inspect_archive
from .report import write_admission_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="retained data-extract root (read only)")
    parser.add_argument("--output", default="artifacts/mammut-retrospective")
    args = parser.parse_args()
    audit = inspect_archive(args.archive)
    gate = initial_gate(audit)
    write_admission_report(args.output, audit, gate)
    print(f"Source admission incomplete; admission report written to {args.output}")


if __name__ == "__main__":
    main()
