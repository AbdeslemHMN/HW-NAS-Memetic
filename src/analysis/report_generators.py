"""
Architectural Genotype Validation and Uniqueness Reporting.

Iterates the full 3×3 (dataset × hardware) sweep, computes set-algebra
statistics on each algorithm's Pareto front at the architecture-string level,
and writes a concise Markdown summary.

Output
------
``results/reports/pareto/uniqueness_summary.md``
    Master Markdown table with one row per (dataset, hardware) combination.

``results/reports/pareto/uniqueness_detail.md``
    Per-combination detailed breakdowns including individual baseline columns.
"""
from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from pathlib import Path

from src.analysis.comparisons import (
    analyze_all_baselines,
    MultiBaselineOverlap,
)
from src.analysis.data_loader import (
    DATASETS,
    HARDWARE,
    ALGO_SPEC,
    combo_archive_sets,
)
from src.api.hw_nas_wrapper import HWNASApi


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Minimal Markdown table formatter."""
    col_widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    header_row = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    body = "\n".join(
        "| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(row)) + " |"
        for row in rows
    )
    return "\n".join([header_row, sep, body])


def _fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def _hw_label(hw: str) -> str:
    """Short display label for hardware identifiers."""
    mapping = {
        "edgegpu_latency":  "Edge GPU",
        "raspi4_latency":   "Raspi4",
        "eyeriss_latency":  "Eyeriss",
    }
    return mapping.get(hw, hw)


def _ds_label(ds: str) -> str:
    return ds  # keep as-is; short enough


def _parse_architecture_ops(arch_str: str) -> list[str]:
    """Extract operation names from a NAS-Bench architecture string.

    Supports both named NAS-Bench strings and integer index lists.
    """
    import ast
    import re

    ops: list[str] = []
    if "[" in arch_str and "]" in arch_str:
        try:
            parsed = ast.literal_eval(arch_str)
            if isinstance(parsed, (list, tuple)) and all(isinstance(x, int) for x in parsed):
                ops = [HWNASApi.OPS[i] for i in parsed if 0 <= i < len(HWNASApi.OPS)]
                return ops
        except (ValueError, SyntaxError):
            pass

    groups = [group.strip("|") for group in arch_str.split("+") if group.strip("|")]
    for group in groups:
        ops.extend(re.findall(r"([a-z_0-9]+)~\d+", group))

    return ops


def _structural_profile_lines(archs: list[str]) -> list[str]:
    """Summarise op-level counts for a list of unique architecture strings."""
    op_counts: dict[str, int] = {}
    for arch in archs:
        ops = _parse_architecture_ops(arch)
        for op in ops:
            op_counts[op] = op_counts.get(op, 0) + 1

    total_archs = len(archs)
    total_edges = sum(op_counts.values())
    lines = [
        f"- **Unique architecture count**: {total_archs}",
        f"- **Total operation edges**: {total_edges} ({6 * total_archs} expected if all strings are valid)",
    ]

    if total_archs > 0:
        most_common = sorted(op_counts.items(), key=lambda x: x[1], reverse=True)
        for op, count in most_common:
            pct = 100.0 * count / total_edges if total_edges else 0.0
            lines.append(f"- **{op}**: {count} ({pct:.1f}% of unique op occurrences)")

        conv3 = op_counts.get("nor_conv_3x3", 0)
        skip = op_counts.get("skip_connect", 0)
        if total_edges:
            lines.append(f"- **3x3 convolutions**: {conv3} ({100.0 * conv3 / total_edges:.1f}% of ops)")
            lines.append(f"- **Skip connections**: {skip} ({100.0 * skip / total_edges:.1f}% of ops)")
        else:
            lines.append(f"- **3x3 convolutions**: {conv3}")
            lines.append(f"- **Skip connections**: {skip}")
    return lines


def _collect_overlap_results(
    results_dir: Path,
    ds_list: list[str],
    hw_list: list[str],
    spec: dict[str, tuple[str, str]],
    proposed_key: str,
) -> list[MultiBaselineOverlap]:
    results: list[MultiBaselineOverlap] = []
    baseline_names = [k for k in spec if k != proposed_key]

    for ds in ds_list:
        for hw in hw_list:
            archive_sets = combo_archive_sets(results_dir, ds, hw, spec)
            proposed_archives = archive_sets.get(proposed_key, [])
            baseline_archives = {
                bname: archive_sets.get(bname, [])
                for bname in baseline_names
            }
            overlap = analyze_all_baselines(
                dataset=ds,
                hardware=hw,
                proposed_archives=proposed_archives,
                baselines=baseline_archives,
            )
            results.append(overlap)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Core generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_uniqueness_reports(
    results_dir: Path,
    output_dir: Path,
    datasets: list[str] | None = None,
    hardware: list[str] | None = None,
    algo_spec: dict | None = None,
    proposed_key: str = "Proposed",
) -> None:
    """Run the 3×3 sweep and write Markdown reports to *output_dir*.

    :param results_dir:  Project ``results/`` root.
    :param output_dir:   Output directory for reports (will be created).
    :param datasets:     Override dataset list (default: :data:`DATASETS`).
    :param hardware:     Override hardware list (default: :data:`HARDWARE`).
    :param algo_spec:    Override algorithm spec (default: :data:`ALGO_SPEC`).
    :param proposed_key: Key in *algo_spec* that refers to the Proposed alg.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    ds_list = datasets if datasets is not None else DATASETS
    hw_list = hardware if hardware is not None else HARDWARE
    spec    = algo_spec if algo_spec is not None else ALGO_SPEC
    baseline_names = [k for k in spec if k != proposed_key]

    all_results = _collect_overlap_results(
        results_dir=results_dir,
        ds_list=ds_list,
        hw_list=hw_list,
        spec=spec,
        proposed_key=proposed_key,
    )

    # ── Build summary table ───────────────────────────────────────────────────
    now_utc = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    summary_headers = (
        ["Dataset", "Hardware", "Proposed|Front", "Global|Novelty%"]
        + [f"{b}|Front" for b in baseline_names]
        + [f"Shared vs|{b}" for b in baseline_names]
        + [f"Unique vs|{b}" for b in baseline_names]
        + ["Fully Unique|(all BLs)"]
    )

    summary_rows: list[list[str]] = []
    for ovlp in all_results:
        row = [
            _ds_label(ovlp.dataset),
            _hw_label(ovlp.hardware),
            str(ovlp.proposed_total),
            _fmt_pct(ovlp.global_novelty_score),
        ]
        for bname in baseline_names:
            r = ovlp.per_baseline.get(bname)
            row.append(str(r.baseline_total) if r else "N/A")
        for bname in baseline_names:
            r = ovlp.per_baseline.get(bname)
            row.append(str(r.shared_count) if r else "N/A")
        for bname in baseline_names:
            r = ovlp.per_baseline.get(bname)
            row.append(str(r.unique_to_proposed_count) if r else "N/A")
        row.append(str(len(ovlp.fully_unique_to_proposed)))
        summary_rows.append(row)

    # Flatten header list (remove "|" used for readability above)
    clean_headers = [h.replace("|", " ") for h in summary_headers]

    summary_md = textwrap.dedent(f"""\
        # Architectural Genotype Uniqueness Summary
        _Generated: {now_utc}_

        ## Overview

        This report measures how many distinct architecture strings (genotypes)
        appear on the Pareto front of each algorithm, and how many of the
        Proposed algorithm's Pareto-optimal architectures were **not** discovered
        by any baseline.

        **Novelty score** = (unique-to-proposed / proposed front size) × 100%

        ## Summary Table

    """)
    summary_md += _md_table(clean_headers, summary_rows)
    summary_md += "\n"

    # ── Build detailed report ─────────────────────────────────────────────────
    detail_sections: list[str] = [
        textwrap.dedent(f"""\
            # Architectural Genotype Uniqueness — Detailed Report
            _Generated: {now_utc}_

        """)
    ]

    for ovlp in all_results:
        section_lines = [
            f"## {_ds_label(ovlp.dataset)} / {_hw_label(ovlp.hardware)}\n",
            f"- **Proposed front size**: {ovlp.proposed_total} distinct genotypes",
            f"- **Global novelty**: {_fmt_pct(ovlp.global_novelty_score)} "
            f"({len(ovlp.fully_unique_to_proposed)} archs missed by ALL baselines)",
            "",
        ]

        for bname in baseline_names:
            r = ovlp.per_baseline.get(bname)
            if r is None:
                section_lines.append(f"### vs {bname}: no data\n")
                continue

            section_lines += [
                f"### vs {bname}",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Baseline front size | {r.baseline_total} |",
                f"| Shared architectures | {r.shared_count} |",
                f"| Unique to Proposed | {r.unique_to_proposed_count} |",
                f"| Unique to {bname} | {r.unique_to_baseline_count} |",
                f"| Novelty score (Proposed vs {bname}) | {_fmt_pct(r.novelty_score)} |",
                "",
            ]

            if r.unique_to_proposed:
                section_lines.append(
                    "<details><summary>Architectures unique to Proposed "
                    f"(vs {bname})</summary>\n"
                )
                for arch in sorted(r.unique_to_proposed):
                    section_lines.append(f"- `{arch}`")
                section_lines.append("</details>\n")

        detail_sections.append("\n".join(section_lines))

    detail_md = "\n---\n\n".join(detail_sections)

    # ── Write files ───────────────────────────────────────────────────────────
    summary_path = output_dir / "uniqueness_summary.md"
    detail_path  = output_dir / "uniqueness_detail.md"

    summary_path.write_text(summary_md, encoding="utf-8")
    detail_path.write_text(detail_md, encoding="utf-8")

    print(f"[report_generators] Summary  → {summary_path}")
    print(f"[report_generators] Detail   → {detail_path}")


def generate_genotype_catalog(
    results_dir: Path,
    output_dir: Path,
    datasets: list[str] | None = None,
    hardware: list[str] | None = None,
    algo_spec: dict | None = None,
    proposed_key: str = "Proposed",
) -> None:
    """Create a catalog of fully unique architecture strings for Proposed."""
    output_dir.mkdir(parents=True, exist_ok=True)

    ds_list = datasets if datasets is not None else DATASETS
    hw_list = hardware if hardware is not None else HARDWARE
    spec    = algo_spec if algo_spec is not None else ALGO_SPEC

    all_results = _collect_overlap_results(
        results_dir=results_dir,
        ds_list=ds_list,
        hw_list=hw_list,
        spec=spec,
        proposed_key=proposed_key,
    )

    now_utc = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections: list[str] = [
        textwrap.dedent(f"""\
            # Genotype Catalog — Fully Unique Proposed Architectures
            _Generated: {now_utc}_

            This catalog records the exact architecture strings discovered by the
            Proposed search that were not found by any baseline algorithm.

        """)
    ]

    for ovlp in all_results:
        section = [
            f"## {_ds_label(ovlp.dataset)} / {_hw_label(ovlp.hardware)}",
            "",
        ]

        unique_archs = ovlp.fully_unique_to_proposed_list
        if not unique_archs:
            section.append("No fully unique Proposed architectures were found for this combination.")
            sections.append("\n".join(section))
            continue

        section.append(f"- **Fully unique Proposed architectures**: {len(unique_archs)}")
        section.append("- **Structural profile:**")
        section.extend(_structural_profile_lines(unique_archs))
        section.append("")
        section.append("<details><summary>Show fully unique architectures</summary>")
        section.append("")
        for arch in unique_archs:
            section.append(f"- `{arch}`")
        section.append("</details>")

        sections.append("\n".join(section))

    catalog_md = "\n---\n\n".join(sections)
    catalog_path = output_dir / "genotype_catalog.md"
    catalog_path.write_text(catalog_md, encoding="utf-8")
    print(f"[report_generators] Catalog  → {catalog_path}")
