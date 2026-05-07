"""JobFit CLI — V2 entry point with intent routing.

Routes a user request through the V2 dispatcher (assistant/core.py),
which classifies intent and runs the matching specialized handler.

V1 fallback: if no --request is provided, just parses the JD into a
JobDescription (the original V1 behavior, preserved for backwards
compatibility).

Usage:
    # V2 — full routing
    python main.py examples/elad_jd.txt --request "Should I apply?"
    python main.py examples/elad_jd.txt --request "Write me a cover letter"
    python main.py examples/elad_jd.txt --request "Prep me for the interview"

    # V1 — JD parser only (no --request)
    python main.py examples/elad_jd.txt

    # JSON output (pipeable)
    python main.py examples/elad_jd.txt --request "Should I apply?" --json
"""
import argparse
import sys
from pathlib import Path

from assistant.chains.jd_parser import parse_job_description
from assistant.core import process_request
from models.schemas import CoverLetter, FitReport, InterviewPrep, JobDescription


# ─── Input helpers ─────────────────────────────────────────
def read_jd_from_stdin() -> str:
    print(
        "Paste the job description below, then press "
        "Ctrl+D (Linux/Mac) or Ctrl+Z + Enter (Windows):",
        file=sys.stderr,
    )
    print("─" * 60, file=sys.stderr)
    return sys.stdin.read()


def read_jd_from_file(path: Path) -> str:
    if not path.exists():
        sys.exit(f"❌ File not found: {path}")
    return path.read_text(encoding="utf-8")


# ─── Pretty-printers (one per result schema) ───────────────
def render_fit_report(report: FitReport) -> str:
    lines = [
        f"FIT REPORT  →  Score: {report.overall_score}/100  →  Recommendation: {report.recommendation}",
        "─" * 70,
        f"Matched skills: {', '.join(report.matched_skills) or '(none)'}",
        f"Gap skills:     {', '.join(report.gap_skills) or '(none)'}",
        "",
        "Strengths:",
        *(f"  • {s}" for s in report.strengths),
        "",
        "Concerns:",
        *(f"  • {c}" for c in report.concerns),
        "",
        f"Reasoning: {report.reasoning}",
    ]
    return "\n".join(lines)


def render_cover_letter(letter: CoverLetter) -> str:
    body = "\n\n".join(letter.body_paragraphs)
    return (
        f"COVER LETTER  →  ~{letter.word_count} words  →  Tone: {letter.tone}\n"
        + "─" * 70 + "\n"
        f"{letter.opening_paragraph}\n\n"
        f"{body}\n\n"
        f"{letter.closing_paragraph}"
    )


def render_interview_prep(prep: InterviewPrep) -> str:
    lines = ["INTERVIEW PREP", "─" * 70, "", "TECHNICAL QUESTIONS:"]
    for i, qa in enumerate(prep.technical_questions, 1):
        lines.extend([
            f"\nQ{i}. {qa.question}",
            f"    → {qa.suggested_answer}",
            f"    (Reference: {qa.relevant_experience})",
        ])
    lines.extend(["", "BEHAVIORAL QUESTIONS:"])
    for i, qa in enumerate(prep.behavioral_questions, 1):
        lines.extend([
            f"\nQ{i}. {qa.question}",
            f"    → {qa.suggested_answer}",
            f"    (Reference: {qa.relevant_experience})",
        ])
    lines.extend(["", "QUESTIONS TO ASK THEM:"])
    lines.extend(f"  {i}. {q}" for i, q in enumerate(prep.questions_to_ask_them, 1))
    return "\n".join(lines)


def render_jd(jd: JobDescription) -> str:
    return (
        f"JOB DESCRIPTION  →  {jd.title} @ {jd.company_name}\n"
        + "─" * 70 + "\n"
        f"Seniority:        {jd.seniority_level}\n"
        f"Years required:   {jd.years_of_experience_required or 'unspecified'}\n"
        f"Location:         {jd.location or 'unspecified'}\n"
        f"Work arrangement: {jd.work_arrangement}\n\n"
        f"Required skills:  {', '.join(jd.required_skills) or '(none)'}\n"
        f"Nice to have:     {', '.join(jd.nice_to_have_skills) or '(none)'}\n\n"
        "Key responsibilities:\n"
        + "\n".join(f"  • {r}" for r in jd.key_responsibilities)
    )


def render_result(result) -> str:
    """Dispatch to the right pretty-printer based on result type."""
    match result:
        case FitReport():
            return render_fit_report(result)
        case CoverLetter():
            return render_cover_letter(result)
        case InterviewPrep():
            return render_interview_prep(result)
        case JobDescription():
            return render_jd(result)
        case str():  # placeholder for unimplemented intents
            return result
        case _:
            return repr(result)


# ─── Main ──────────────────────────────────────────────────
def main() -> None:
    arg_parser = argparse.ArgumentParser(
        description="JobFit — AI job application assistant.",
        epilog=(
            'Examples:\n'
            '  python main.py examples/elad_jd.txt\n'
            '  python main.py examples/elad_jd.txt --request "Should I apply?"\n'
            '  python main.py examples/elad_jd.txt --request "Cover letter please" --json'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arg_parser.add_argument(
        "file",
        nargs="?",
        type=Path,
        help="Path to a text file containing the job description.",
    )
    arg_parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read the job description from stdin instead of a file.",
    )
    arg_parser.add_argument(
        "--request",
        "-r",
        type=str,
        default=None,
        help=(
            "What you want JobFit to do (e.g. 'Should I apply?', "
            "'Write me a cover letter', 'Prep me for the interview'). "
            "Without this, JobFit just parses the JD into structured form (V1 mode)."
        ),
    )
    arg_parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured JSON instead of pretty-printed text.",
    )
    args = arg_parser.parse_args()

    # 1. Get the JD text
    if args.stdin:
        jd_text = read_jd_from_stdin()
    elif args.file is not None:
        jd_text = read_jd_from_file(args.file)
    else:
        arg_parser.print_help()
        sys.exit(1)

    if not jd_text.strip():
        sys.exit("❌ Empty job description.")

    # 2. Branch on whether a user request was provided
    if args.request:
        # V2 mode — route through the dispatcher
        print(f"\n⏳ Processing request: {args.request!r}\n", file=sys.stderr)
        classification, result = process_request(jd_text, args.request)

        # Routing summary always goes to stderr (status, not output)
        print(
            f"→ Intent: {classification.intent}  "
            f"(confidence: {classification.confidence:.2f})",
            file=sys.stderr,
        )
        print(f"  Reasoning: {classification.reasoning}\n", file=sys.stderr)
    else:
        # V1 mode — just parse the JD
        print("\n⏳ Parsing job description (V1 mode)...\n", file=sys.stderr)
        result = parse_job_description(jd_text)

    # 3. Print the result to stdout
    if args.json and not isinstance(result, str):
        print(result.model_dump_json(indent=2))
    else:
        print(render_result(result))

    # 4. Status footer to stderr
    print(
        "\n✓ Done — view the trace at https://smith.langchain.com (project: JobFit)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
