"""JobFit V1 CLI — parse a job description into structured JSON.

This is the user-facing entry point for the JD parser. It takes
input from a file or stdin, runs the LCEL chain defined in
assistant/chains/jd_parser.py, and prints the structured result.

Usage:
    # From a file
    python main.py examples/elad_jd.txt

    # From stdin (paste JD, then Ctrl+D / Ctrl+Z+Enter on Windows)
    python main.py --stdin

    # Pipe the structured output to a file or another tool
    python main.py examples/elad_jd.txt > parsed.json
    python main.py examples/elad_jd.txt | jq '.required_skills'
"""
import argparse
import sys
from pathlib import Path

from assistant.chains.jd_parser import parse_job_description


def read_jd_from_stdin() -> str:
    """Read a job description from stdin until EOF."""
    print(
        "Paste the job description below, then press "
        "Ctrl+D (Linux/Mac) or Ctrl+Z + Enter (Windows):",
        file=sys.stderr,
    )
    print("─" * 60, file=sys.stderr)
    return sys.stdin.read()


def read_jd_from_file(path: Path) -> str:
    """Read a job description from a UTF-8 text file."""
    if not path.exists():
        sys.exit(f"❌ File not found: {path}")
    return path.read_text(encoding="utf-8")


def main() -> None:
    arg_parser = argparse.ArgumentParser(
        description="Parse a job description into structured JSON via LangChain.",
        epilog="Example: python main.py examples/elad_jd.txt",
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
    args = arg_parser.parse_args()

    # 1. Get the job description text
    if args.stdin:
        jd_text = read_jd_from_stdin()
    elif args.file is not None:
        jd_text = read_jd_from_file(args.file)
    else:
        arg_parser.print_help()
        sys.exit(1)

    if not jd_text.strip():
        sys.exit("❌ Empty job description.")

    # 2. Run the chain (this calls OpenAI and produces a LangSmith trace)
    print("\n⏳ Parsing job description via LangChain...\n", file=sys.stderr)
    result = parse_job_description(jd_text)

    # 3. Print the structured result to stdout (so it's pipeable)
    print(result.model_dump_json(indent=2))

    # 4. Status footer to stderr (so it doesn't pollute pipes)
    print(
        "\n✓ Done — view the trace at https://smith.langchain.com (project: JobFit)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
