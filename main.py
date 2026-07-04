import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.config import load_config
from src.export import save_quiz_as_json, save_quiz_as_pdf
from src.generator import QuizGenerator, generate_quiz_from_pdf


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate JSON and PDF quizzes from a source PDF.",
    )
    parser.add_argument("pdf", help="Path to the source PDF file.")
    parser.add_argument("-n", "--questions", type=int, default=10, help="Number of questions to generate.")
    parser.add_argument(
        "-t",
        "--type",
        choices=["single-choice", "multiple-choice", "open"],
        default="single-choice",
        help="Question type.",
    )
    parser.add_argument("-l", "--language", default="English", help="Quiz language, for example English or Polish.")
    parser.add_argument("-o", "--output-dir", default=None, help="Directory for generated files.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    load_dotenv(".env")
    config = load_config("config.toml")

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        parser.error(f"PDF file does not exist: {pdf_path}")

    output_dir = Path(args.output_dir or config.output.directory).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_token = os.getenv("HF_TOKEN") or None

    print(f"Loading model: {config.model.name}")
    generator = QuizGenerator(
        model_name=config.model.name,
        hf_token=hf_token,
        quantized=config.model.quantized,
        max_new_tokens=config.generation.max_new_tokens,
    )

    quiz = generate_quiz_from_pdf(
        pdf_path=str(pdf_path),
        generator=generator,
        total_questions=args.questions,
        question_type=args.type,
        language=args.language,
        chunk_size=config.generation.chunk_size,
        chunk_overlap=config.generation.chunk_overlap,
        min_text_length=config.ocr.min_text_length,
        ocr_lang=config.ocr.language,
        ocr_dpi=config.ocr.dpi,
        ocr_max_pages=config.ocr.max_pages,
        disable_ocr=not config.ocr.enabled,
    )

    basename = config.output.basename
    json_path = output_dir / f"{basename}.json"
    pdf_output_path = output_dir / f"{basename}.pdf"

    quiz_data = save_quiz_as_json(
        quiz_text=quiz,
        filename=json_path,
        question_type=args.type,
        language=args.language,
        source_pdf=pdf_path,
    )
    save_quiz_as_pdf(quiz_data=quiz_data, filename=pdf_output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
