from datetime import datetime
import glob
import json
import os
from pathlib import Path
import re
from xml.sax.saxutils import escape

from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def parse_quiz_text_to_json(quiz_text, question_type):
    blocks = re.split(r"\n(?=\s*Question\s+\d+\s*:)", quiz_text.strip())
    questions = []

    for block in blocks:
        number_match = re.search(r"Question\s+(\d+)\s*:", block)
        question_match = re.search(
            r"Question:\s*(.*?)(?=\n\s*(?:A\.|B\.|C\.|D\.|Correct answer:))",
            block,
            flags=re.DOTALL,
        )
        correct_match = re.search(r"Correct answer:\s*(.+)", block)

        if not number_match or not question_match or not correct_match:
            continue

        question_data = {
            "number": int(number_match.group(1)),
            "type": question_type,
            "question": normalize_text(question_match.group(1)),
            "correct_answer": normalize_text(correct_match.group(1)),
        }

        if question_type in {"single-choice", "multiple-choice"}:
            options = {}
            for letter in ["A", "B", "C", "D"]:
                option_match = re.search(
                    rf"^\s*{letter}\.\s*(.*?)(?=\n\s*[A-D]\.|\n\s*Correct answer:|\Z)",
                    block,
                    flags=re.MULTILINE | re.DOTALL,
                )
                if option_match:
                    options[letter] = normalize_text(option_match.group(1))
            question_data["options"] = options

        questions.append(question_data)

    return questions


def save_quiz_as_json(quiz_text, filename, question_type="single-choice", language="English", source_pdf=None):
    quiz_data = {
        "metadata": {
            "source_pdf": str(source_pdf) if source_pdf is not None else None,
            "language": language,
            "question_type": question_type,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        "questions": parse_quiz_text_to_json(quiz_text, question_type),
    }

    Path(filename).write_text(
        json.dumps(quiz_data, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )
    return quiz_data


def find_font_file(preferred_names):
    font_dirs = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        "C:/Windows/Fonts",
    ]
    all_fonts = []

    for font_dir in font_dirs:
        if os.path.exists(font_dir):
            all_fonts.extend(glob.glob(font_dir + "/**/*.ttf", recursive=True))

    for preferred_name in preferred_names:
        for font_path in all_fonts:
            if os.path.basename(font_path).lower() == preferred_name.lower():
                return font_path

    return None


def prepare_text_for_pdf(text, max_word_length=60):
    value = "" if text is None else str(text)
    value = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", value)

    def split_long_word(match):
        word = match.group(0)
        return " ".join(word[i : i + max_word_length] for i in range(0, len(word), max_word_length))

    value = re.sub(r"\S{60,}", split_long_word, value)
    value = escape(value)
    return value.replace("\n", "<br/>")


def register_pdf_fonts():
    font_regular = find_font_file(["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "arial.ttf"])
    font_bold = find_font_file(["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "arialbd.ttf"])

    if font_regular is None:
        return "Helvetica", "Helvetica-Bold"

    if font_bold is None:
        font_bold = font_regular

    pdfmetrics.registerFont(TTFont("QuizFont", font_regular))
    pdfmetrics.registerFont(TTFont("QuizFont-Bold", font_bold))
    return "QuizFont", "QuizFont-Bold"


def save_quiz_as_pdf(quiz_data, filename):
    regular_font, bold_font = register_pdf_fonts()

    doc = SimpleDocTemplate(
        str(filename),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    title_style = ParagraphStyle(
        name="Title",
        fontName=bold_font,
        fontSize=18,
        leading=22,
        spaceAfter=14,
        alignment=TA_LEFT,
    )
    heading_style = ParagraphStyle(
        name="Heading",
        fontName=bold_font,
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=6,
    )
    normal_style = ParagraphStyle(
        name="Normal",
        fontName=regular_font,
        fontSize=10.5,
        leading=14,
        spaceAfter=6,
    )
    question_style = ParagraphStyle(
        name="Question",
        fontName=bold_font,
        fontSize=11.5,
        leading=15,
        spaceBefore=8,
        spaceAfter=4,
    )
    answer_style = ParagraphStyle(
        name="Answer",
        fontName=bold_font,
        fontSize=10.5,
        leading=14,
        spaceAfter=10,
    )

    story = [Paragraph("Generated Quiz", title_style)]
    metadata = quiz_data.get("metadata", {})

    story.append(Paragraph(f"<b>Source PDF:</b> {prepare_text_for_pdf(metadata.get('source_pdf', '-'))}", normal_style))
    story.append(Paragraph(f"<b>Language:</b> {prepare_text_for_pdf(metadata.get('language', '-'))}", normal_style))
    story.append(
        Paragraph(
            f"<b>Question type:</b> {prepare_text_for_pdf(metadata.get('question_type', '-'))}",
            normal_style,
        )
    )
    story.append(Paragraph(f"<b>Created at:</b> {prepare_text_for_pdf(metadata.get('created_at', '-'))}", normal_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Quiz questions", heading_style))

    for question in quiz_data.get("questions", []):
        story.append(Paragraph(f"Question {question.get('number', '')}:", question_style))
        story.append(Paragraph(prepare_text_for_pdf(question.get("question", "")), normal_style))

        options = question.get("options", {})
        for letter in ["A", "B", "C", "D"]:
            if letter in options:
                story.append(Paragraph(f"<b>{letter}.</b> {prepare_text_for_pdf(options[letter])}", normal_style))

        correct_answer = prepare_text_for_pdf(question.get("correct_answer", "-"))
        story.append(Paragraph(f"<b>Correct answer:</b> {correct_answer}", answer_style))

    doc.build(story)
