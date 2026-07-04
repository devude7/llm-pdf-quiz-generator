import math
import random
import re
from importlib.util import find_spec

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .extraction import extract_text_from_pdf_auto


def get_question_type_instructions(question_type):
    if question_type == "single-choice":
        return """
            Create only single-choice questions.

            Rules:
            - Every question must have exactly four options: A, B, C, D.
            - Exactly one option must be correct.
            - The correct answer line must contain only one letter: A, B, C, or D.
            - Correct format: Correct answer: B
            - Do not generate multiple-choice questions.
            - Do not generate open questions.

            Required format:

            Question 1:
            Question: ...
            A. ...
            B. ...
            C. ...
            D. ...
            Correct answer: B
            """

    if question_type == "multiple-choice":
        return """
            Create only multiple-choice questions.

            Rules:
            - Every question must have exactly four options: A, B, C, D.
            - More than one option may be correct.
            - The correct answer line must contain only letters separated by commas.
            - Correct format: Correct answer: A, C
            - Do not generate single-choice questions.
            - Do not generate open questions.

            Required format:

            Question 1:
            Question: ...
            A. ...
            B. ...
            C. ...
            D. ...
            Correct answer: A, C
            """

    if question_type == "open":
        return """
            Create only open questions.

            Rules:
            - Every question must be short and clear.
            - The correct answer should be short and factual.
            - Do not generate A, B, C, D options.
            - Do not generate single-choice questions.
            - Do not generate multiple-choice questions.

            Required format:

            Question 1:
            Question: ...
            Correct answer: ...
            """

    return get_question_type_instructions("single-choice")


class QuizGenerator:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", hf_token=None, quantized=True, max_new_tokens=2500):
        self.max_new_tokens = max_new_tokens

        model_kwargs = {"device_map": "auto"}
        if hf_token:
            model_kwargs["token"] = hf_token

        can_quantize = quantized and torch.cuda.is_available() and find_spec("bitsandbytes") is not None
        if can_quantize:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            if quantized and torch.cuda.is_available():
                print("bitsandbytes is not installed. Loading the model without 4-bit quantization.")
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model_kwargs["torch_dtype"] = dtype

        tokenizer_kwargs = {}
        if hf_token:
            tokenizer_kwargs["token"] = hf_token

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Model device: {self.model.device}")

    def generate_quiz(self, notes, number_of_questions=5, question_type="single-choice", language="English"):
        type_instructions = get_question_type_instructions(question_type)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an educational quiz generator. "
                    "Create clear and factual quizzes based only on the provided notes. "
                    "Follow the required output format exactly."
                ),
            },
            {
                "role": "user",
                "content": f"""
                Based only on the notes below, create a learning quiz.

                General requirements:
                - Create exactly {number_of_questions} questions.
                - Quiz language: {language}.
                - Use only information from the notes.
                - Do not invent facts that are not present in the notes.
                - Do not create questions about incidental example sentences, references, author affiliations,
                page headers, footers, captions, bibliography, figure numbers, table numbers, row labels,
                visual layout details, or diagram labels.
                - Do not ask what is shown in Figure N/Table N, which figure/table contains something,
                or which row/label is highlighted in a visualization.
                - If a figure or table explains an important concept, ask about the underlying concept,
                method, result, or conclusion instead of the figure/table identifier.
                - Avoid questions that require exact mathematical notation or formula transcription when
                the notes contain symbols, superscripts, subscripts, or formatting that may be ambiguous.
                Focus on definitions, methods, architecture, results, comparisons, limitations and conclusions.
                - Do not add explanations.
                - Do not add any text before or after the quiz.
                - Do not include a line called Type.
                - Every question must contain a line starting exactly with: Correct answer:
                - Present questions in a mixed order instead of following the notes order exactly.

                {type_instructions}

                Notes:
                {notes}
                """,
            },
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return result.strip()


def normalize_pdf_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    paragraphs = []
    current_lines = []

    def flush_current_lines():
        if current_lines:
            paragraphs.append(" ".join(current_lines).strip())
            current_lines.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush_current_lines()
            continue

        if re.match(r"^---\s*(?:OCR\s+)?Page\s+\d+\s*---$", line, flags=re.IGNORECASE):
            flush_current_lines()
            paragraphs.append(line)
            continue

        current_lines.append(line)

    flush_current_lines()
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph).strip()


def is_visual_caption_block(text):
    caption_patterns = [
        r"^\s*(?:\d+\s+)?(?:figure|fig\.)\s+\d+\s*[:.]",
        r"^\s*(?:\d+\s+)?(?:figure|fig\.)\s+\d+\b",
    ]
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in caption_patterns):
        return True

    figure_mentions = len(re.findall(r"\b(?:figure|fig\.)\s+\d+\b", text, flags=re.IGNORECASE))
    word_count = len(re.findall(r"\w+", text))
    return figure_mentions > 0 and word_count <= 35


def split_into_sentences(text):
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if sentence.strip()]


def split_long_text(text, max_chars):
    if len(text) <= max_chars:
        return [text]

    blocks = []
    current = ""
    for sentence in split_into_sentences(text):
        if len(sentence) > max_chars:
            if current.strip():
                blocks.append(current.strip())
                current = ""
            for start in range(0, len(sentence), max_chars):
                blocks.append(sentence[start : start + max_chars].strip())
            continue

        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            if current.strip():
                blocks.append(current.strip())
            current = sentence

    if current.strip():
        blocks.append(current.strip())

    return blocks


def split_into_blocks(text, max_block_chars=1200):
    normalized_text = normalize_pdf_text(text)
    blocks = []

    for paragraph in re.split(r"\n\s*\n", normalized_text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if is_visual_caption_block(paragraph):
            continue
        blocks.extend(split_long_text(paragraph, max_block_chars))

    return blocks


def get_overlap_text(text, max_chars):
    if max_chars <= 0 or len(text) <= max_chars:
        return text.strip()

    overlap = text[-max_chars:]
    sentence_start = re.search(r"(?<=[.!?])\s+", overlap)
    if sentence_start:
        overlap = overlap[sentence_start.end() :]

    return overlap.strip()


def split_text(text, max_chars=5000, overlap_chars=700):
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0.")
    if overlap_chars < 0:
        raise ValueError("overlap_chars cannot be negative.")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars.")

    blocks = split_into_blocks(text)
    chunks = []
    current_blocks = []
    current_length = 0

    for block in blocks:
        added_length = len(block) + (2 if current_blocks else 0)
        if current_blocks and current_length + added_length > max_chars:
            chunk = "\n\n".join(current_blocks).strip()
            chunks.append(chunk)

            overlap_limit = max(0, min(overlap_chars, max_chars - len(block) - 2))
            overlap = get_overlap_text(chunk, overlap_limit)
            current_blocks = [overlap] if overlap else []
            current_length = len(overlap) if overlap else 0

        if len(block) > max_chars:
            chunks.extend(split_long_text(block, max_chars))
            current_blocks = []
            current_length = 0
        elif current_blocks:
            current_blocks.append(block)
            current_length += len(block) + 2
        else:
            current_blocks = [block]
            current_length = len(block)

    if current_blocks:
        chunks.append("\n\n".join(current_blocks).strip())

    return chunks


def renumber_questions(text):
    question_number = 1
    new_lines = []

    for line in text.splitlines():
        if re.match(r"^\s*Question\s+\d+\s*:", line):
            new_lines.append(f"Question {question_number}:")
            question_number += 1
        else:
            new_lines.append(line)

    return "\n".join(new_lines).strip()


def split_question_blocks(text):
    return [
        block.strip()
        for block in re.split(r"\n(?=\s*Question\s+\d+\s*:)", text.strip())
        if block.strip()
    ]


def is_low_quality_question_block(block):
    question_match = re.search(
        r"Question:\s*(.*?)(?=\n\s*(?:A\.|B\.|C\.|D\.|Correct answer:)|\Z)",
        block,
        flags=re.DOTALL,
    )
    question_text = question_match.group(1) if question_match else block
    question_text = re.sub(r"\s+", " ", question_text).strip()

    blocked_patterns = [
        r"\b(?:figure|fig\.)\s*\d+\b",
        r"\btable\s*\d+\b",
        r"\bfigures?\b",
        r"\btables?\b",
        r"\brow\s*\([A-Z]\)",
        r"\bwhich\s+(?:figure|fig\.|table|row)\b",
        r"\bshown\s+in\s+(?:figure|fig\.|table)\b",
        r"\billustrated\s+in\s+(?:figure|fig\.|table|figures)\b",
        r"\baccording\s+to\s+(?:figure|fig\.|table)\b",
        r"\bhighlighted\s+in\s+(?:figure|fig\.|table|a visualization)\b",
        r"\bvisuali[sz]ation\b",
    ]
    return any(re.search(pattern, question_text, flags=re.IGNORECASE) for pattern in blocked_patterns)


def add_accepted_question_blocks(quiz, accepted_question_blocks, total_questions):
    for block in split_question_blocks(quiz):
        if len(accepted_question_blocks) >= total_questions:
            break
        if is_low_quality_question_block(block):
            continue
        accepted_question_blocks.append(block)


def calculate_number_of_chunks(total_questions, available_chunks=None, max_selected_chunks=8):
    if total_questions <= 5:
        desired_chunks = 2
    else:
        desired_chunks = math.ceil(total_questions / 3)

    desired_chunks = min(desired_chunks, max_selected_chunks)
    if available_chunks is None:
        return desired_chunks

    return min(available_chunks, desired_chunks)


def select_evenly_spaced_chunks(chunks, number_of_chunks):
    if len(chunks) <= number_of_chunks:
        return chunks

    if number_of_chunks == 1:
        return [chunks[len(chunks) // 2]]

    selected_chunks = []
    for i in range(number_of_chunks):
        index = round(i * (len(chunks) - 1) / (number_of_chunks - 1))
        selected_chunks.append(chunks[index])

    return selected_chunks


def generate_quiz_from_pdf(
    pdf_path,
    generator,
    total_questions=10,
    question_type="single-choice",
    language="English",
    chunk_size=5000,
    chunk_overlap=700,
    min_text_length=300,
    ocr_lang="eng",
    ocr_dpi=200,
    ocr_max_pages=None,
    disable_ocr=False,
):
    pdf_text = extract_text_from_pdf_auto(
        pdf_path,
        min_text_length=min_text_length,
        ocr_lang=ocr_lang,
        ocr_dpi=ocr_dpi,
        ocr_max_pages=ocr_max_pages,
        disable_ocr=disable_ocr,
    )

    if len(pdf_text.strip()) == 0:
        return "No text could be extracted from the PDF file."

    chunks = split_text(pdf_text, max_chars=chunk_size, overlap_chars=chunk_overlap)
    selected_chunks = select_evenly_spaced_chunks(
        chunks,
        calculate_number_of_chunks(total_questions, available_chunks=len(chunks)),
    )
    random.shuffle(selected_chunks)

    if len(selected_chunks) == 0:
        return "No valid text chunks found."

    accepted_question_blocks = []
    questions_per_chunk = math.ceil(total_questions / len(selected_chunks))

    for index, chunk in enumerate(selected_chunks, start=1):
        remaining_questions = total_questions - len(accepted_question_blocks)
        if remaining_questions <= 0:
            break

        current_questions = min(questions_per_chunk, remaining_questions)
        print(f"Generating {current_questions} question(s) from chunk {index}/{len(selected_chunks)}...")
        quiz = generator.generate_quiz(
            notes=chunk,
            number_of_questions=current_questions,
            question_type=question_type,
            language=language,
        )
        add_accepted_question_blocks(quiz, accepted_question_blocks, total_questions)

    if len(accepted_question_blocks) != total_questions:
        raise RuntimeError(
            f"Could not generate exactly {total_questions} valid question(s). "
            f"Accepted {len(accepted_question_blocks)} after filtering."
        )

    return renumber_questions("\n\n".join(accepted_question_blocks))
