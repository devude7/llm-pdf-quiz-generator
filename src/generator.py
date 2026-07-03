import math
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
                page headers, footers, captions, or bibliography unless they explain a key concept.
                Focus on definitions, methods, architecture, results, comparisons, limitations and conclusions.
                - Do not add explanations.
                - Do not add any text before or after the quiz.
                - Do not include a line called Type.
                - Every question must contain a line starting exactly with: Correct answer:

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


def split_text(text, max_chars=5000):
    chunks = []
    current_chunk = ""

    for paragraph in text.split("\n"):
        if len(current_chunk) + len(paragraph) < max_chars:
            current_chunk += paragraph + "\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

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


def calculate_number_of_chunks(total_questions):
    if total_questions <= 5:
        return 2
    if total_questions <= 10:
        return 3
    if total_questions <= 20:
        return 4
    return 5


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

    chunks = split_text(pdf_text, max_chars=chunk_size)
    selected_chunks = select_evenly_spaced_chunks(chunks, calculate_number_of_chunks(total_questions))

    if len(selected_chunks) == 0:
        return "No valid text chunks found."

    questions_per_chunk = math.ceil(total_questions / len(selected_chunks))
    all_quizzes = []
    generated_questions = 0

    for index, chunk in enumerate(selected_chunks, start=1):
        remaining_questions = total_questions - generated_questions
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
        generated_questions += current_questions
        all_quizzes.append(quiz)

    return renumber_questions("\n\n".join(all_quizzes))
