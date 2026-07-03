from pdf2image import convert_from_path
from pypdf import PdfReader
import pytesseract


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(str(pdf_path))
    full_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            full_text.append(f"\n\n--- Page {page_number} ---\n{text}")

    return "".join(full_text)


def extract_text_from_pdf_ocr(pdf_path, lang="eng", dpi=200, max_pages=None):
    images = convert_from_path(str(pdf_path), dpi=dpi)
    if max_pages is not None:
        images = images[:max_pages]

    full_text = []

    for page_number, image in enumerate(images, start=1):
        print(f"OCR processing page {page_number}/{len(images)}...")
        text = pytesseract.image_to_string(image, lang=lang)
        if text:
            full_text.append(f"\n\n--- OCR Page {page_number} ---\n{text}")

    return "".join(full_text)


def extract_text_from_pdf_auto(pdf_path, min_text_length=300, ocr_lang="eng", ocr_dpi=200, ocr_max_pages=None, disable_ocr=False):
    text = extract_text_from_pdf(pdf_path)

    if len(text.strip()) >= min_text_length:
        print("Text extracted using pypdf.")
        return text

    if disable_ocr:
        print("Very little text extracted with pypdf. OCR is disabled.")
        return text

    print("Very little text extracted with pypdf. Switching to OCR...")
    return extract_text_from_pdf_ocr(pdf_path, lang=ocr_lang, dpi=ocr_dpi, max_pages=ocr_max_pages)
