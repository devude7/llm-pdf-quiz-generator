from pathlib import Path
import tomllib

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_OUTPUT_BASENAME = "generated_quiz"


class ModelConfig:
    def __init__(self, name=DEFAULT_MODEL_NAME, quantized=True):
        self.name = name
        self.quantized = quantized


class GenerationConfig:
    def __init__(self, max_new_tokens=2500, chunk_size=5000):
        self.max_new_tokens = max_new_tokens
        self.chunk_size = chunk_size


class OcrConfig:
    def __init__(self, enabled=True, language="eng", dpi=200, max_pages=None, min_text_length=300):
        self.enabled = enabled
        self.language = language
        self.dpi = dpi
        self.max_pages = max_pages
        self.min_text_length = min_text_length


class OutputConfig:
    def __init__(self, directory=DEFAULT_OUTPUT_DIR, basename=DEFAULT_OUTPUT_BASENAME):
        self.directory = directory
        self.basename = basename


class AppConfig:
    def __init__(self, model=None, generation=None, ocr=None, output=None):
        self.model = model or ModelConfig()
        self.generation = generation or GenerationConfig()
        self.ocr = ocr or OcrConfig()
        self.output = output or OutputConfig()


def _section(data, name):
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section [{name}] must be a table.")
    return value


def _optional_positive_int(value):
    if value is None or value == 0:
        return None
    return int(value)


def load_config(path="config.toml"):
    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()

    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    model = _section(data, "model")
    generation = _section(data, "generation")
    ocr = _section(data, "ocr")
    output = _section(data, "output")

    return AppConfig(
        model=ModelConfig(
            name=str(model.get("name", DEFAULT_MODEL_NAME)),
            quantized=bool(model.get("quantized", True)),
        ),
        generation=GenerationConfig(
            max_new_tokens=int(generation.get("max_new_tokens", 2500)),
            chunk_size=int(generation.get("chunk_size", 5000)),
        ),
        ocr=OcrConfig(
            enabled=bool(ocr.get("enabled", True)),
            language=str(ocr.get("language", "eng")),
            dpi=int(ocr.get("dpi", 200)),
            max_pages=_optional_positive_int(ocr.get("max_pages")),
            min_text_length=int(ocr.get("min_text_length", 300)),
        ),
        output=OutputConfig(
            directory=str(output.get("directory", DEFAULT_OUTPUT_DIR)),
            basename=str(output.get("basename", DEFAULT_OUTPUT_BASENAME)),
        ),
    )
