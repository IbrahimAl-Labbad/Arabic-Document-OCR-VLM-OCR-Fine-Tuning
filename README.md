# 🚀 Arabic Document OCR VLM - Multimodal RAG & OCR Fine-Tuning

## Project Overview
This project demonstrates an end-to-end pipeline for fine-tuning a Vision-Language Model (VLM) for complex Arabic document parsing. The project leverages **Gemma-3-4b-it** to extract structured JSON data from official Arabic documents, such as decrees, regulations, and official letters.

By heavily leveraging prompt engineering and strict JSON schemas, the pipeline instructs the VLM to parse text, tables, and charts accurately while preserving the original Arabic scripts, titles, honorifics, and date formats (such as Hijri vs. Gregorian calendars).

## Key Features
- **PDF to Image Preprocessing**: Converts raw official PDF documents into optimized grayscale images suitable for VLM inference and training using `pdf2image` and `PIL`.
- **Strict JSON Schema Parsing**: Uses `Pydantic` (in `schema.py`) to enforce a highly detailed, strict JSON structure for the extraction. This ensures that the generated outputs are robust, type-safe, and auto-validating.
- **Advanced Data Extraction**: 
  - Preserves Arabic text without transliteration.
  - Distinguishes between primary and additional referenced dates.
  - Correctly extracts charts (Bar, Line, Pie) into structured key-value objects instead of loose textual summaries.
  - Differentiates between physical "Seals" (الختم) and ink "Stamps" (الطابع).
- **Automated SFT Dataset Generation**: Uses `litellm` and cloud LLMs to process the document images and generate ground truth labels, building `ShareGPT` formatted datasets (`train-v1.json` and `val-v1.json`) for supervised fine-tuning.
- **Efficient Fine-Tuning with LLaMA-Factory**: Employs Low-Rank Adaptation (LoRA) via **LLaMA-Factory** to fine-tune the 4B parameter model efficiently, enabling training on consumer-grade GPUs without the need for expensive computing clusters.

## Repository Structure
- `ocr_finetune_vlm.ipynb`: The core Jupyter Notebook that walks through the entire process: environment setup, data preprocessing, baseline model evaluation, dataset formatting, and triggering the fine-tuning process.
- `schema.py`: Contains the Pydantic models that generate the JSON schema injected into the VLM prompt.
- `utils.py`: Utility functions for tasks such as PDF-to-image conversion, image preprocessing, and base64 encoding.
- `src/generate_dataset.py`: Script for generating the SFT dataset.
- `train.sh`: Shell script used to launch the fine-tuning process.

## Getting Started

### 1. Requirements & Installation
Ensure you have the required system dependencies (e.g., `poppler-utils` for PDF processing):
```bash
sudo apt-get install -y poppler-utils
```

Install the Python dependencies:
```bash
pip install pdf2image pillow tqdm json-repair litellm
```

### 2. Data Preparation
Place your raw PDF documents in the `./data/downloaded_pdfs/` directory. The pipeline will process these PDFs into images located in `./data/pdf_images/`.

### 3. Generate Fine-Tuning Dataset
Run the cells in `ocr_finetune_vlm.ipynb` (or the equivalent generation scripts) to:
1. Pass the processed images to an LLM via OpenRouter.
2. Extract the structured JSON containing markdown content, structural elements, official marks, attachments, etc.
3. Save the result into `ocr-image-sft.jsonl`.
4. Format the final output into the `ShareGPT` format expected by LLaMA-Factory (`train-v1.json`, `val-v1.json`).

### 4. Fine-Tune the Model
The training is executed using `LLaMA-Factory`. To set it up:
```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
git checkout 762b480131908d37736ad9aa3f12e87f8f7e6313
pip install -e .
pip install -r requirements/metrics.txt
```

Register your dataset in `LlamaFactory/data/dataset_info.json` and start the fine-tuning process:
```bash
llamafactory-cli train /workspace/LLaMA-Factory/examples/train_lora/ocr_finetune.yaml
```

## Critical Extraction Guidelines
When updating the prompts or fine-tuning parameters, keep the following project rules in mind:
- **Original Script**: Do not romanize, transliterate, or translate Arabic names and titles.
- **Numbers & Formatting**: Preserve all numbers, separators, and brackets exactly as they appear in the original document (e.g., "13/ت/8795" instead of "13-8795").
- **Chart Data**: Extract actual data values (labels and numerical values) into structured JSON arrays instead of writing summary sentences.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
