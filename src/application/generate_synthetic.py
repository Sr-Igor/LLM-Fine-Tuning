"""
Synthetic Data Generator for Planuze LLM Fine-Tuning.

This module generates training data by:
1. Reading PDFs from source_documents
2. Chunking the content
3. Using Ollama to generate Q&A pairs in the exact production format
4. Saving to JSONL in Alpaca format for training
"""

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv
from pypdf import PdfReader
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Load environment variables
load_dotenv("envs/.env.global")

console = Console()


class SyntheticDataGenerator:
    """Generates synthetic training data from PDFs using Ollama."""

    def __init__(self):
        """Initialize the generator with environment configuration."""
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = os.getenv("SYNTHETIC_GENERATOR_MODEL", "qwen2.5:14b")
        self.chunk_size = int(os.getenv("SYNTHETIC_CHUNK_SIZE", "3500"))
        self.overlap = int(os.getenv("SYNTHETIC_OVERLAP", "500"))
        self.source_dir = Path(
            os.getenv("SYNTHETIC_SOURCE_DIR", "data/source_documents"))
        self.output_file = Path(
            os.getenv("SYNTHETIC_OUTPUT_FILE", "data/raw/train_data_synthetic.jsonl"))

        # Load prompts from env
        self.prompt_ask = os.getenv("AI_PROMPT_ASK_INSTRUCTIONS", "")
        self.prompt_action = os.getenv("AI_PROMPT_ACTION_INSTRUCTIONS", "")

        # Load XML Tags from env
        self.tag_subject = os.getenv("AI_CHAT_SUBJECT", "subject")
        self.tag_context = os.getenv("AI_CHAT_CONTEXT", "context")
        self.tag_question = os.getenv("AI_CHAT_QUESTION", "question")
        self.tag_history = os.getenv("AI_CHAT_HISTORY", "history")
        self.tag_language = os.getenv("AI_CHAT_LANGUAGE", "language")
        self.tag_mode = os.getenv("AI_CHAT_MODE", "mode")
        self.tag_sys_req = os.getenv(
            "AI_CHAT_SYSTEM_REQUIRED", "system_action_required")
        self.tag_sys_instr = os.getenv(
            "AI_CHAT_SYSTEM_INSTRUCTIONS", "system_instructions")

        # Ensure output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text content from a PDF file."""
        try:
            reader = PdfReader(str(pdf_path))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            console.print(f"[red]Error reading {pdf_path.name}: {e}[/red]")
            return ""

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < text_len:
                last_period = chunk.rfind(". ")
                if last_period > self.chunk_size * 0.7:  # At least 70% of chunk
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1

            chunks.append(chunk.strip())
            start = end - self.overlap

        return chunks

    def call_ollama(self, prompt: str) -> str:
        """Call Ollama API to generate content."""
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.8,
                        "top_p": 0.9,
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            console.print(f"[red]Ollama API error: {e}[/red]")
            return ""

    def generate_ask_example(self, context_chunk: str, language: str = "pt") -> Dict[str, Any]:
        """Generate an ASK mode training example."""
        generation_prompt = f"""You are creating training data for an AI assistant.

Given this context from a document:
---
{context_chunk}
---

Generate a realistic user question that can be answered using ONLY the information in this context.
Then provide the answer.

Requirements:
- Question must be in {language} language
- Answer must be concise and based ONLY on the context
- Use natural, conversational language
- Vary question types (what, how, when, why, who)

Return ONLY a JSON object with this exact structure:
{{
  "question": "the user question here",
  "answer": "the assistant answer here"
}}

JSON:"""

        response = self.call_ollama(generation_prompt)

        try:
            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                qa_data = json.loads(response[json_start:json_end])

                # Build full prompt in production format
                full_prompt = f"""<{self.tag_sys_instr}>
{self.prompt_ask}
</{self.tag_sys_instr}>

<{self.tag_mode}>ASK</{self.tag_mode}>

<{self.tag_subject}>Company</{self.tag_subject}>

<{self.tag_context}>
{{
  "userName": "User",
  "content": [
    {{
      "source": "documents",
      "data": [
        {json.dumps(context_chunk[:500])}
      ]
    }}
  ],
  "files": {{
    "files": [
      {json.dumps(context_chunk)}
    ]
  }}
}}
</{self.tag_context}>

<{self.tag_history}>
</{self.tag_history}>

<{self.tag_question}>
{qa_data['question']}
</{self.tag_question}>

<{self.tag_language}>{language}</{self.tag_language}>"""

                return {
                    "instruction": full_prompt,
                    "input": "",
                    "output": qa_data['answer']
                }
        except Exception as e:
            console.print(f"[yellow]Failed to parse Q&A: {e}[/yellow]")

        return None

    def generate_action_example(self, language: str = "pt") -> Dict[str, Any]:
        """Generate an ACTION mode training example with dynamic scenarios."""

        # Scenario 1: Company Update
        company_ctx = {
            "@id": "cml_" + "".join(random.choices("abcdef0123456789", k=10)),
            "name": random.choice(["Planuze", "DevCorp"]),
            "email": "contact@planuze.com"
        }

        scenarios = [
            {
                "id": "company_update",
                "source": "company",
                "subject": "Company",
                "context": company_ctx,
                "tool_def": """- company.update: {
    "p":{"id":"string"},
    "q": {"action":"string"},
    "b":{
        "name":"string (optional)",
        "email":"email (optional)"
    }
}""",
                "action_req": (
                    "Altere o nome da empresa para NovoNome"
                    if language == "pt"
                    else "Change company name to NewName"
                ),
                "expected": {
                    "action_id": None,
                    "subject": "company",
                    "action": "update",
                    "message": (
                        f"Vou alterar o nome de **{company_ctx['name']}** para **NovoNome**."
                        if language == "pt"
                        else f"Changing name from **{company_ctx['name']}** to **NewName**."
                    ),
                    "payload": {
                        "p": {"id": company_ctx["@id"]},
                        "q": {},
                        "b": {"name": "NovoNome" if language == "pt" else "NewName"}
                    }
                }
            },
            {
                "id": "user_update",
                "source": "user_profile",
                "subject": "User",
                "context": {
                    "@id": "usr_" + "".join(random.choices("abcdef0123456789", k=10)),
                    "role": "Analista",
                    "department": "Finance"
                },
                "tool_def": """- user.update: {
    "p":{"id":"string"},
    "q": {},
    "b":{
        "role":"string (optional)",
        "department":"string (optional)"
    }
}""",
                "action_req": (
                    "Mude meu cargo para Gerente"
                    if language == "pt"
                    else "Change my role to Manager"
                ),
                "expected": {
                    "action_id": None,
                    "subject": "user",
                    "action": "update",
                    "message": (
                        "Alterando cargo de **Analista** para **Gerente**."
                        if language == "pt"
                        else "Changing role from **Analista** to **Manager**."
                    ),
                    "payload": {
                        "p": {"id": "usr_..."},  # Will be fixed below
                        "q": {},
                        "b": {"role": "Gerente" if language == "pt" else "Manager"}
                    }
                }
            }
        ]

        # Fix ID reference in second scenario
        scenarios[1]["expected"]["payload"]["p"]["id"] = scenarios[1]["context"]["@id"]

        scenario = random.choice(scenarios)

        full_prompt = f"""<{self.tag_sys_req}>
{self.prompt_action}

# AVAILABLE TOOLS

{scenario['tool_def']}
    
⚠️ IMPORTANT: You MUST response with the JSON format above if the user asks for these actions.

</{self.tag_sys_req}>

<{self.tag_mode}>ACTION</{self.tag_mode}>

<{self.tag_subject}>{scenario['subject']}</{self.tag_subject}>

<{self.tag_context}>
{{
  "userName": "User",
  "content": [
    {{
      "source": "{scenario['source']}",
      "data": [
        {json.dumps(json.dumps(scenario['context']))}
      ]
    }}
  ],
  "files": {{
    "files": []
  }}
}}
</{self.tag_context}>

<{self.tag_history}>
User: Olá
Planus: Olá! Como posso ajudar você hoje?
</{self.tag_history}>

<{self.tag_question}>
{scenario['action_req']}
</{self.tag_question}>

<{self.tag_language}>{language}</{self.tag_language}>"""

        return {
            "instruction": full_prompt,
            "input": "",
            "output": json.dumps(scenario['expected'], ensure_ascii=False)
        }

    def _process_pdf_file(self, pdf_file: Path) -> List[Dict[str, Any]]:
        """Process a single PDF file and generate examples."""
        text = self.extract_text_from_pdf(pdf_file)
        if not text:
            return []

        chunks = self.chunk_text(text)
        console.print(f"  → {len(chunks)} chunks from {pdf_file.name}")

        examples = []
        # Generate 2-3 examples per chunk (randomly)
        for chunk in chunks[:10]:  # Limit to first 10 chunks per PDF
            num_examples = random.randint(2, 3)
            for _ in range(num_examples):
                lang = random.choice(["pt", "en"])
                example = self.generate_ask_example(chunk, lang)
                if example:
                    examples.append(example)
        return examples

    def _get_pdf_checkpoint_path(self, pdf_file: Path, partial_dir: Path) -> Path:
        """Generate checkpoint path for a PDF file."""
        file_hash = hashlib.md5(pdf_file.name.encode()).hexdigest()[:8]
        return partial_dir / f"synthetic_{pdf_file.stem}_{file_hash}.jsonl"

    def _load_checkpoint(self, checkpoint_file: Path) -> List[Dict[str, Any]] | None:
        """Load examples from a checkpoint file if it exists."""
        if not checkpoint_file.exists():
            return None
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        except Exception:
            return []

    def _save_checkpoint(self, examples: List[Dict[str, Any]], checkpoint_file: Path) -> None:
        """Save examples to a checkpoint file."""
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    def _process_pdf_with_checkpoint(
        self,
        pdf_file: Path,
        partial_dir: Path,
        progress: Progress,
        task_id: Any
    ) -> List[Dict[str, Any]]:
        """Process a single PDF with checkpointing logic."""
        checkpoint_file = self._get_pdf_checkpoint_path(pdf_file, partial_dir)

        # Try loading existing
        existing = self._load_checkpoint(checkpoint_file)
        if existing is not None:
            progress.console.print(
                f"[dim]  Skipping {pdf_file.name} (already processed)[/dim]")
            return existing

        progress.update(
            task_id, description=f"[cyan]Reading {pdf_file.name}...")

        # Generate
        examples = self._process_pdf_file(pdf_file)

        if examples:
            self._save_checkpoint(examples, checkpoint_file)
            progress.console.print(
                f"  [green]✓ Saved {len(examples)} examples to parts/{checkpoint_file.name}[/green]"
            )

        return examples or []

    def _generate_actions_with_checkpoint(
        self,
        partial_dir: Path,
        num_action_examples: int,
        progress: Progress
    ) -> List[Dict[str, Any]]:
        """Generate action examples with checkpointing."""
        action_file = partial_dir / "synthetic_actions.jsonl"

        # Check if actions are already generated
        existing = self._load_checkpoint(action_file)
        if existing is not None:
            progress.console.print(
                "[dim]  Skipping Actions (already processed)[/dim]")
            return existing

        task = progress.add_task(
            f"[cyan]Generating {num_action_examples} ACTION examples...",
            total=num_action_examples
        )

        action_examples = []
        for _ in range(num_action_examples):
            lang = random.choice(["pt", "en"])
            example = self.generate_action_example(lang)
            if example:
                action_examples.append(example)
            progress.advance(task)

        # Save Action Checkpoint
        if action_examples:
            self._save_checkpoint(action_examples, action_file)
            progress.console.print(
                f"  [green]✓ Saved {len(action_examples)} ACTION examples[/green]"
            )

        return action_examples

    def generate_dataset(self, num_action_examples: int = 50) -> List[Dict[str, Any]]:
        """Generate the complete synthetic dataset with checkpoints."""
        final_dataset = []

        # Directory for partial saves (checkpoints)
        partial_dir = self.output_file.parent / "synthetic_parts"
        partial_dir.mkdir(parents=True, exist_ok=True)

        # Get all PDFs
        pdf_files = list(self.source_dir.glob("*.pdf"))

        if not pdf_files:
            console.print("[red]No PDF files found in source_documents![/red]")
            return final_dataset

        console.print(f"[green]Found {len(pdf_files)} PDF files[/green]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:

            # 1. Process PDFs for ASK examples
            task = progress.add_task(
                "[cyan]Processing PDFs...", total=len(pdf_files))

            for pdf_file in pdf_files:
                examples = self._process_pdf_with_checkpoint(
                    pdf_file, partial_dir, progress, task
                )
                final_dataset.extend(examples)
                progress.advance(task)

            # 2. Generate ACTION examples
            action_examples = self._generate_actions_with_checkpoint(
                partial_dir, num_action_examples, progress
            )
            final_dataset.extend(action_examples)

        return final_dataset

    def save_dataset(self, dataset: List[Dict[str, Any]]):
        """Save dataset to JSONL file."""
        with open(self.output_file, "w", encoding="utf-8") as f:
            for example in dataset:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        console.print(
            f"\n[green]✅ Saved {len(dataset)} examples to {self.output_file}[/green]")

    def run(self, num_action_examples: int = 50):
        """Run the complete generation pipeline."""
        console.print(
            "[bold cyan]🚀 Planuze Synthetic Data Generator[/bold cyan]\n")
        console.print(f"📁 Source: {self.source_dir}")
        console.print(f"💾 Output: {self.output_file}")
        console.print(f"🤖 Model: {self.model}")
        console.print(f"🌐 Ollama: {self.ollama_host}\n")

        dataset = self.generate_dataset(num_action_examples)

        if dataset:
            # Shuffle for better training
            random.shuffle(dataset)
            self.save_dataset(dataset)
        else:
            console.print("[red]No examples generated![/red]")


def main():
    """Entry point for the synthetic data generator."""
    generator = SyntheticDataGenerator()
    generator.run(num_action_examples=50)


if __name__ == "__main__":
    main()
