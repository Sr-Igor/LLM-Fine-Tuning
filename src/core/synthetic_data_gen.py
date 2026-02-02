"""
Módulo para geração de dados sintéticos usando Ollama.
"""
import os
import json
import re
import ollama
from pypdf import PdfReader
from tqdm import tqdm
from dotenv import load_dotenv
from src.core.utils.logger import logger
from config.settings import SyntheticConfig

# ==========================================
# CONFIGURAÇÕES
# ==========================================

# Carrega variáveis de ambiente
load_dotenv()

# Carrega configurações via dataclass centralizada
CONFIG = SyntheticConfig.from_env()

if not CONFIG.system_instruction:
    logger.warning(
        "⚠️ AVISO: Variável SYNTHETIC_SYSTEM_INSTRUCTION não definida!")
    logger.warning("   Usando fallback vazio (Isso pode prejudicar o treino).")


# ==========================================
# FUNÇÕES
# ==========================================


def _read_file_content(filepath):
    """
    Lê o conteúdo de um arquivo PDF ou TXT.
    """
    content = ""
    if filepath.endswith(".pdf"):
        reader = PdfReader(filepath)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                content += extracted + "\n"
    elif filepath.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    return content


def _create_chunks(content, filename):
    """
    Divide o conteúdo em chunks para processamento.
    """
    chunks = []
    # Chunk size e overlap configuráveis
    chunk_size = CONFIG.chunk_size
    # Sobreposição para não perder contexto
    overlap = CONFIG.overlap

    for i in range(0, len(content), chunk_size - overlap):
        chunk = content[i:i+chunk_size]
        if len(chunk) > 100:  # Ignora pedaços muito pequenos
            chunks.append({
                "source": filename,
                "text": chunk
            })
    return chunks


def extract_text_from_files(directory):
    """
    Lê todos os arquivos .pdf e .txt da pasta especificada.
    Retorna uma lista de dicionários com 'source' (nome do arquivo) e 'text'
    (conteúdo).
    """
    documents = []

    # Verifica se a pasta existe
    if not os.path.exists(directory):
        logger.error("❌ Erro: A pasta '%s' não existe.", directory)
        return []

    files = [f for f in os.listdir(directory) if f.endswith(('.pdf', '.txt'))]
    logger.info("📂 Encontrados %d arquivos em: %s", len(files), directory)

    for filename in files:
        filepath = os.path.join(directory, filename)

        try:
            content = _read_file_content(filepath)

            # Se extraiu texto, quebra em pedaços (chunks) para não estourar o
            # limite do Ollama
            if content:
                documents.extend(_create_chunks(content, filename))

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("⚠️ Erro ao ler %s: %s", filename, e)
            continue

    logger.info("✅ Texto extraído e fragmentado em %d partes.", len(documents))
    return documents


def _cleanup_json_response(content):
    """
    Tenta extrair e limpar o JSON da resposta do LLM.
    """
    # 1. Tenta encontrar bloco JSON com regex
    json_match = re.search(r'\[.*\]', content, re.DOTALL)
    if json_match:
        return json_match.group(0)

    # 2. Se não achar lista, tenta objeto único
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        # Se for um único objeto, envelopa em lista
        return f"[{json_match.group(0)}]"

    # 3. Fallback: limpeza básica
    content = content.replace("```json", "").replace("```", "").strip()
    return content


def _process_single_document(doc):
    """
    Processa um único documento e gera pares de QA.
    """
    generated_rows = []

    # Número variável de exemplos por documento (5-8)
    num_examples = CONFIG.examples_per_chunk

    # O Prompt que pede para o LLM criar os dados
    prompt = (
        f"Analise o seguinte texto técnico extraído do arquivo "
        f"'{doc['source']}':\n\n"
        f"TEXTO:\n"
        f"\"{doc['text']}\"\n\n"
        f"TAREFA CRÍTICA:\n"
        f"Você é um especialista em criar datasets de alta qualidade "
        f"para fine-tuning de LLMs.\n\n"
        f"Crie {num_examples} pares de interação Usuário/Assistente "
        f"baseados EXCLUSIVAMENTE neste texto.\n\n"
        f"REQUISITOS OBRIGATÓRIOS:\n"
        f"1. VARIEDADE: Crie perguntas de diferentes tipos:\n"
        f"   - Perguntas simples e diretas (30%)\n"
        f"   - Perguntas compostas que requerem múltiplos dados (30%)\n"
        f"   - Perguntas que exigem raciocínio ou comparação (20%)\n"
        f"   - Perguntas contextualizadas (continuação de conversa) (20%)\n\n"
        f"2. REALISMO: As perguntas devem simular usuários reais do ERP:\n"
        f"   - Use linguagem natural e coloquial\n"
        f"   - Inclua ambiguidades ocasionais\n"
        f"   - Varie o nível de detalhe solicitado\n\n"
        f"3. SAUDAÇÕES: EVITE começar TODAS as respostas com "
        f"'Olá [nome]'.\n"
        f"   - 40% das respostas: SEM saudação (direto ao ponto)\n"
        f"   - 30% das respostas: Saudação variada (Bom dia, Claro, etc)\n"
        f"   - 30% das respostas: Com 'Olá [nome]'\n\n"
        f"4. COMPLEXIDADE: Varie a complexidade das respostas:\n"
        f"   - Respostas curtas e objetivas\n"
        f"   - Respostas com múltiplos dados estruturados\n"
        f"   - Respostas que explicam o raciocínio\n\n"
        f"5. CONTEXTO: Inclua detalhes relevantes do texto original\n\n"
        f"FORMATO DE SAÍDA:\n"
        f"Retorne APENAS um JSON válido (lista de objetos) sem markdown:\n\n"
        f"[\n"
        f"    {{\n"
        f"        \"contexto\": \"Resumo denso e rico da informação "
        f"relevante\",\n"
        f"        \"pergunta\": \"Pergunta natural e variada do usuário\",\n"
        f"        \"resposta\": \"Resposta precisa SEM saudação repetitiva\"\n"
        f"    }}\n"
        f"]\n\n"
        f"IMPORTANTE: Gere exatamente {num_examples} exemplos diversos e "
        f"de alta qualidade."
    )

    try:

        # Chamada ao Ollama
        response = ollama.chat(model=CONFIG.generator_model, messages=[
            {
                'role': 'system',
                'content': ('Você é um gerador de datasets JSON estrito. '
                            'Responda apenas com JSON válido.')
            },
            {'role': 'user', 'content': prompt},
        ])

        content = response['message']['content']
        cleaned_content = _cleanup_json_response(content)

        # Tenta converter string para JSON
        data = json.loads(cleaned_content)

        # Se não for lista, garante que seja
        if not isinstance(data, list):
            if isinstance(data, dict):
                data = [data]
            else:
                raise ValueError("JSON não é nem lista nem objeto")

        # Formata para o padrão final do treino (Unsloth/Alpaca format)
        for item in data:
            # Validação básica de chaves
            required_keys = ("contexto", "pergunta", "resposta")
            if not all(k in item for k in required_keys):
                logger.warning("Skipping item missing keys")
                continue

            row = {
                "instruction": CONFIG.system_instruction,
                # O Input simula o que o sistema RAG entregaria para o
                # modelo em produção
                "input": (
                    f"[{CONFIG.chat_history}]: \n"
                    f"[{CONFIG.chat_subject}]: Documentação {doc['source']}\n"
                    f"[{CONFIG.chat_context}]: {item['contexto']}\n"
                    f"[{CONFIG.chat_question}]: {item['pergunta']}\n"
                    f"[{CONFIG.chat_language}]: pt"
                ),
                "output": item['resposta']
            }
            generated_rows.append(row)

    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON for doc %s", doc['source'])
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("\n❌ Erro na API Ollama: %s", e)

    return generated_rows


def generate_synthetic_data(documents):
    """
    Usa o Ollama para ler cada trecho de texto e criar pares
    de Pergunta/Resposta.
    """
    logger.info(
        "🤖 Iniciando geração com o modelo '%s'... (Pode demorar)",
        CONFIG.generator_model
    )

    all_generated_rows = []

    # Barra de progresso para acompanhar
    for doc in tqdm(documents, desc="Processando documentos"):
        rows = _process_single_document(doc)
        all_generated_rows.extend(rows)

    return all_generated_rows


def save_jsonl(data, filename):
    """Salva a lista de objetos em um arquivo .jsonl"""
    # Garante que a pasta de destino existe (data/raw)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    logger.info("💾 Salvando %d exemplos em %s...", len(data), filename)

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for entry in data:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        logger.info("🚀 Sucesso! Arquivo gerado.")

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("❌ Erro ao salvar arquivo: %s", e)


# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================
if __name__ == "__main__":
    logger.info("--- INICIANDO GERADOR SINTÉTICO ---")

    # 1. Extrair Texto
    docs = extract_text_from_files(CONFIG.source_dir)

    if docs:
        # 2. Gerar Dados com IA
        dataset = generate_synthetic_data(docs)

        if dataset:
            # 3. Salvar Resultado
            save_jsonl(dataset, CONFIG.output_file)
        else:
            logger.warning(
                "⚠️ O modelo não gerou dados válidos. "
                "Verifique se Ollama está rodando."
            )
    else:
        logger.warning("⚠️ Nenhum documento encontrado.")
        logger.info("👉 Coloque arquivos .pdf ou .txt na pasta: %s",
                    CONFIG.source_dir)
