"""
Módulo para geração de dados sintéticos usando Ollama.
"""
import os
import json
import ollama
from pypdf import PdfReader
from tqdm import tqdm
from dotenv import load_dotenv

# ==========================================
# CONFIGURAÇÕES
# ==========================================

# Carrega variáveis de ambiente
load_dotenv()

# Onde estão seus PDFs/TXTs originais
SOURCE_DIR = os.getenv("SYNTHETIC_SOURCE_DIR", "data/source_documents")

# Onde o arquivo pronto para treino será salvo
OUTPUT_FILE = os.getenv("SYNTHETIC_OUTPUT_FILE",
                        "data/raw/train_data_synthetic.jsonl")

# Modelo que vai GERAR os dados (deve estar rodando no Ollama)
GENERATOR_MODEL = os.getenv("SYNTHETIC_GENERATOR_MODEL", "llama3.1")

# A Instrução do Sistema (Persona) que será gravada no dataset
# A Instrução do Sistema (Persona) que será gravada no dataset
SYSTEM_INSTRUCTION = os.getenv("SYNTHETIC_SYSTEM_INSTRUCTION", "")

if not SYSTEM_INSTRUCTION:
    print("⚠️ AVISO: Variável SYNTHETIC_SYSTEM_INSTRUCTION não definida!")
    print("   Usando fallback vazio (Isso pode prejudicar o treino).")


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
    # Chunk de ~2000 caracteres (aprox 500 tokens)
    chunk_size = 2000
    # Sobreposição para não perder contexto
    overlap = 200

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
        print(f"❌ Erro: A pasta '{directory}' não existe.")
        return []

    files = [f for f in os.listdir(directory) if f.endswith(('.pdf', '.txt'))]
    print(f"📂 Encontrados {len(files)} arquivos em: {directory}")

    for filename in files:
        filepath = os.path.join(directory, filename)

        try:
            content = _read_file_content(filepath)

            # Se extraiu texto, quebra em pedaços (chunks) para não estourar o
            # limite do Ollama
            if content:
                documents.extend(_create_chunks(content, filename))

        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"⚠️ Erro ao ler {filename}: {e}")
            continue

    print(f"✅ Texto extraído e fragmentado em {len(documents)} partes.")
    return documents


def generate_synthetic_data(documents):
    """
    Usa o Ollama para ler cada trecho de texto e criar pares
    de Pergunta/Resposta.
    """
    print(
        f"🤖 Iniciando geração com o modelo '{GENERATOR_MODEL}'... "
        "(Pode demorar)"
    )

    generated_rows = []

    # Barra de progresso para acompanhar
    for doc in tqdm(documents, desc="Processando documentos"):

        # O Prompt que pede para o LLM criar os dados
        prompt = (
            f"Analise o seguinte texto técnico extraído do arquivo "
            f"'{doc['source']}':\n\n"
            f"TEXTO:\n"
            f"\"{doc['text']}\"\n\n"
            f"TAREFA:\n"
            f"Atue como um especialista em criar datasets para treinamento.\n"
            f"Crie 2 pares de interação Usuário/Assistente baseados "
            f"EXCLUSIVAMENTE neste texto.\n"
            f"As perguntas devem simular um usuário do sistema ERP Planuze "
            f"com dúvidas reais.\n\n"
            f"Retorne APENAS um JSON válido (lista de objetos) no seguinte "
            f"formato, sem markdown:\n\n"
            f"[\n"
            f"    {{\n"
            f"        \"contexto\": \"Resumo curto e denso da informação\",\n"
            f"        \"pergunta\": \"Pergunta natural do usuário\",\n"
            f"        \"resposta\": \"Resposta técnica baseada no texto\"\n"
            f"    }}\n"
            f"]"
        )

        try:
            # Chamada ao Ollama
            response = ollama.chat(model=GENERATOR_MODEL, messages=[
                {
                    'role': 'system',
                    'content': ('Você é um gerador de datasets JSON estrito. '
                                'Responda apenas com JSON válido.')
                },
                {'role': 'user', 'content': prompt},
            ])

            content = response['message']['content']

            # Limpeza cirúrgica do JSON
            content = content.replace("```json", "").replace("```", "").strip()

            # Tenta converter string para JSON
            data = json.loads(content)

            # Formata para o padrão final do treino (Unsloth/Alpaca format)
            for item in data:
                row = {
                    "instruction": SYSTEM_INSTRUCTION,
                    # O Input simula o que o sistema RAG entregaria para o
                    # modelo em produção
                    "input": (f"[TEMA]: Documentação {doc['source']}\n"
                              f"[CONTEXTO]: {item['contexto']}\n"
                              f"[PERGUNTA]: {item['pergunta']}"),
                    "output": item['resposta']
                }
                generated_rows.append(row)

        except json.JSONDecodeError:
            # Erro comum: O modelo falou algo antes do JSON ou errou a vírgula.
            # Ignoramos este chunk.
            continue
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"\n❌ Erro na API Ollama: {e}")
            continue

    return generated_rows


def save_jsonl(data, filename):
    """Salva a lista de objetos em um arquivo .jsonl"""
    # Garante que a pasta de destino existe (data/raw)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    print(f"💾 Salvando {len(data)} exemplos em {filename}...")

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for entry in data:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        print("🚀 Sucesso! Arquivo gerado.")

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"❌ Erro ao salvar arquivo: {e}")


# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================
if __name__ == "__main__":
    print("--- INICIANDO GERADOR SINTÉTICO PLANUS ---")

    # 1. Extrair Texto
    docs = extract_text_from_files(SOURCE_DIR)

    if docs:
        # 2. Gerar Dados com IA
        dataset = generate_synthetic_data(docs)

        if dataset:
            # 3. Salvar Resultado
            save_jsonl(dataset, OUTPUT_FILE)
        else:
            print(
                "⚠️ O modelo não gerou dados válidos. "
                "Verifique se Ollama está rodando."
            )
    else:
        print("⚠️ Nenhum documento encontrado.")
        print(f"👉 Coloque arquivos .pdf ou .txt na pasta: {SOURCE_DIR}")
