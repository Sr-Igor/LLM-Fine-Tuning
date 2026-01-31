# 📘 Protocolo Planus: Pipeline de Fine-Tuning e Exportação de LLMs

**Stack:** Unsloth (Qwen2.5), Google Colab (T4 GPU), Google Drive, Hugging Face, Ollama.
**Objetivo:** Treinar modelos adaptados ao contexto de negócio e exportá-los para execução local via GGUF.

---

## 🏗️ Fase 1: Preparação do Ambiente (Zero Ground)

O sucesso da exportação depende de como os arquivos são estruturados no início.

### 1. Upload e Estrutura

1.  **Compactação:** No seu computador local, compacte (ZIP) a pasta do seu projeto contendo o dataset `data/processed` e os scripts necessários.
2.  **Upload:** Suba o arquivo `.zip` para a **raiz** do seu Google Drive.
3.  **Descompactação Controlada:** No Colab, **não** descompacte na raiz `/content` (que é volátil) se quiser persistência, mas para performance de I/O, `/content` é melhor.
    - _Recomendação:_ Mantenha o dataset e códigos pesados no Drive para persistência, mas esteja ciente da latência. Ou copie para `/content` no início da sessão.

### 2. Mapeamento de Caminhos (Dinâmico)

Como o nome da pasta pode mudar (ex: `meu-projeto-v1`, `planus-final`, `teste-dev`), **nunca** use caminhos absolutos hardcoded ("chumbados") nos scripts.

**Script de Setup Inicial no Colab:**

```python
import os
from google.colab import drive

# 1. Montar Drive
drive.mount('/content/drive')

# 2. Definir a Raiz do Projeto (VARIÁVEL CRÍTICA)
# Altere APENAS esta linha conforme o nome da pasta atual no seu Drive
PROJECT_ROOT_NAME = "planuze-llm-collab"

# Caminho absoluto construído dinamicamente
PROJECT_PATH = f"/content/drive/MyDrive/llm/{PROJECT_ROOT_NAME}"

print(f"📂 Diretório de trabalho definido: {PROJECT_PATH}")
```

---

## 🧠 Fase 2: Treinamento (Fine-Tuning)

Utilize o **Unsloth** para eficiência de memória e velocidade.

1.  **Dependências:** Instalar `unsloth[colab-new]`.
2.  **Configuração:** Carregar modelo base (ex: `unsloth/Qwen2.5-7B-Instruct`) em 4-bit.
3.  **Treino:** Executar `SFTTrainer`.
4.  **Salvamento dos Adaptadores (Checkpoints):**
    - O Unsloth salva apenas os adaptadores (arquivos pequenos, ~200MB).
    - **Dica de Ouro:** Configure o `output_dir` para salvar os checkpoints dentro do Drive (`f"{PROJECT_PATH}/outputs"`) para não perdê-los se o Colab desconectar.

---

## ⚠️ Fase 3: O Gargalo da Exportação (Aprendizados Críticos)

Esta é a fase onde 90% dos erros ocorrem (Disco cheio, RAM estourada, Permissões).

### Aprendizado 1: O Dilema do Disco

O Google Colab tem disco local limitado. Tentar fazer o merge do modelo full (15GB) + quantização (5GB) no `/content` pode falhar por falta de espaço.

- **Solução:** Usar o método `push_to_hub_gguf` direto (se suportado) ou montar diretórios temporários no Drive.

### Aprendizado 2: Compilação do llama.cpp

A ferramenta de conversão (GGUF) precisa ser compilada. O `make` padrão falha ou o processo é morto pela gestão de memória do Colab ("Killed").

- **Solução:** Usar `cmake` com limitação de threads (`-j 1`) para poupar memória.

**Script de Build "À Prova de Falhas":**

```bash
%%bash
# Garante execução na raiz volátil do Colab (mais rápido que o Drive para compilar)
cd /content
git clone --depth 1 https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build

# -DGGML_NATIVE=OFF aumenta compatibilidade
# -j 1 evita o erro "Killed" (OOM) na Tesla T4
cmake .. -DGGML_NATIVE=OFF
cmake --build . --config Release -j 1
```

### Aprendizado 3: O Link Simbólico

O Unsloth espera o binário `llama-quantize` na raiz da pasta `llama.cpp`, mas o CMake o cria dentro de `build/bin`.

**Ação Obrigatória:**

```python
!ln -sf /content/llama.cpp/build/bin/llama-quantize /content/llama.cpp/llama-quantize
```

---

## ☁️ Fase 4: Autenticação e Upload (Hugging Face)

**Erros comuns:** 401 (Não autorizado) e 403 (Proibido - Namespace errado).

### Checklist de Autenticação:

1.  **Token:** Precisa ser do tipo **WRITE** (Escrita). Tokens de leitura geram erro 401 na criação do repositório.
2.  **Arquivo .env:** Se estiver no Drive, o `load_dotenv()` precisa do caminho completo.

```python
from dotenv import load_dotenv
load_dotenv(f"{PROJECT_PATH}/.env") # Usa a variável dinâmica da Fase 1
```

3.  **Namespace (O erro 403):** Você não pode criar um repo para uma organização que não pertence (ex: `planuze/modelo`) se o seu usuário for `joao-dev` e não tiver permissão.

**Script de Upload Seguro:**

```python
from huggingface_hub import HfApi, login
import os

# 1. Autenticação
token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN não encontrado!")
login(token=token)

# 2. Identificação Automática do Usuário (Evita erro 403)
api = HfApi()
username = api.whoami()['name']
repo_name = "planus-qwen-v1" # Nome do modelo desejado
full_repo_id = f"{username}/{repo_name}"

print(f"🚀 Enviando para: {full_repo_id}")

# 3. Upload (Exemplo de upload manual, caso o método automático falhe)
api.create_repo(repo_id=full_repo_id, repo_type="model", exist_ok=True)
api.upload_file(
    path_or_fileobj="qwen2.5-7b-instruct.Q4_K_M.gguf", # Arquivo local gerado
    path_in_repo="planus.gguf",
    repo_id=full_repo_id
)
```

---

## 🖥️ Fase 5: Consumo Local (Deploy)

Após o sucesso no upload, o desenvolvedor baixa o modelo para sua máquina local.

1.  Instalar [Ollama](https://ollama.com).
2.  **Execução via Link Direto (Hugging Face):**

```bash
ollama run hf.co/<SEU_USER>/planus-qwen-v1
```

### Customização (Modelfile)

Para travar o Prompt do Sistema, crie um arquivo `Modelfile`:

```dockerfile
FROM ./planus.gguf
SYSTEM "Você é o Tech Lead da Planuze, especialista em..."
PARAMETER temperature 0.3
```

E crie o modelo: `ollama create planus -f Modelfile`

---

## 🚨 Troubleshooting (Resumo de Erros Reais)

| Erro Observado                                  | Causa Raiz                                     | Solução Definitiva                                                              |
| :---------------------------------------------- | :--------------------------------------------- | :------------------------------------------------------------------------------ |
| `RuntimeError: No disk space left`              | Merge do modelo estourou o disco do Colab.     | Fazer upload direto (`push_to_hub_gguf`) ou limpar cache do Hugging Face antes. |
| `RuntimeError: llama.cpp folder does not exist` | Unsloth não achou a pasta ou binário.          | Clonar manualmente e criar link simbólico para o binário (passo da Fase 3).     |
| `make: ... Build system changed`                | O repo llama.cpp mudou de Make para CMake.     | Usar script de build com `cmake` (Fase 3).                                      |
| `c++: fatal error: Killed signal`               | Compilação usou muita RAM (multithread).       | Compilar com flag `-j 1` (single thread).                                       |
| `HTTPError 401 Unauthorized`                    | Token inválido, Read-only ou não carregado.    | Gerar token **WRITE** no HF e confirmar carregamento do `.env`.                 |
| `HTTPError 403 Forbidden`                       | Tentativa de criar repo em org errada.         | Usar `api.whoami()['name']` para pegar o namespace correto.                     |
| `FileNotFoundError` (Config json)               | Drive demorou a sincronizar ou caminho errado. | Usar script de busca (`find`) ou `force_remount=True`.                          |
