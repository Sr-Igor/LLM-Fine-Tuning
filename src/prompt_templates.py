"""
Módulo contendo templates de prompt para o modelo.
"""
from transformers import PreTrainedTokenizer

ALPACA_TEMPLATE = (
    "Abaixo está uma instrução que descreve uma tarefa, "
    "emparelhada com uma entrada que fornece mais contexto. "
    "Escreva uma resposta que complete adequadamente a solicitação.\n\n"
    "### Instrução:\n"
    "{instruction}\n\n"
    "### Entrada (Contexto):\n"
    "{input}\n\n"
    "### Resposta:\n"
    "{output}"
)


def apply_chat_template(
    examples: dict, tokenizer: PreTrainedTokenizer
) -> dict:
    """
    Recebe um batch do dataset e aplica a formatação com o EOS token correto.
    """
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []

    # Pega dinamicamente do modelo (Seja Qwen ou Llama)
    eos_token = tokenizer.eos_token

    for instr, inp, out in zip(instructions, inputs, outputs):
        formatted_text = ALPACA_TEMPLATE.format(
            instruction=instr,
            input=inp,
            output=out
        ) + eos_token
        texts.append(formatted_text)

    return {"text": texts}
