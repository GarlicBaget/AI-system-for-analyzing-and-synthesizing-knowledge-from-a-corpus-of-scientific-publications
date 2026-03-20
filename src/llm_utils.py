from llama_index.llms.ollama import Ollama


class SafeOllama(Ollama):
    """Ollama wrapper that disables usage injection for newer ollama clients."""

    def _get_response_token_counts(self, raw_response: dict) -> dict:
        return {}
