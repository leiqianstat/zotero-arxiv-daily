from .base import BaseReranker, register_reranker
import logging
from time import sleep
import warnings
import numpy as np
from loguru import logger

MODEL_LOAD_MAX_ATTEMPTS = 3
MODEL_LOAD_RETRY_DELAY = 30


def _load_sentence_transformer(model: str):
    from sentence_transformers import SentenceTransformer

    for attempt in range(MODEL_LOAD_MAX_ATTEMPTS):
        try:
            return SentenceTransformer(model, trust_remote_code=True)
        except OSError as exc:
            if attempt == MODEL_LOAD_MAX_ATTEMPTS - 1:
                raise
            wait = MODEL_LOAD_RETRY_DELAY * (attempt + 1)
            logger.warning(
                f"Failed to load embedding model {model}: {exc}. "
                f"Retry {attempt + 1}/{MODEL_LOAD_MAX_ATTEMPTS - 1} in {wait}s"
            )
            sleep(wait)


@register_reranker("local")
class LocalReranker(BaseReranker):
    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        if not self.config.executor.debug:
            from transformers.utils import logging as transformers_logging
            from huggingface_hub.utils import logging as hf_logging
    
            transformers_logging.set_verbosity_error()
            hf_logging.set_verbosity_error()
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", category=FutureWarning)

        encoder = _load_sentence_transformer(self.config.reranker.local.model)
        if self.config.reranker.local.encode_kwargs:
            encode_kwargs = self.config.reranker.local.encode_kwargs
        else:
            encode_kwargs = {}
        s1_feature = encoder.encode(s1,**encode_kwargs,show_progress_bar=True)
        s2_feature = encoder.encode(s2,**encode_kwargs,show_progress_bar=True)
        sim = encoder.similarity(s1_feature, s2_feature)
        return sim.numpy()