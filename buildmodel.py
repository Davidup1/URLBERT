from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
)


def buildBERT(vocab_size):
    vocab_size = vocab_size

    config_kwargs = {
        "cache_dir": None,
        "revision": 'main',
        "use_auth_token": None,
        "hidden_dropout_prob": 0.2,
        "vocab_size": vocab_size
    }

    config = AutoConfig.from_pretrained('./bert_config/', **config_kwargs)

    model = AutoModelForMaskedLM.from_config(
        config=config,
    )
    model.resize_token_embeddings(vocab_size)

    return model
