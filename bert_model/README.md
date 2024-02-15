# URLBERT Model

If you want to get the model gradient of URLBERT, please visit the following link

[Google Drive Repository](https://drive.google.com/drive/folders/16pNq7C1gYKR9inVD-P8yPBGS37nitE-D?usp=drive_link)

### Usage

``` python
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
)
import torch

config_kwargs = {
    "cache_dir": None,
    "revision": 'main',
    "use_auth_token": None,
    "hidden_dropout_prob": 0.2,
    "vocab_size": 5000,
}

config = AutoConfig.from_pretrained("./bert_config/", **config_kwargs)
print(config)

bert_model = AutoModelForMaskedLM.from_config(
    config=config,
)
bert_model.resize_token_embeddings(config_kwargs["vocab_size"])

bert_dict = torch.load("./bert_model/urlBERT.pt", map_location='cpu')
bert_model.load_state_dict(bert_dict)
```

