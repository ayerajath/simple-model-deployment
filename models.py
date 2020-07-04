import torch
import torch.nn as nn
from constants import Constants
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig

class ReviewModel(nn.Module):
    def __init__(self):
        super(ReviewModel, self).__init__()
        tokenizer = RobertaTokenizer(
                vocab_file = Constants.VOCAB_FILE,
                merges_file = Constants.MERGES_FILE,
                add_prefix_space = True
            )
        config = RobertaConfig(output_hidden_states = True)
        self.backbone = RobertaModel(config)
        self.backbone.resize_token_embeddings(len(tokenizer))
        self.fc = nn.Linear(in_features = config.hidden_size, out_features = 1, bias = True)

    def forward(self, input_ids, attention_masks):
        outputs = self.backbone(input_ids, attention_masks)
        features = outputs[0][:, 0, :]
        x = self.fc(features)
        x = torch.squeeze(x)
        return x
