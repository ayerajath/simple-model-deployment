from constants import Constants
from transformers import RobertaTokenizer
import torch
import torch.nn as nn

def get_ids_masks(text):
    tokenizer = RobertaTokenizer(
        vocab_file = Constants.VOCAB_FILE,
        merges_file = Constants.MERGES_FILE,
        add_prefix_space = True
    )

    text = " " + " ".join(text.lower().split())
    ids = tokenizer.encode(text)
    ids = [0] + ids + [2]

    pad_len = Constants.MAX_LEN - len(ids)

    if pad_len > 0:
        ids += [0] * (pad_len)

    ids = torch.tensor(ids)
    masks = torch.where(ids != 0, torch.tensor(1), torch.tensor(0))
    
    # to match the shape (batch_size, seq_length) as per the code of huggingface
    ids = torch.unsqueeze(ids, 0)
    masks = torch.unsqueeze(masks, 0)

    ids = ids.to(torch.device('cpu'))
    masks = masks.to(torch.device('cpu'))
    return ids, masks

def load_weights(model):
    model.load_state_dict(torch.load(Constants.WEIGHT_PATH, map_location=torch.device('cpu')))
    return model

def predict_text(model, text, threshold):
    ids, masks = get_ids_masks(text)
    prediction = predict_probs(model, ids, masks)
    print("Probability (before sigmoid) :", prediction)
    sigmoid = nn.Sigmoid()
    prediction = sigmoid(prediction)
    print("Probability (after sigmoid) :", prediction)
    if prediction >= threshold:
        return "Positive"
    return "Negative"

def predict_probs(model, ids, masks):
    predictions = model(ids, masks)
    return predictions