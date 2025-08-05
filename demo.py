import torch
from genov0_1b.geno_infer_moba import GenoModel
from sklearn.metrics import matthews_corrcoef
import json
from loguru import logger

if __name__ == "__main__":
    model = "model_Zero.pth"
    tokenizer = "tokenizer_v2_k6"
    config = "params_moba_cds.config"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CKPT_PATH = "./ckpt"
    gemo_model = GenoModel(CKPT_PATH,
                           device=DEVICE,
                           model_name=model,
                           tokenizer_file=tokenizer,
                           config=config,
                           target_labels=None)

    sequence = 'acgtatcgatcg'
    print("input ids: ", gemo_model.tokenizer.encode(sequence))

    input_ids = torch.tensor(
        gemo_model.tokenizer.encode(sequence),
        dtype=torch.int,
    ).unsqueeze(0).to(DEVICE)
    print("origin seq: ", gemo_model.tokenizer.decode(input_ids[0].tolist()))
    print(input_ids.shape)
    gemo_model.model = gemo_model.model.to(DEVICE)
    outputs = gemo_model.infer(input_ids)

    # logits
    logits = outputs.logits
    print('Shape (batch, length, vocab): ', logits.shape)

    # embeddings
    hidden_states = outputs.hidden_states
    print('hidden_states (batch, length, embedding_dim): ', hidden_states.shape)





