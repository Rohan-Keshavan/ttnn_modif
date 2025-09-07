import os

import torch

interest = "decoder_0.pt"
if __name__ == "__main__":
    root = os.getcwd()
    ref_data_path = os.path.join(root, "reference_data")
    ref_data_file = os.path.join(ref_data_path, interest)
    ref_data = torch.load(ref_data_file)

    example = ref_data[len(ref_data) - 1]

    # Ref Inputs
    hidden_states = example["args/hidden_states"]
    attention_mask = example["args/attention_mask"]
    position_ids = example["args/position_ids"]
    past_key_value = example["args/past_key_value"]
    past_k = past_key_value[0][:, :, 0 : attention_mask.shape[3] - attention_mask.shape[2], :]
    past_v = past_key_value[1][:, :, 0 : attention_mask.shape[3] - attention_mask.shape[2], :]
    attention_mask = attention_mask[:, :, :, attention_mask.shape[3] - attention_mask.shape[2] :]
    # Ref Inputs

    print("Inputs presented...")
    print("Hiddens          : ", hidden_states.shape)
    print("Attention mask   : ", attention_mask.shape)
    print("Position IDs     : ", position_ids)
    print("Past key value   : ", past_k.shape, past_v.shape)
