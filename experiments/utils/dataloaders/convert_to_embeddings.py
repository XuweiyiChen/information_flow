import torch
import tqdm
import numpy as np
import os

def convert_image_dataset_to_embeddings(dataloader, model, save_path):
    embeddings = []
    labels = []

    with torch.no_grad():
        model.model.eval()

        for batch in tqdm.tqdm(dataloader, desc=f"Converting set to embeddings"):
            x, y = model.prepare_inputs(batch, return_labels=True)
            outputs = model(**x)
            
            if hasattr(outputs, 'hidden_states'):
                hidden_states = outputs.hidden_states
            elif isinstance(outputs, dict) and 'hidden_states' in outputs:
                hidden_states = outputs['hidden_states']
            else:
                hidden_states = outputs
            
            layerwise_feats = []
            for layer in hidden_states:
                pooled_feats = model._get_pooled_hidden_states(layer, None, method="mean") # [batch_size, hidden_size]
                layerwise_feats.append(pooled_feats.cpu().half().detach().squeeze().numpy())
            
            layerwise_feats = np.array(layerwise_feats).swapaxes(0, 1)
            embeddings.extend(layerwise_feats)
            labels.extend(y.cpu().detach().squeeze().numpy())
    
    embeddings = torch.tensor(np.array(embeddings))
    labels = torch.tensor(np.array(labels), dtype=torch.int64)

    print(embeddings.shape, labels.shape)
    tensor_dataset = torch.utils.data.TensorDataset(embeddings, labels)

    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(tensor_dataset, save_path)

    return tensor_dataset


