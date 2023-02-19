# extract clip feat from test.json
import clip
import json
from pathlib import Path

import torch
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

save_dir = "/data/yulu/PSVL_Exp/Language-Free/annotations/charades/test_text_feats"

with open("annotations/charades/train_proposal.json", "r") as f:
    test = json.load(f)

for i, anno in enumerate(tqdm(test)):
    vid = anno["vid"]
    timestamp = anno["timestamp"]
    sentence = anno["sentence"]
    duration = anno["duration"]
    with torch.no_grad():
        clip_feat = model.encode_text(clip.tokenize(sentence).to(device)).squeeze()
    moments = [timestamp[0] * duration, timestamp[1] * duration]
    # save clip feat
    torch.save(clip_feat.cpu().float().numpy(), Path(save_dir) / f"{vid}_{moments[0]}_{moments[1]}.npy")

"""

# extract proposal from train.json
json_data = 'annotations/charades/train.json'
out_dir = 'annotations/charades/train_proposal.json'
with open(json_data, 'r') as f:
    train = json.load(f)

output = []
for vid in train:
    duration = train[vid]["duration"]
    sentences = train[vid]["sentences"]
    timestamps = train[vid]["timestamps"]
    for sentence, timestamp in zip(sentences, timestamps):
        if timestamp[0] > timestamp[1] or (timestamp[1] - timestamp[0] < 0.5):
            continue
        tokens = sentence.split()
        moments = [timestamp[0] / duration, timestamp[1] / duration]
        output.append({"vid": vid, "sentence": sentence, "timestamp": moments, 'duration': duration, 'tokens': tokens})

with open(out_dir, 'w') as f:
    json.dump(output, f)
"""