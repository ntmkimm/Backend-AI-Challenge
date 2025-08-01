
# import torch
# from PIL import Image
# from strhub.data.module import SceneTextDataModule

# # Load model and image transforms
#  model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
#     hp = model.hparams
#     datamodule = SceneTextDataModule(
#         args.data_root,
#         '_unused_',
#         hp.img_size,
#         hp.max_label_length,
#         hp.charset_train,
#         hp.charset_test,
#         args.batch_size,
#         args.num_workers,
#         False,
#         rotation=args.rotation,
#     )
# parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()

# img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

# img = Image.open('/path/to/image.png').convert('RGB')
# # Preprocess. Model expects a batch of images with shape: (B, C, H, W)
# img = img_transform(img).unsqueeze(0)

# logits = parseq(img)
# logits.shape  # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol

# # Greedy decoding
# pred = logits.softmax(-1)
# label, confidence = parseq.tokenizer.decode(pred)
# print('Decoded label = {}'.format(label[0]))

import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint
import os
import json
from unidecode import unidecode

class Args:
    checkpoint = 'new-parseq.ckpt'
    data_root = './data'
    output_path = './output'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 1
    num_workers = 4
    rotation = 0

args = Args()

# Load model and image transforms
model = load_from_checkpoint(args.checkpoint).eval().to(args.device)
hp = model.hparams
datamodule = SceneTextDataModule(
    args.data_root,
    '_unused_',
    hp.img_size,
    hp.max_label_length,
    hp.charset_train,
    hp.charset_test,
    args.batch_size,
    args.num_workers,
    False,
    rotation=args.rotation,
)

img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)

# Create the output directory if it doesn't exist
os.makedirs(args.output_path, exist_ok=True)

# Recursively process each image in the input directory
for root, _, files in os.walk(args.data_root):
    folder_name = os.path.basename(root)
    all_text = []
    for img_name in files:
        img_path = os.path.join(root, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
                
            # Preprocess. Model expects a batch of images with shape: (B, C, H, W)
            img = img_transform(img).unsqueeze(0).to(args.device)
            
            logits = model(img)
            logits.shape  # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol

            # Greedy decoding
            pred = logits.softmax(-1)
            label, confidence = model.tokenizer.decode(pred)
            print(f'Decoded label for {img_name} = {label[0]}')
            
            all_text.append(label[0])  # Fix: append the correct text label
            # Save result to output path
            relative_path = os.path.relpath(root, args.data_root)
            output_dir = os.path.join(args.output_path, relative_path)
            print(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f'{img_name}.json')
            print(output_file)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({"text": label[0]}, f, ensure_ascii=False, indent=1)
        except Exception as e:
            print(f"Yo: {e}")
    if all_text:
        output_dir = os.path.join(args.output_path, os.path.relpath(root, args.data_root))
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'{folder_name}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"text": all_text}, f, indent=4)

print("Detection completed.")



# import torch
# from PIL import Image
# from strhub.data.module import SceneTextDataModule
# from strhub.models.utils import load_from_checkpoint
# import os
# import json
# from unidecode import unidecode

# class Args:
#     checkpoint = 'new-parseq.ckpt'
#     data_root = '/AIC/Detector/DeepSolo/Loc/demo/test2.png'
#     output_path = '/AIC/Detector/Parseq_Vietnamese'
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     batch_size = 1
#     num_workers = 4
#     rotation = 0

# args = Args()

# # Load model and image transforms
# model = load_from_checkpoint(args.checkpoint).eval().to(args.device)
# hp = model.hparams
# datamodule = SceneTextDataModule(
#     args.data_root,
#     '_unused_',
#     hp.img_size,
#     hp.max_label_length,
#     hp.charset_train,
#     hp.charset_test,
#     args.batch_size,
#     args.num_workers,
#     False,
#     rotation=args.rotation,
# )

# img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)

# # Create the output directory if it doesn't exist
# os.makedirs(args.output_path, exist_ok=True)

# # Recursively process each image in the input directory
# for root, _, files in os.walk(args.data_root):
#     folder_name = os.path.basename(root)
#     all_text = []
#     for img_name in files:
#         img_path = os.path.join(root, img_name)
#         try:
#             img = Image.open(img_path).convert('RGB')
                
#             # Preprocess. Model expects a batch of images with shape: (B, C, H, W)
#             img = img_transform(img).unsqueeze(0).to(args.device)
            
#             logits = model(img)
#             logits.shape  # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol

#             # Greedy decoding
#             pred = logits.softmax(-1)
#             label, confidence = model.tokenizer.decode(pred)
#             decoded_text = label[0]  # Corrected variable name
            
#             # Remove accents and convert to lowercase
#             text_no_accent = unidecode(decoded_text).lower()
            
#             print(f'Decoded label for {img_name} = {text_no_accent}')
            
#             all_text.append(text_no_accent)
#         except Exception as e:
#             print(f"Error processing {img_name}: {e}")
    
#     if all_text:
#         output_dir = os.path.join(args.output_path, os.path.relpath(root, args.data_root))
#         os.makedirs(output_dir, exist_ok=True)
#         output_file = os.path.join(output_dir, f'{folder_name}.json')
#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump({"text": all_text}, f, ensure_ascii=False, indent=1)

# print("Detection completed.")
