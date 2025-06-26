import os
import clip
import torch
import argparse
import open_clip
import numpy as np
from tqdm import tqdm

from datasets import get_dataloader
from adapt import get_method
from utils import datasets 
from utils.misc import set_global_seeds, save_configuration
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel
from torchvision import transforms
from ret_clip.utils import add_ret_args, prepare_ret_clip_model
from ViLReF.utils import add_vilref_args, prepare_ViLRef_model
from conch.open_clip_custom import create_model_from_pretrained


def argparser():
    parser = argparse.ArgumentParser("Weight Average Test Time Adaptation of CLIP")

    # Directories
    parser.add_argument('--data_dir', type=str, default='./data/', help='Root directory for datasets')
    parser.add_argument('--save_dir', type=str, default='save/', help='Path for saving base training weights and results')

    # General settings
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # Model
    parser.add_argument('--backbone', type=str, default='ViT-B/32', help='Model backbone to use') 
    parser.add_argument('--siglip', action='store_true', help='Use siglip, else use clip')
    parser.add_argument('--model', type=str, default='CLIP', help='Model to use: e.g CLIP, Quilt, ViLRef')

    # Dataset settings
    parser.add_argument('--dataset', type=str, default='nct', choices=('nct','skincancer', 'sicap_mil', 'lc_lung', 'sicapv2','chexpert', 'covid_4classes', 'covid_2classes', 'rsna', 'mimic', 'messidor', 'fives', 'odir'), help='Dataset to use')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers for data loading')

    # Training settings
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--trials', default=3, type=int, help='Number of trials to repeat the experiments')
    parser.add_argument('--steps', default=10, type=int, help='Number of adaptation iterations')

    # Evaluation settings
    parser.add_argument('--adapt', action='store_true', help='Enable adaptation')

    # Corruptions settings
    parser.add_argument('--corruptions_list', nargs='+', default=None, type=str, help='List of corruptions to apply to the dataset (Cifar datasets)')

    # Method name
    parser.add_argument('--method', type=str, default='watt', choices=('watt', 'sar', 'transclip', 'memo', 'tent', 'clipartt', 'limo', 'ostta', 'lame'), help='Method to use for adaptation')

    return parser

def add_method_specific_args(parser, method):
    '''
    Add method-specific arguments to the parser
    '''
    if method == 'watt':
        parser.add_argument('--watt_type', type=str, default='sequential', choices=('parallel', 'sequential'), help='Type of WATT adaptation (parallel or sequential)')
        parser.add_argument('--watt_l', default=2, type=int, help='Number of adaptation iterations for each text embedding before weight averaging')
        parser.add_argument('--watt_m', default=5, type=int, help='Number of repetitions of the adaptation and weight averaging process')
        parser.add_argument('--watt_temps', type=str, default='templates.yaml', help='Path to the templates.yaml file')
        parser.add_argument('--watt_reference_for_evaluation', action='store_true', help='Use REFERENCE_TEMPLATE during evaluation instead of averaging text embeddings of different templates')

    elif method == 'limo': 
        # LoRA arguments
        parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
        parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
        parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'], help='list of attention matrices where putting a LoRA') 
        parser.add_argument('--r', default=2, type=int, help='the rank of the low-rank matrices')
        parser.add_argument('--alpha', default=1, type=int, help='scaling (see LoRA paper)')
        parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applied before the LoRA module')

        # LIMO arguments
        parser.add_argument('--text_reg_weight', default=0.1, type=float)
        parser.add_argument('--marginal_entropy_weight', default=5, type=float)
    # Add other methods here
    # else:
    #     raise ValueError(f"Unknown method: {method}")
    
    return parser

def _convert_to_rgb(image):
    return image.convert('RGB')

def main():
    # Initial argument parsing to get the method
    initial_parser = argparser()
    initial_args, _ = initial_parser.parse_known_args()

    # Create a new parser with method-specific arguments
    parser = argparser()
    parser = add_method_specific_args(parser, initial_args.method)
    if initial_args.model == 'Retclip':
        parser = add_ret_args(parser)
    if initial_args.model == 'ViLRef':
        parser = add_vilref_args(parser)
    args = parser.parse_args()
    print(args)

    # Set the global random seed for reproducibility
    set_global_seeds(args.seed)

    # Save the configuration settings
    save_configuration(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    
    # Get Model
    _, preprocess_val = clip.load(args.backbone)
    if args.model == 'CLIP':
        model, preprocess = clip.load(args.backbone, device)
    elif args.model == 'Quilt':
        model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32')
    # model.eval().to(device)
    
    elif args.model == 'CONCH':
        model, preprocess = create_model_from_pretrained("conch_ViT-B-32", checkpoint_path="checkpoints/conch/pytorch_model.bin")

    elif args.model == 'Medclip':
        model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        model.from_pretrained()
        preprocess = transforms.Compose([
                # transforms.RandomHorizontalFlip(0.5),
                # transforms.ColorJitter(0.1,0.1),
                transforms.ToTensor(),
                transforms.Resize((224,224)),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])],
            )

    elif args.model == 'Retclip':
        model = prepare_ret_clip_model(args)
        state_dict = torch.load("/export/livia/home/vision/Fshakeri/miccai_tta/TTAMedVLM/ret-clip.pt")
        new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
        result = model.load_state_dict(new_state_dict)
        loaded_keys = set(new_state_dict) - set(result.unexpected_keys)
        #print("Missing keys in the model:", result.missing_keys)
        preprocess = transforms.Compose([
                # transforms.RandomHorizontalFlip(0.5),
                # transforms.ColorJitter(0.1,0.1),
                transforms.ToTensor(),
                transforms.Resize((224,224)),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])],
            )
        
    elif args.model == 'ViLRef':
        model = prepare_ViLRef_model(args)
        #state_dict = torch.load("/export/livia/home/vision/Fshakeri/miccai_tta/TTAMedVLM/ViLReF_ViT.pt")
        #new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
        #result = model.load_state_dict(new_state_dict)
        #loaded_keys = set(new_state_dict) - set(result.unexpected_keys)
        #print("Missing keys in the model:", result.missing_keys)
        preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        _convert_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    
    results_path = os.path.join(args.save_dir, "results.txt")
    
    data_loader, classes, templates = get_dataloader(args.data_dir, args.dataset, args.batch_size, args.workers, preprocess, args.seed, args.model)
    if args.model == "CLIP":
        templates = ["a photo of a {}."]
    # Setting up the model and the method
    args.templates = templates
    adapt_method = get_method(args, device, model)
    # data_loader, classes = datasets.prepare_data(args.dataset, args.data_dir, corruption=corruption, batch_size=args.batch_size, num_workers=args.workers)
    acc = []
    for t in range(args.trials):
        correct = 0
        for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            if len(batch) == 2:
                inputs, labels = batch  # Unpack normally
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            elif len(batch) == 3:
                inputs = batch['image']  # Ignore the first element
                labels = batch['label']
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements: {batch}")

            # reset the model before adapting to a new batch
            adapt_method.reset()
            
            # perform adaptation
            if args.method == "transclip" or args.method == "lame":
                pred = adapt_method.adapt(inputs, classes)
            else:
                if args.adapt:
                    adapt_method.adapt(inputs, classes)

                # perform evaluation 
                pred = adapt_method.evaluate(inputs, classes)

            # Calculate the number of correct predictions
            correctness = pred.eq(labels.view(1, -1).expand_as(pred))
            correct += correctness.sum().item()
            #print(correct)

        acc.append(correct / len(data_loader.dataset))
        print(correct / len(data_loader.dataset))
    
    print(str(round(np.array(acc).mean()*100, 2)) + ',' + str(round(np.array(acc).std()*100, 2)))
    with open(results_path, 'w') as fichier:
        fichier.write(str(round(np.array(acc).mean()*100, 2)) + ',' + str(round(np.array(acc).std()*100, 2)) + '\n')

if __name__ == "__main__":
    main()
