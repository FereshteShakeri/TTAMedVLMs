from pathlib import Path
import argparse
import json
from ret_clip.RET_CLIP.clip.model import convert_weights, CLIP
from ret_clip.RET_CLIP.training.main import convert_models_to_fp32
import os
import torch

    
    
def add_ret_args(parser):
    parser.add_argument(
        "--vision-model",
        choices=["ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-L-14-336", "ViT-H-14", "RN50"],
        default="ViT-B-16",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--text-model",
        choices=["RoBERTa-wwm-ext-base-chinese", "RoBERTa-wwm-ext-large-chinese", "RBT3-chinese"],
        default="RoBERTa-wwm-ext-base-chinese",
        help="Name of the text backbone to use.",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precition."
    )
    parser.add_argument(
        "--resume",
        default='/home/ghassen/Desktop/TTAMedVLM/ret_clip/ret-clip.pt',
        type=str,
        help="path to latest checkpoint (default: none)",
    )    
    return parser 

def prepare_ret_clip_model(args): 

    #args = parse_args()
    # Initialize the model.
    vision_model_config_file = Path(__file__).parent / f"RET_CLIP/clip/model_configs/{args.vision_model.replace('/', '-')}.json"
    print('Loading vision model config from', vision_model_config_file)
    assert os.path.exists(vision_model_config_file)
        
    text_model_config_file = Path(__file__).parent / f"RET_CLIP/clip/model_configs/{args.text_model.replace('/', '-')}.json"
    print('Loading text model config from', text_model_config_file)
    assert os.path.exists(text_model_config_file)
        
    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)
        if isinstance(model_info['vision_layers'], str):
            model_info['vision_layers'] = eval(model_info['vision_layers'])        
        for k, v in json.load(ft).items():
            model_info[k] = v

    model = CLIP(**model_info)
    convert_weights(model)

    # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
    if args.precision == "amp" or args.precision == "fp32":
        convert_models_to_fp32(model)
    # model.cuda(args.gpu)
    if args.precision == "fp16":
        convert_weights(model)

     # Resume from a checkpoint.
    print("Begin to load model checkpoint from {}.".format(args.resume))
    assert os.path.exists(args.resume), "The checkpoint file {} not exists!".format(args.resume)
    checkpoint = torch.load(args.resume, map_location='cpu')
    sd = checkpoint
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
    model.load_state_dict(sd)
    print(
        f"=> loaded checkpoint '{args.resume}'"
    )


    return model 
