from .watt import WATT
# from .generic_contrastive import GenericContrastive
from .sar import SAR
# from .memo import MEMO
from .tent import TENT
# from .clipartt import CLIPARTT
from .limo import LIMO
from .ostta import OSTTA
from .TransCLIP import TransCLIP
from .lame import LAME

def get_method(args, device, model):
    if args.method == 'watt':
        print(f"Selected method: WATT with parameters: backbone={args.backbone}, lr={args.lr}, type={args.watt_type}, l={args.watt_l}, m={args.watt_m}, templates dir={args.watt_temps}, use reference template for evaluation={args.watt_reference_for_evaluation}, device={device}")
        return WATT(args, model, args.lr, type=args.watt_type, l=args.watt_l, m=args.watt_m, temps_dir=args.watt_temps, ref_eval=args.watt_reference_for_evaluation, device=device, siglip=args.siglip, dir = args.data_dir)
    elif args.method == 'sar':
        print(
            f"Selected method: SAR with parameters: backbone={args.backbone}, lr={args.lr}, steps={args.steps}, device={device}")
        return SAR(args, model,  args.lr, steps=args.steps, device=device)
    elif args.method == 'lame':
        print(
            f"Selected method: LAME with parameters: backbone={args.backbone}, device={device}")
        return LAME(args, model, device=device)
    elif args.method == 'transclip':
        print(
            f"Selected method: SAR with parameters: backbone={args.backbone}, lr={args.lr}, steps={args.steps}, device={device}")
        return TransCLIP(args, model,  args.lr, steps=args.steps, device=device)
    elif args.method == 'ostta':
        print(
            f"Selected method: OSTTA with parameters: backbone={args.backbone}, lr={args.lr}, steps={args.steps}, device={device}")
        return OSTTA(args, model,  args.lr, device=device)
    elif args.method == 'memo':
        print(
            f"Selected method: MEMO with parameters: backbone={args.backbone}, lr={args.lr}, steps={args.steps}, device={device}")
        return MEMO(args.backbone, args.lr, steps=args.steps, device=device)
    elif args.method == 'tent':
        print(
            f"Selected method: TENT with parameters: backbone={args.backbone}, lr={args.lr}, steps={args.steps}, device={device}")
        return TENT(args, model, args.lr, steps=args.steps, device=device)
    elif args.method == 'clipartt':
        print(
            f"Selected method: CLIPARTT with parameters: backbone={args.backbone}, lr={args.lr}, steps={args.steps}, device={device}")
        return CLIPARTT(args.backbone, args.lr, K=args.clipartt_K, steps=args.steps, device=device)
    elif args.method == 'limo':
        print(
            f"Selected method: LIMO with parameters: backbone={args.backbone}, lr={args.lr}, steps={args.steps}, device={device}")
        return LIMO(args, model, args.lr, steps=args.steps, device=device)



    # Add other methods here 
    else:
        raise ValueError(f"Unknown method: {args.method}")
