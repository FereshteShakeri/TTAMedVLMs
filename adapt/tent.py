import copy
from collections import OrderedDict

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import AutoProcessor, AutoModel

from utils.misc import load_templates_from_yaml

from conch.open_clip_custom import tokenize, get_tokenizer

from transformers import AutoTokenizer
import ret_clip.RET_CLIP.clip as cnclip 

from MedCLIP.medclip.prompts import generate_chexpert_class_prompts, process_class_prompts, generate_rsna_class_prompts, generate_class_prompts
from medclip import constants

from ViLReF.ViLReF.eval.data import _preprocess_text

REFERENCE_TEMPLATE = "an H&E stained image of {}."
REFERENCE_TEMPLATE = "a histopathology slide showing {}."
REFERENCE_TEMPLATE = "histopathology image of {}."
REFERENCE_TEMPLATE = "pathology tissue showing {}."
#REFERENCE_TEMPLATE = "presence of {} tissue on image."
chexpert_prompts = generate_chexpert_class_prompts(n=10)
# chexpert_prompts = generate_class_prompts()
#chexpert_prompts = generate_rsna_class_prompts(n=1)

class TENT:


    def __init__(self, args, base_model, lr, steps=10, device='cpu'):
        # loading the base model
        # base_model, _ = clip.load(model, device)
        self.model = base_model
        self.model_name = args.model
        self.templates = args.templates
        self.lr = lr
        self.type = type
        self.steps = steps
        self.device = device
        self.model.to(device)
        # Set the gradients for LayerNorm layers only for visual encoder
        if self.model_name == 'Medclip':
            self.model.visual = self.set_ln_grads(self.model.vision_model)
        else:
            self.model.visual = self.set_ln_grads(self.model.visual)

        # Collect the LayerNorm parameters and set the optimizer
        params, _ = self.collect_ln_params(self.model.visual)
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)

        # Save the initial model and optimizer states
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(self.model, self.optimizer)
        


    def adapt(self, x, classes):
        """
        Forward pass with adaptation.

        Args:
            x: Input image tensor.
            classes: List of class names.

        """

        self.reset()
        self.perform_adaptation(x, classes)


    @torch.no_grad()
    def evaluate(self, x, classes):
        """
        Forward pass without adaptation.

        Args:
            x: Input image tensor.
            classes: List of class names.

        Returns:
            pred: Predicted class labels for the input images.

        """
        
        with torch.no_grad():
            # extracting features
            if self.model == "Retclip":
                image_features, _, _ = self.model.encode_image(x).float()
            else:
                # image_features = self.model.encode_image(x).float()
                image_features = self.model.encode_image(x)
            #image_features = self.model(x, None)

            # Pick the top most similar labels for the image
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # print(classes)
            text_features, _ = self.extract_text_embeddings(classes, self.templates, average=True)
            #text_features, _ = self.extract_text_embeddings(classes, [REFERENCE_TEMPLATE], average=True)
            #text_features = text_features.float().T
            text_features = text_features.T

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, pred = similarity.topk(1, 1, True, True)
            pred = pred.t()
        return pred


    def reset(self):
        """
        Resets the model and optimizer to their initial states.
        """
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("Cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer(self.model, self.optimizer,
                                      self.model_state, self.optimizer_state)


    def perform_adaptation(self, x, classes):
        """
        Forward pass with adaptation for test-time. The model adapts itself during testing by updating on every forward pass.

        Args:
            x: Input image tensor.
            classes: List of class names.
        """
        
        # text_x, texts = self.extract_text_embeddings(classes, [REFERENCE_TEMPLATE], average=False)
        texts = [REFERENCE_TEMPLATE.format(classname.replace('_', ' ')) for classname in classes]
        texts = clip.tokenize(texts).to(self.device)
        text_x, _ = self.extract_text_embeddings(classes, self.templates, average=True)
        text_x = text_x.squeeze()

        for _ in range(self.steps):
            
            
                       
            if self.model_name == "CLIP":
                
                logits, image_features, text_features = self.model(x, text_x.T, True)
            elif self.model_name == "Quilt":
                # image_features, text_features, logit_scale = self.model(x, texts)
                #_, logits = self.model.get_logits(x, text_x.T)
                logits, image_features, text_features, _ = self.model(x, text_x.T)
            elif self.model_name == "CONCH":
                logits, image_features, text_features = self.model(x, texts)
                # logits = output["logits"]
            elif self.model_name == "Medclip":
                logits_dict  = self.model(x, text_x.T)
                logits = logits_dict["logits"]
            elif self.model_name == "Retclip":
                logits = self.model(x, None, text_x.T)
            elif self.model_name == "ViLRef":
                logits, _, _, _ = self.model(x, text_x.T)
            # logits, image_features, text_features, _ = self.model(x, text_x)
            # logits, _ = self.model.get_logits(x, text_x)
            # adapt
            loss = self.softmax_entropy(logits).mean(0)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def extract_text_embeddings(self, class_names, templates, average=True):  
        """
        Extracts text embeddings for given class names and templates.

        Args:
            class_names: List of class names to generate text embeddings for.
            templates: List of text templates to use for generating text embeddings.
            average: Boolean indicating whether to average the embeddings of different templates for each class.

        Returns:
            text_features: Tensor of text embeddings for the given class names and templates.
        """
        with torch.no_grad():
            text_features = []
            if self.model_name == 'Medclip':
                tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
                tokenizer.model_max_length = 77

                class_embeddings_dict = {}  # Store averaged embeddings per class

                for class_name, prompts in chexpert_prompts.items(): 
                    texts = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
                    # Move the tensors to the correct device
                    texts = {key: value.to(self.device) for key, value in texts.items()}
                    #with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        # Pass `input_ids` and `attention_mask` to the model
                    class_embeddings = self.model.encode_text(
                        input_ids=texts["input_ids"],
                        attention_mask=texts["attention_mask"]
                        )
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    # Average embeddings for this class
                    if average:
                        class_embedding_avg = class_embeddings.mean(dim=0) 
                        # Store in dictionary
                        class_embedding_avg /= class_embedding_avg.norm()
                    class_embeddings_dict[class_name] = class_embedding_avg

                # Convert dictionary to tensor (batch_size, embedding_dim)
                text_features = torch.stack(list(class_embeddings_dict.values())).to(self.device).T
                    
            else:    
                for class_name in class_names:
                    if self.model_name == 'CONCH':
                        texts = [template.format(class_name) for template in templates]

                        tokenizer = get_tokenizer()
                        texts = tokenize(texts=texts, tokenizer=tokenizer).to(self.device)
                        class_embeddings = self.model.encode_text(texts)
                    elif self.model_name == 'Retclip': 
                        from ret_clip.RET_CLIP.clip import tokenize 
                        texts = [template.format(class_name) for template in templates]
                        texts = tokenize(texts).to(self.device)
                        self.model.eval()
                        # class_embeddings = self.model(None, texts)
                        # with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        class_embeddings, _, _ = self.model.encode_text(texts)
                    elif self.model_name == 'ViLRef':
                        from ViLReF.ViLReF.clip import tokenize
                        texts = [_preprocess_text(template(class_name)) for template in templates]
                        texts = tokenize(texts).to(self.device)
                        class_embeddings = self.model.encode_text(texts)
                    else:
                        texts = [template.format(class_name) for template in templates]
                        texts = clip.tokenize(texts).to(self.device)
                        class_embeddings = self.model.encode_text(texts)
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    if average:
                        class_embeddings = class_embeddings.mean(dim=0)
                        class_embeddings /= class_embeddings.norm()
                    text_features.append(class_embeddings)
                text_features = torch.stack(text_features, dim=1).to(self.device)
        return text_features, texts


    @staticmethod
    def set_ln_grads(model):
        """
        Set gradient settings for LayerNorm layers within the model, disabling gradients globally except for these LN layers.

        Args:
            model: The model whose LayerNorm layers' gradients are to be set.
        
        Returns:
            The model with modified gradient settings.
        """
        model.requires_grad_(False)
        for m in model.modules():
            # print(m)
            if isinstance(m, nn.LayerNorm) :
                m.requires_grad_(True)
        return model


    @staticmethod
    def collect_ln_params(model):
        """
        Collect the affine scale and shift parameters from LayerNorm layers.

        Args:
            model: The model from which to collect LayerNorm parameters.
        
        Returns:
            params: List of LayerNorm parameters.
            names: List of parameter names.
        """
        params = []
        names = []
        for nm, m in model.named_modules(): 
            print(m)
            if isinstance(m, nn.LayerNorm) :
                for np, p in m.named_parameters():
                    print(np)
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"visual.{nm}.{np}")
        return params, names


    @staticmethod
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        return -(x.softmax(1) * x.log_softmax(1)).sum(1)


    @staticmethod
    def weight_average(all_weights):
        """
        Compute the average of the weights from multiple models.

        Args:
            all_weights: List of state dictionaries from different models.

        Returns:
            avg_state_dict: Averaged state dictionary.
        """
        K = len(all_weights)
        avg_state_dict = OrderedDict()
        for param_name, param in all_weights[0].items():
            avg_param = sum(sd[param_name] for sd in all_weights) / K
            avg_state_dict[param_name] = avg_param
        return avg_state_dict


    @staticmethod
    def copy_model_and_optimizer(model, optimizer):
        """
        Copy the model and optimizer states for resetting after adaptation.

        Args:
            model: The model to copy.
            optimizer: The optimizer to copy.

        Returns:
            model_state: Copied state of the model.
            optimizer_state: Copied state of the optimizer.
        """
        model_state = copy.deepcopy(model.state_dict())
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        return model_state, optimizer_state


    @staticmethod
    def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
        """
        Restore the model and optimizer states from copies.

        Args:
            model: The model to restore.
            optimizer: The optimizer to restore.
            model_state: The state to restore the model to.
            optimizer_state: The state to restore the optimizer to.
        """
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
