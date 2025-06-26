import copy
from collections import OrderedDict

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import AutoProcessor, AutoModel

from utils.misc import load_templates_from_yaml
from loralib.utils import apply_lora, mark_only_lora_as_trainable, get_lora_parameters, apply_lora_for_medclip, apply_lora_for_retclip
#from transformers import BertTokenizer
from transformers import AutoTokenizer
#from ret_clip.RET_CLIP.clip import tokenize 
from medclip.prompts import generate_chexpert_class_prompts, process_class_prompts
from medclip import constants

from ViLReF.ViLReF.eval.data import _preprocess_text

REFERENCE_TEMPLATE = "A radiology image of {}."
chexpert_prompts = generate_chexpert_class_prompts(n=10)
#print(chexpert_prompts)

REFERENCE_TEMPLATE = "an H&E stained image of {}."
#REFERENCE_TEMPLATE = "a histopathology slide showing {}."
#REFERENCE_TEMPLATE = "histopathology image of {}."
REFERENCE_TEMPLATE = "pathology tissue showing {}."
#REFERENCE_TEMPLATE = "presence of {} tissue on image."
#REFERENCE_TEMPLATE = "a {}."


class LIMO:



    def __init__(self, args, base_model, lr, steps=10, device='cpu'):

        # loading the base model and applying LoRA
        # base_model, _ = clip.load(model, device)
        self.model = base_model
        self.model_name = args.model
        self.templates = args.templates
        self.lr = lr 
        self.steps = steps 
        self.device =  device 
        self.model.to(device)
        # print(self.model)
        if self.model_name == 'Medclip':
            list_lora_layers = apply_lora_for_medclip(args, self.model)
            # self.model.visual = self.set_ln_grads(self.model.vision_model)
        elif self.model_name == 'ViLRef':
           list_lora_layers = apply_lora_for_retclip(args, self.model)
        else:
            list_lora_layers = apply_lora(args, self.model)
            #print(list_lora_layers)
        self.marginal_entropy_weight = args.marginal_entropy_weight
        self.text_reg_weight  = args.text_reg_weight

        
        # Making only LoRA as trainable
        mark_only_lora_as_trainable(self.model)
        #print(self.model)
        # Set the gradients for LayerNorm layers only for visual encoder
        
        # Collect LoRA parameters and set the optimizer
        params = get_lora_parameters(self.model) 
        
        # params, _ = self.collect_ln_params(self.model.visual)
        self.optimizer = torch.optim.AdamW(params, weight_decay=1e-2, betas=(0.9, 0.999), lr=self.lr)

        self.zero_shot_model = copy.deepcopy(self.model)
        self.freeze_model(self.zero_shot_model)

        self.model.to(device)
        self.zero_shot_model.to(device)

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
            if isinstance(m, nn.LayerNorm): 
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"visual.{nm}.{np}")
        return params, names

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

        # extracting features
        with torch.no_grad():
            # with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            image_features = self.model.encode_image(x)

            # Pick the top most similar labels for the image
            image_features /= image_features.norm(dim=-1, keepdim=True)
            #text_features, _ = self.extract_text_embeddings(classes, [REFERENCE_TEMPLATE], average=True)
            text_features = self.extract_text_embeddings(classes, self.templates, average=True)
            #if self.model_name == "ViLRef" or self.model_name == "Quilt":
            #    text_features = text_features.half().T
            #else:
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


    def extract_text_embeddings(self, class_names, templates, average=True, state="normal"):  
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
            texts_l = []
            if self.model_name == 'Medclip':
                tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
                tokenizer.model_max_length = 77

                class_embeddings_dict = {}  # Store averaged embeddings per class
                # print()
                for class_name, prompts in chexpert_prompts.items(): 
                    # print(class_name)
                    #print(prompts)
                    #print("&&&&&&&&&&&&&&&&&&&&&&&&")
                    #print(chexpert_prompts[class_name])

                    texts = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
                    
                    # Move the tensors to the correct device
                    texts = {key: value.to(self.device) for key, value in texts.items()}
                    #print(texts)
                    input_ids = texts["input_ids"].to(self.device).to(torch.long)
                    attention_mask = texts["attention_mask"].to(self.device).to(torch.long)
                    # print(input_ids)
                    #with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    # Pass `input_ids` and `attention_mask` to the model
                    #if state == "zero_shot":
                    #    class_embeddings = self.zero_shot_model.encode_text(
                    #    input_ids=input_ids,
                    #    attention_mask=attention_mask
                    #    )
                    #else:
                    class_embeddings = self.model.encode_text(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                    )
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    # Average embeddings for this class
                    if average:
                        class_embedding_avg = class_embeddings.mean(dim=0) 
                        # Store in dictionary
                        class_embedding_avg /= class_embedding_avg.norm()
                    class_embeddings_dict[class_name] = class_embedding_avg
                    texts_l.append(texts)

                # Convert dictionary to tensor (batch_size, embedding_dim)
                text_features = torch.stack(list(class_embeddings_dict.values())).to(self.device).T
                #print(text_features.shape)
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
                        #with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        texts = [template.format(class_name) for template in templates]
                        texts = clip.tokenize(texts).to(self.device)
                        class_embeddings = self.model.encode_text(texts)
                    class_embeddings = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
                    if average:
                        class_embeddings = class_embeddings.mean(dim=0)
                        class_embeddings = class_embeddings/class_embeddings.norm()
                    text_features.append(class_embeddings)
                text_features = torch.stack(text_features, dim=1).to(self.device)
        return text_features



    def perform_adaptation(self, x, classes):
        """
        Forward pass with adaptation for test-time. The model adapts itself during testing by updating on every forward pass.

        Args:
            x: Input image tensor.
            classes: List of class names.
        """

        # text_x, texts = self.extract_text_embeddings(classes, [REFERENCE_TEMPLATE], average=True)
        # text_x, texts = self.extract_text_embeddings(classes, templates, average=True)
        #texts = []
        # for class_name in classes:
        #     texts.append([template.format(class_name.replace('_', ' ')) for template in templates])
        # texts = [item for sublist in texts for item in sublist]

        
        # Compute cosine similarity logits for the zero-shot model
        with torch.no_grad():
            # extracting features
            #with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            #if self.model_name == "Medclip":
            #    zero_shot_text_features = self.extract_text_embeddings(classes, self.templates, average=True, state = "zero_shot")
            #    logits_dict  = self.zero_shot_model(x, zero_shot_text_features.T)
            #    zero_shot_cosine_similarity = logits_dict['logits']
            #else:
            zero_shot_image_features = self.zero_shot_model.encode_image(x)

            # Pick the top most similar labels for the image
            zero_shot_image_features /= zero_shot_image_features.norm(dim=-1, keepdim=True)
            #text_features, _ = self.extract_text_embeddings(classes, [REFERENCE_TEMPLATE], average=True)
            zero_shot_text_features = self.extract_text_embeddings(classes, self.templates, average=True)

            #if self.model_name == "ViLRef" or self.model_name == "Quilt":
            #    zero_shot_text_features = zero_shot_text_features.half().T
            #else:
            zero_shot_text_features = zero_shot_text_features.T
            zero_shot_cosine_similarity = 100 * zero_shot_image_features @ zero_shot_text_features.T

        #torch.autograd.set_detect_anomaly(True)
        for _ in range(self.steps):
            text_features = self.extract_text_embeddings(classes, self.templates, average=True)
            text_features  = text_features.squeeze()
            if self.model_name == "Medclip":
                logits_dict  = self.model(x, text_features.T)
                image_features = logits_dict['img_embeds']
                text_features = logits_dict['text_embeds'].T
                cosine_similarity = logits_dict['logits']
                # cosine_similarity = 100 * image_features @ text_features
            else:
                #with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                #with torch.no_grad():

                image_features = self.model.encode_image(x)
                image_features = image_features/image_features.norm(dim=-1, keepdim=True)

                cosine_similarity = 100 * image_features @ text_features
            loss = self.softmax_entropy(cosine_similarity).mean(0) + self.marginal_entropy_weight * self.marginal_entropy(cosine_similarity) + self.text_reg_weight * self.text_regularization(cosine_similarity, zero_shot_cosine_similarity)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


    
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


    @staticmethod
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        return -(x.softmax(1) * x.log_softmax(1)).sum(1)


    @staticmethod
    def marginal_entropy(similarities): 
        """Compute the KL divergence between the marginal probabilities of the current model and a uniform distribution (marginal entropy)"""

        input_dist = F.softmax(similarities, dim=-1)
        num_classes = input_dist.size(1)
        input_dist = input_dist.mean(dim=0) +1e-5
 
        target_dist = torch.full_like(input_dist, 1.0 / num_classes)
        input_dist = input_dist.log()
                    
        #compute the marginal entropy
        marginal_kl = F.kl_div(input_dist, target_dist, reduction='batchmean')

        return marginal_kl

    @staticmethod
    def text_regularization(similarities, zero_shot_similarities): 
        """Compute the KL divergence between current model and zero-shot model (text regularization term)"""
        
        log_probs = F.log_softmax(similarities, dim=-1)
        zero_shot_probs = F.softmax(zero_shot_similarities, dim=-1)
        kl_loss = F.kl_div(log_probs, zero_shot_probs, reduction='batchmean')

        return kl_loss

    
    @staticmethod
    def freeze_model(model: nn.Module) -> None:
        for n, p in model.named_parameters():
            p.requires_grad = False







