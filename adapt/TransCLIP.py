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
# from ret_clip.RET_CLIP.clip import tokenize 
from MedCLIP.medclip.prompts import generate_chexpert_class_prompts, process_class_prompts, generate_rsna_class_prompts
from medclip import constants

from ViLReF.ViLReF.eval.data import _preprocess_text

chexpert_prompts = generate_chexpert_class_prompts(n=10)
chexpert_prompts = generate_rsna_class_prompts(n=10)
# chexpert_prompts = generate_class_prompts()


def get_zero_shot_logits(query_features, clip_prototypes):

    clip_logits = 100 * query_features @ clip_prototypes

    return clip_logits.squeeze()


def build_affinity_matrix(query_features, n_neighbors):
    
    device = query_features.device
    num_samples = query_features.size(0)
    affinity = query_features.matmul(query_features.T).cpu()
    num_rows = num_samples
    num_cols = num_samples
        
    knn_index = affinity.topk(n_neighbors + 1, -1, largest=True).indices[:, 1:]
    row_indices = torch.arange(num_rows).unsqueeze(1).repeat(1, n_neighbors).flatten()
    col_indices = knn_index.flatten()
    values = affinity[row_indices, col_indices].to(device)
    W = torch.sparse_coo_tensor(torch.stack([row_indices, col_indices]).to(device), values, size=(num_rows, num_cols),
                                device=device)
    return W


class Gaussian(nn.Module):
    def __init__(self, mu, cov):
        super().__init__()
        self.mu = mu.clone()
        self.cov = cov.clone()

    def forward(self, x, no_exp=False):
        chunk_size = 2500
        N = x.shape[0]
        M, D = self.mu.shape[0], self.cov.shape[0]

        likelihoods = torch.empty((N, M), dtype=x.dtype, device=x.device)

        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)
            
            likelihoods[start_idx:end_idx] = -0.5 * ((x[start_idx:end_idx][:, None, :] - self.mu[None, :, 0, :]) ** 2 * (1 / self.cov[None, None, :])).sum(dim=2)


        if not no_exp:
            likelihoods = torch.exp(likelihoods)
        
        return likelihoods

    def set_cov(self, cov):
        self.cov = cov
        
    def set_mu(self, mu):
        self.mu = mu


def update_z(likelihoods, y_hat, z, W, lambda_y_hat, lambda_laplacian, n_neighbors, max_iter=5):
    for it in range(max_iter):
        intermediate = likelihoods.clone()
        intermediate += lambda_laplacian*(50 / (n_neighbors * 2)) * (
                W.T @ z + (W @ z))
        # For numerical stability
        intermediate -= torch.max(intermediate, dim=1, keepdim=True)[0]
        intermediate = (y_hat ** lambda_y_hat) * torch.exp(1 / 50 * intermediate)
        z = intermediate / torch.sum(intermediate, dim=1, keepdim=True)
    return z


def update_mu(adapter, query_features, z):

    mu = torch.einsum('ij,ik->jk', z, query_features) 
    mu /= torch.sum(z, dim=0).unsqueeze(-1)
    mu = mu.unsqueeze(1)
    mu /= mu.norm(dim=-1, keepdim=True)
    return mu


def update_sigma(adapter, query_features, z):
    
    n_query = z.size(0)
    chunk_size = 2500  # Iterate over query_features in chunks to avoid large memory consumption
    
    for start_idx in range(0, n_query, chunk_size):
        end_idx = min(start_idx + chunk_size, n_query)
        query_features_chunk = query_features[start_idx:end_idx]
        
        chunk_result = torch.einsum(
            'ij,ijk->k',
            z[start_idx:end_idx, :],
            (query_features_chunk[:, None, :] - adapter.mu[None, :,
                                               0, :]) ** 2)
        # If this is the first chunk, initialize cov; otherwise, accumulate
        if start_idx == 0:
            cov = chunk_result
        else:
            cov += chunk_result
        cov /= n_query 
    return cov


class TransCLIP:


    def __init__(self, args, base_model, lr, steps=10, device='cpu'):
        # loading the base model
        # base_model, _ = clip.load(model, device)
        self.model = base_model
        self.model_name = args.model
        self.templates = args.templates
        self.lambda_y_hat = 1 
        self.lambda_laplacian = 1

        self.lr = lr
        self.type = type
        self.steps = steps
        self.device = device
        self.model.to(device)
        self.model.eval()
        # Set the gradients for LayerNorm layers only for visual encoder
        # self.model.visual = self.set_ln_grads(self.model.visual)

        # Collect the LayerNorm parameters and set the optimizer
        # params, _ = self.collect_ln_params(self.model.visual)
        # self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)

        # Save the initial model and optimizer states
        # self.model_state, self.optimizer_state = self.copy_model_and_optimizer(self.model, self.optimizer)
        


    def adapt(self, x, classes):
        """
        Forward pass with adaptation.

        Args:
            x: Input image tensor.
            classes: List of class names.

        """

        self.reset()
        pred = self.perform_adaptation(x, classes)
        return pred


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
        image_features = self.model.encode_image(x)

        # Pick the top most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features, _ = self.extract_text_embeddings(classes, self.templates, average=True)
        # text_features, _ = self.extract_text_embeddings(classes, [REFERENCE_TEMPLATE], average=True)
        text_features = text_features.T

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, pred = similarity.topk(1, 1, True, True)
        pred = pred.t()
       
        return pred


    def reset(self):
        """
        Resets the model and optimizer to their initial states.
        """
        # if self.model_state is None or self.optimizer_state is None:
        #     raise Exception("Cannot reset without saved model/optimizer state")
        # self.load_model_and_optimizer(self.model, self.optimizer,
        #                               self.model_state, self.optimizer_state)
        # print("No model parameters is updated in Lame")


    def perform_adaptation(self, x, classes):
        """
        Forward pass with adaptation for test-time. The model adapts itself during testing by updating on every forward pass.

        Args:
            x: Input image tensor.
            classes: List of class names.
        """

        # text_x, texts = self.extract_text_embeddings(classes, [REFERENCE_TEMPLATE], average=False)
        # texts = [REFERENCE_TEMPLATE.format(classname.replace('_', ' ')) for classname in classes]
        # texts = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            text_x, _ = self.extract_text_embeddings(classes, self.templates, average=True)
            #text_x = torch.stack(text_x, dim=1).to(self.device)
            text_x = text_x.squeeze()


            # zs_logits = get_zero_shot_logits(query_features, clip_prototypes)
            if self.model_name == "CLIP":
                zs_logits, query_features, clip_prototypes = self.model(x, text_x, True)
            elif self.model_name == "Quilt" :
                # image_features, text_features, logit_scale = self.model(x, texts)
                #_, logits = self.model.get_logits(x, text_x.T)
                zs_logits, query_features, clip_prototypes, _ = self.model(x, text_x.T)
            elif self.model_name == "Medclip":
                logits_dict  = self.model(x, text_x.T)
                query_features = logits_dict['img_embeds']
                clip_prototypes = logits_dict['text_embeds']
                zs_logits = logits_dict['logits']

                
            elif self.model_name == "ViLRef":
                zs_logits, query_features, clip_prototypes, _  = self.model(x, text_x.T)
        if self.model_name == "Medclip":
            y_hat = F.softmax(zs_logits, dim=1)
        else:
            y_hat = F.softmax(zs_logits, dim=1).T
        z = y_hat.clone()
        
        

        max_iter = 10  # number of iterations
        n_neighbors = 3

        ###########
        # MU init #
        ###########
        clip_prototypes, _ = self.extract_text_embeddings(classes, self.templates, average=False)
        mu = clip_prototypes.permute(2,0,1) 

        ##############
        # SIGMA init #
        ##############
        
        cov = torch.ones(query_features.size(-1)).cuda() * 1/query_features.size(-1)
        
        adapter = Gaussian(mu=mu, cov=cov).cuda()
        
        ###################
        # Affinity matrix #
        ###################
        
        W = build_affinity_matrix(query_features.float(), n_neighbors)
        
        for k in range(max_iter + 1):
            
            likelihoods = adapter(query_features, no_exp=True)
            
            ############
            # Z update #
            ############

            z = update_z(likelihoods, y_hat, z, W, self.lambda_y_hat, self.lambda_laplacian, n_neighbors)
            
            if k == max_iter:  # STOP
                break

            #############
            # MU update #
            #############

            mu = update_mu(adapter, query_features, z)
            adapter.set_mu(mu)

            ################
            # SIGMA update #
            ################
            
            cov = update_sigma(adapter, query_features, z)
            adapter.set_cov(cov)
        
        z = z.softmax(dim=-1)
        values, pred = z.topk(1, 1, True, True)
        return pred.T


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
                    
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        # Pass `input_ids` and `attention_mask` to the model
                        class_embeddings = self.model.encode_text(
                        input_ids=texts["input_ids"],
                        attention_mask=texts["attention_mask"]
                        )
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    # Average embeddings for this class
                    if average:
                        class_embeddings = class_embeddings.mean(dim=0) 
                        # Store in dictionary
                        class_embeddings /= class_embeddings.norm()
                    #class_embeddings_dict[class_name] = class_embeddings
                    
                    text_features.append(class_embeddings)
                text_features = torch.stack(text_features, dim=-1).to(self.device)
                # Convert dictionary to tensor (batch_size, embedding_dim)
                #text_features = torch.stack(list(class_embeddings_dict.values())).to(self.device).T
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
                text_features = torch.stack(text_features, dim=-1).to(self.device)
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
            if isinstance(m, nn.LayerNorm):
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
