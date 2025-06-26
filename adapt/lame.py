from typing import Dict, Any, Tuple
from typing_extensions import TypeAlias

import clip
import torch
import torch.nn as nn
import torch.jit

from torch import Tensor

from transformers import AutoTokenizer
from ret_clip.RET_CLIP.clip import tokenize 
from medclip.prompts import generate_chexpert_class_prompts, process_class_prompts
from medclip import constants

Kwargs: TypeAlias = Dict[str, Any]

chexpert_prompts = generate_chexpert_class_prompts()
# chexpert_prompts = generate_class_prompts()
from ViLReF.ViLReF.eval.data import _preprocess_text

class LAME:
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, args, base_model, affinity='knn', episodic=False, device='cpu'):
        super().__init__()
        self.model = base_model
        self.model_name = args.model
        self.templates = args.templates
        self.model.eval()
        self.device = device
        self.model.to(device)
        self.affinity = affinity
        self.episodic = episodic


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
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            image_features = self.model.encode_image(x)

        # Pick the top most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        # text_features, _ = self.extract_text_embeddings(classes, [REFERENCE_TEMPLATE], average=True)
        text_features, _ = self.extract_text_embeddings(classes, self.templates, average=True)
        text_features = text_features.T

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, pred = similarity.topk(1, 1, True, True)
        pred = pred.t()
       
        return pred

    
    def reset(self):
        """
        Resets the model and optimizer to their initial states.
        """
        #if self.model_state is None or self.optimizer_state is None:
        #    raise Exception("Cannot reset without saved model/optimizer state")
        #self.load_model_and_optimizer(self.model, self.optimizer,
        #                              self.model_state, self.optimizer_state)
        # print("No model parameters is updated in Lame")

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

    def perform_adaptation(self, x, classes):

        with torch.no_grad():
            text_features, _ = self.extract_text_embeddings(classes, self.templates, average=True)
            text_features = text_features.T
            
            if self.model_name == "Medclip":
                logits_dict  = self.model(x, text_features)
                image_features = logits_dict['img_embeds']
                text_features = logits_dict['text_embeds']
            elif self.model_name == "ViLRef":
                logits, image_features, text_features, _ = self.model(x, text_features)
            else:
                image_features = self.model.encode_image(x)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                # text_features, _ = self.extract_text_embeddings(classes, [REFERENCE_TEMPLATE], average=True)

            
            similarity = (100.0 * image_features @ text_features.T)
            probas = torch.softmax(similarity, dim=1)
            unary = -torch.log(probas + 1e-10)

        # Choosing affinity matrix for Laplacian optimization
        if self.affinity == 'knn':
            kernel = kNN_affinity(knn=5)(image_features)
        elif self.affinity == 'rbf':
            kernel = rbf_affinity(knn=5)(image_features)
        else:
            kernel = linear_affinity()(image_features)

        # Laplacian optimization (gradient-free)
        logits = self.laplacian_optimization(unary.type(torch.float32), kernel.type(torch.float32))
        logits = logits.softmax(dim=-1)
        #print(logits)
        values, pred = logits.topk(1, 1, True, True)
        #print(pred)
        pred = pred.t()
        return pred

    def laplacian_optimization(
            self,
            unary: Tensor,
            kernel: Tensor,
            bound_lambda: int = 1,
            max_steps: int = 100
    ) -> Tensor:
        E_list = []
        oldE = float('inf')
        Y = (-unary).softmax(-1)  # [N, K]
        for i in range(max_steps):
            pairwise = bound_lambda * kernel.matmul(Y)  # [N, K]
            exponent = -unary + pairwise
            Y = exponent.softmax(-1)
            E = self.entropy_energy(Y, unary, pairwise, bound_lambda).item()
            E_list.append(E)

            if (i > 1 and (abs(E - oldE) <= 1e-8 * abs(oldE))):
                # logger.info(f'Converged in {i} iterations')
                break
            else:
                oldE = E

        return Y

    def entropy_energy(
            self,
            Y: Tensor,
            unary: Tensor,
            pairwise: Tensor,
            bound_lambda: int,
    ) -> Tensor:
        E = (unary * Y - bound_lambda * pairwise * Y + Y * torch.log(Y.clip(1e-20))).sum()

        return E



class AffinityMatrix:
    def __init__(self, **kwargs: Kwargs) -> None:
        pass

    def __call__(X, **kwargs: Kwargs) -> Tensor:
        raise NotImplementedError

    def is_psd(self, mat: Tensor) -> Tuple[Tensor, float]:
        eigenvalues = torch.eig(mat)[0][:, 0].sort(descending=True)[0]

        return eigenvalues, float((mat == mat.t()).all() and (eigenvalues >= 0).all())

    def symmetrize(self, mat: Tensor) -> Tensor:
        return 1 / 2 * (mat + mat.t())


class kNN_affinity(AffinityMatrix):
    def __init__(self, knn: int) -> None:
        self.knn = knn

    def __call__(self, X: Tensor) -> Tensor:
        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
        n_neighbors = min(self.knn + 1, N)

        knn_index = dist.topk(n_neighbors, -1, largest=False).indices[:, 1:]  # [N, knn]

        W = torch.zeros(N, N, device=X.device)
        W.scatter_(dim=-1, index=knn_index, value=1.0)

        return W


class rbf_affinity(AffinityMatrix):
    def __init__(self, **kwargs: Kwargs) -> None:
        self.k = kwargs['knn']

    def __call__(self, X: Tensor) -> Tensor:
        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
        n_neighbors = min(self.k, N)
        kth_dist = dist.topk(k=n_neighbors, dim=-1, largest=False).values[:, -1]  # compute k^th distance for each point, [N, knn + 1]
        sigma = kth_dist.mean()
        rbf = torch.exp(- dist ** 2 / (2 * sigma ** 2))

        return rbf


class linear_affinity(AffinityMatrix):
    def __call__(self, X: Tensor) -> Tensor:
        """
        X: [N, d]
        """
        return torch.matmul(X, X.t())