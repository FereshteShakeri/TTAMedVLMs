import copy
from collections import OrderedDict

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import math

from utils.misc import load_templates_from_yaml


"""
from https://github.com/davda54/sam
"""

import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class SAR:

    def __init__(self, args, base_model, lr, steps=10, margin_e0=0.4*math.log(1000), device='cpu'):
        # base_model, _ = clip.load(model, device)
        self.model = base_model
        self.model_name = args.model
        self.lr = lr
        self.steps = steps
        self.templates = args.templates
        self.margin = margin_e0
        self.ema = None
        self.device = device
        self.model.to(device)

        self.model.visual = self.set_ln_grads(self.model.visual)

        # Collect the LayerNorm parameters and set the optimizer
        params, _ = self.collect_ln_params(self.model.visual)
        base_optimizer = torch.optim.SGD
        self.optimizer = SAM(params, base_optimizer, lr=self.lr, momentum=0.9)

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

        # extracting features
        image_features = self.model.encode_image(x)

        # Pick the top most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
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

        texts = [REFERENCE_TEMPLATE.format(classname.replace('_', ' ')) for classname in classes]
        texts = clip.tokenize(texts).to(self.device)
        text_feat, _ = self.extract_text_embeddings(classes, [REFERENCE_TEMPLATE], average=False)
        text_feat = text_feat.squeeze()

        for _ in range(self.steps):
            self.optimizer.zero_grad()
            # forward
            if self.model_name == "CLIP":
                outputs, _, _ = self.model(x, text_feat, True)
            elif self.model_name == "Quilt":
                outputs, image_features, text_features, _ = self.model(x, text_feat)
            # adapt
            # filtering reliable samples/gradients for further adaptation; first time forward
            entropys = self.softmax_entropy(outputs)
            filter_ids_1 = torch.where(entropys < self.margin)
            entropys = entropys[filter_ids_1]
            loss = entropys.mean(0)
            loss.backward()

            self.optimizer.first_step(
                zero_grad=True)  # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
            if self.model_name == "CLIP":
                entropys2 = self.softmax_entropy(self.model(x, text_feat, True)[0])
            elif self.model_name == "Quilt":
                entropys2 = self.softmax_entropy(self.model(x, text_feat)[0])
            entropys2 = entropys2[filter_ids_1]  # second time forward
            loss_second_value = entropys2.clone().detach().mean(0)
            filter_ids_2 = torch.where(
                entropys2 < self.margin)  # here filtering reliable samples again, since model weights have been changed to \Theta+\hat{\epsilon(\Theta)}
            loss_second = entropys2[filter_ids_2].mean(0)
            if not np.isnan(loss_second.item()):
                self.ema = self.update_ema(self.ema, loss_second.item())  # record moving average loss values for model recovery

            # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
            loss_second.backward()
            self.optimizer.second_step(zero_grad=True)

            # perform model recovery
            reset_flag = False
            if self.ema is not None:
                if self.ema < 0.2:
                    print("ema < 0.2, now reset the model")
                    reset_flag = True

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
            for class_name in class_names:
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
    def update_ema(ema, new_data):
        if ema is None:
            return new_data
        else:
            with torch.no_grad():
                return 0.9 * ema + (1 - 0.9) * new_data

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
