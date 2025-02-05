from typing import Callable

import torch
import torch.nn as nn
from captum.attr import LayerIntegratedGradients, LayerGradientXActivation, LayerFeatureAblation, LayerGradientShap, \
    LayerGradCam
from captum.attr import visualization as viz

from transformers_interpret.errors import AttributionsNotCalculatedError


class Attributions:
    def __init__(self, custom_forward: Callable, embeddings: nn.Module, tokens: list):
        self.custom_forward = custom_forward
        self.embeddings = embeddings
        self.tokens = tokens


class LIGAttributions(Attributions):
    def __init__(
        self,
        custom_forward: Callable,
        embeddings: nn.Module,
        tokens: list,
        input_ids: torch.Tensor,
        ref_input_ids: torch.Tensor,
        sep_id: int,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        ref_token_type_ids: torch.Tensor = None,
        ref_position_ids: torch.Tensor = None,
        internal_batch_size: int = None,
        n_steps: int = 50,
        target_idx: int = None,
    ):
        super().__init__(custom_forward, embeddings, tokens)
        self.input_ids = input_ids
        self.ref_input_ids = ref_input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        self.ref_token_type_ids = ref_token_type_ids
        self.ref_position_ids = ref_position_ids
        self.internal_batch_size = internal_batch_size
        self.n_steps = n_steps

        self.lig = LayerIntegratedGradients(self.custom_forward, self.embeddings)
        # print('self.ref_input_ids', self.ref_input_ids)
        # print('self.ref_token_type_ids', self.ref_token_type_ids)
        # print('self.ref_position_ids', self.ref_position_ids)
        # print('self.input_ids', self.input_ids)
        # print('self.token_type_ids', self.token_type_ids)
        # print('self.position_ids', self.position_ids)

        # print('target', target_idx)

        # print('case', self.token_type_ids is not None, self.position_ids is not None)
        if self.token_type_ids is not None and self.position_ids is not None:
            self._attributions, self.delta = self.lig.attribute(
                inputs=(self.input_ids, self.token_type_ids, self.position_ids),
                baselines=(
                    self.ref_input_ids,
                    self.ref_token_type_ids,
                    self.ref_position_ids,
                ),
                # target=[target_idx] if target_idx else None,
                return_convergence_delta=True,
                additional_forward_args=(self.attention_mask),
                internal_batch_size=self.internal_batch_size,
                n_steps=self.n_steps,
            )
        elif self.position_ids is not None:
            self._attributions, self.delta = self.lig.attribute(
                inputs=(self.input_ids, self.position_ids),
                baselines=(
                    self.ref_input_ids,
                    self.ref_position_ids,
                ),
                return_convergence_delta=True,
                additional_forward_args=(self.attention_mask),
                internal_batch_size=self.internal_batch_size,
                n_steps=self.n_steps,
            )
        elif self.token_type_ids is not None:
            self._attributions, self.delta = self.lig.attribute(
                inputs=(self.input_ids, self.token_type_ids),
                baselines=(
                    self.ref_input_ids,
                    self.ref_token_type_ids,
                ),
                return_convergence_delta=True,
                additional_forward_args=(self.attention_mask),
                internal_batch_size=self.internal_batch_size,
                n_steps=self.n_steps,
            )

        else:
            self._attributions, self.delta = self.lig.attribute(
                inputs=self.input_ids,
                baselines=self.ref_input_ids,
                return_convergence_delta=True,
                internal_batch_size=self.internal_batch_size,
                n_steps=self.n_steps,
            )

        # print('len(self._attributions)', len(self._attributions[0]), 'len(self.input_ids)', len(self.input_ids[0]), 'self.ref_input_ids', len(self.ref_input_ids[0]))

    @property
    def word_attributions(self) -> list:
        wa = []
        if len(self.attributions_sum) >= 1:
            for i, (word, attribution) in enumerate(zip(self.tokens, self.attributions_sum)):
                wa.append((word, float(attribution.cpu().data.numpy())))
            return wa

        else:
            raise AttributionsNotCalculatedError("Attributions are not yet calculated")

    def summarize(self, end_idx=None):
        self.attributions_sum = self._attributions.sum(dim=-1).squeeze(0)
        self.attributions_sum = self.attributions_sum[:end_idx] / torch.norm(self.attributions_sum[:end_idx])

    def visualize_attributions(self, pred_prob, pred_class, true_class, attr_class, all_tokens):

        return viz.VisualizationDataRecord(
            self.attributions_sum,
            pred_prob,
            pred_class,
            true_class,
            attr_class,
            self.attributions_sum.sum(),
            all_tokens,
            self.delta,
        )


class LGXAAttributions(Attributions):
    def __init__(
            self,
            custom_forward: Callable,
            embeddings: nn.Module,
            tokens: list,
            input_ids: torch.Tensor,
            ref_input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor = None,
            position_ids: torch.Tensor = None,
            internal_batch_size: int = None,
    ):
        super().__init__(custom_forward, embeddings, tokens)
        self.input_ids = input_ids
        self.ref_input_ids = ref_input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        self.internal_batch_size = internal_batch_size

        self.attributor = LayerGradientXActivation(self.custom_forward, self.embeddings)

        # print('case', self.token_type_ids is not None, self.position_ids is not None)
        if self.token_type_ids is not None and self.position_ids is not None:
            self._attributions = self.attributor.attribute(
                inputs=(self.input_ids, self.token_type_ids, self.position_ids),
                # target=[target_idx] if target_idx else None,
                additional_forward_args=(self.attention_mask),
            )
        elif self.position_ids is not None:
            self._attributions = self.attributor.attribute(
                inputs=(self.input_ids, self.position_ids),
                additional_forward_args=(self.attention_mask),
            )
        elif self.token_type_ids is not None:
            self._attributions = self.attributor.attribute(
                inputs=(self.input_ids, self.token_type_ids),
                additional_forward_args=(self.attention_mask),
            )

        else:
            self._attributions = self.attributor.attribute(
                inputs=self.input_ids,
            )

    @property
    def word_attributions(self) -> list:
        wa = []
        if len(self.attributions_sum) >= 1:
            for i, (word, attribution) in enumerate(zip(self.tokens, self.attributions_sum)):
                wa.append((word, float(attribution.cpu().data.numpy())))
            return wa

        else:
            raise AttributionsNotCalculatedError("Attributions are not yet calculated")

    def summarize(self, end_idx=None):
        self.attributions_sum = self._attributions.sum(dim=-1).squeeze(0)
        self.attributions_sum = self.attributions_sum[:end_idx] / torch.norm(self.attributions_sum[:end_idx])

    def visualize_attributions(self, pred_prob, pred_class, true_class, attr_class, all_tokens):

        return viz.VisualizationDataRecord(
            self.attributions_sum,
            pred_prob,
            pred_class,
            true_class,
            attr_class,
            self.attributions_sum.sum(),
            all_tokens,
            0,
        )


class LFAAttributions(Attributions):
    def __init__(
            self,
            custom_forward: Callable,
            embeddings: nn.Module,
            tokens: list,
            input_ids: torch.Tensor,
            ref_input_ids: torch.Tensor,
            sep_id: int,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor = None,
            position_ids: torch.Tensor = None,
            ref_token_type_ids: torch.Tensor = None,
            ref_position_ids: torch.Tensor = None,
            internal_batch_size: int = None,
            n_steps: int = 50,
            target_idx: int = None,
    ):
        super().__init__(custom_forward, embeddings, tokens)
        self.input_ids = input_ids
        self.ref_input_ids = ref_input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        self.ref_token_type_ids = ref_token_type_ids
        self.ref_position_ids = ref_position_ids
        self.internal_batch_size = internal_batch_size
        self.n_steps = n_steps

        self.attributor = LayerFeatureAblation(self.custom_forward, self.embeddings)

        # print('case', self.token_type_ids is not None, self.position_ids is not None)
        if self.token_type_ids is not None and self.position_ids is not None:
            self._attributions = self.attributor.attribute(
                inputs=(self.input_ids, self.token_type_ids, self.position_ids),
                additional_forward_args=(self.attention_mask),
            )
        elif self.position_ids is not None:
            self._attributions = self.attributor.attribute(
                inputs=(self.input_ids, self.position_ids),
                additional_forward_args=(self.attention_mask),
            )
        elif self.token_type_ids is not None:
            self._attributions = self.attributor.attribute(
                inputs=(self.input_ids, self.token_type_ids),
                additional_forward_args=(self.attention_mask),
            )

        else:
            self._attributions = self.attributor.attribute(
                inputs=self.input_ids,
            )

    @property
    def word_attributions(self) -> list:
        wa = []
        if len(self.attributions_sum) >= 1:
            for i, (word, attribution) in enumerate(zip(self.tokens, self.attributions_sum)):
                wa.append((word, float(attribution.cpu().data.numpy())))
            return wa

        else:
            raise AttributionsNotCalculatedError("Attributions are not yet calculated")

    def summarize(self, end_idx=None):
        self.attributions_sum = self._attributions.sum(dim=-1).squeeze(0)
        self.attributions_sum = self.attributions_sum[:end_idx] / torch.norm(self.attributions_sum[:end_idx])

    def visualize_attributions(self, pred_prob, pred_class, true_class, attr_class, all_tokens):

        return viz.VisualizationDataRecord(
            self.attributions_sum,
            pred_prob,
            pred_class,
            true_class,
            attr_class,
            self.attributions_sum.sum(),
            all_tokens,
            0,
        )


class GradCamAttributions(Attributions):
    def __init__(self,
                 custom_forward: Callable,
                 embeddings: nn.Module,
                 tokens: list,
                 input_ids: torch.Tensor,
                 ref_input_ids: torch.Tensor,
                 attention_mask: torch.Tensor,
                 token_type_ids: torch.Tensor = None,
                 position_ids: torch.Tensor = None,
                 ref_token_type_ids: torch.Tensor = None,
                 ref_position_ids: torch.Tensor = None,
                 internal_batch_size: int = None,
                 ):
        super().__init__(custom_forward, embeddings, tokens)
        self.input_ids = input_ids
        self.ref_input_ids = ref_input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        self.ref_token_type_ids = ref_token_type_ids
        self.ref_position_ids = ref_position_ids
        self.internal_batch_size = internal_batch_size

        self.attributor = LayerGradCam(self.custom_forward, self.embeddings)

        # print('case', self.token_type_ids is not None, self.position_ids is not None)
        if self.token_type_ids is not None and self.position_ids is not None:
            self._attributions = self.attributor.attribute(
                inputs=(self.input_ids, self.token_type_ids, self.position_ids),
                additional_forward_args=(self.attention_mask),
            )
        elif self.position_ids is not None:
            self._attributions = self.attributor.attribute(
                inputs=(self.input_ids, self.position_ids),
                additional_forward_args=(self.attention_mask),
            )
        elif self.token_type_ids is not None:
            self._attributions = self.attributor.attribute(
                inputs=(self.input_ids, self.token_type_ids),
                additional_forward_args=(self.attention_mask),
            )

        else:
            self._attributions = self.attributor.attribute(
                inputs=self.input_ids,
            )

    @property
    def word_attributions(self) -> list:
        wa = []
        if len(self.attributions_sum) >= 1:
            for i, (word, attribution) in enumerate(zip(self.tokens, self.attributions_sum)):
                wa.append((word, float(attribution.cpu().data.numpy())))
            return wa

        else:
            raise AttributionsNotCalculatedError("Attributions are not yet calculated")

    def summarize(self, end_idx=None):
        self.attributions_sum = self._attributions.sum(dim=-2).squeeze(0)
        self.attributions_sum = self.attributions_sum[:end_idx] / torch.norm(self.attributions_sum[:end_idx])

    def visualize_attributions(self, pred_prob, pred_class, true_class, attr_class, all_tokens):

        return viz.VisualizationDataRecord(
            self.attributions_sum,
            pred_prob,
            pred_class,
            true_class,
            attr_class,
            self.attributions_sum.sum(),
            all_tokens,
            0,
        )