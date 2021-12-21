from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

from lib.modules.linear import AdaptedLinear, _GeneralizedLinear


class LeanFFN(nn.Module):
    """
    A transformer FFN module that doesn't hog your GPU memory.
    Complete with pre-LayerNorm, residual connections and dropout.

    :param gated: use gated activations based on https://arxiv.org/abs/2002.05202 and https://arxiv.org/abs/2102.11972
      note: gated activations require 1.5x more parameters compared to their non-gated variants.
    :param sandwich_norm: use an additional layer normalization after the two linear layers, before residual
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation=F.gelu,
        gated: bool = False,
        layer_norm_eps: float = 1e-12,
        dropout: float = 0.0,
        sandwich_norm: bool = False,
        dense_i2h: Optional[nn.Linear] = None,
        dense_h2o: Optional[nn.Linear] = None,
    ):
        super().__init__()
        i2h_out_features = intermediate_size * 2 if gated else intermediate_size
        self.dense_i2h = nn.Linear(hidden_size, i2h_out_features) if dense_i2h is None else dense_i2h
        self.dense_h2o = nn.Linear(intermediate_size, hidden_size) if dense_h2o is None else dense_h2o
        assert self.dense_i2h.in_features == self.dense_h2o.out_features == hidden_size
        assert self.dense_i2h.out_features == i2h_out_features and self.dense_h2o.in_features == intermediate_size
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.sandwich_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps) if sandwich_norm is not None else None
        self.activation = activation
        self.dropout = dropout

    def forward(self, input):
        sandwich_ln_weight = sandwich_ln_bias = None
        if self.sandwich_norm is not None:
            sandwich_ln_weight, sandwich_ln_bias = self.sandwich_norm.weight, self.sandwich_norm.bias
        i2h_adapter_first = i2h_adapter_second = h2o_adapter_first = h2o_adapter_second = None
        if isinstance(self.dense_i2h, AdaptedLinear):
            i2h_adapter_first, i2h_adapter_second = self.dense_i2h.adapter_first, self.dense_i2h.adapter_second
        if isinstance(self.dense_h2o, AdaptedLinear):
            h2o_adapter_first, h2o_adapter_second = self.dense_h2o.adapter_first, self.dense_h2o.adapter_second

        output = _LeanFFN.apply(
            input,
            self.layer_norm.weight,
            self.layer_norm.bias,
            self.dense_i2h.weight,
            self.dense_i2h.bias,
            i2h_adapter_first,
            i2h_adapter_second,
            self.dense_h2o.weight,
            self.dense_h2o.bias,
            h2o_adapter_first,
            h2o_adapter_second,
            sandwich_ln_weight,
            sandwich_ln_bias,
            self.activation,
            self.dropout,
            self.training,
            self.layer_norm.eps,
        )
        return output


class _LeanFFN(torch.autograd.Function):
    @staticmethod
    def _apply_activation(pre_activation: torch.Tensor, activation: callable, hid_size: int):
        if pre_activation.shape[-1] == hid_size:
            return activation(pre_activation)
        elif pre_activation.shape[-1] == 2 * hid_size:
            pre_gate, lin = pre_activation.split(pre_activation.shape[-1] // 2, dim=-1)
            return activation(pre_gate).mul_(lin)
        else:
            raise RuntimeError("The output size of FFN layer must be either 1x or 2x the intermediate_size.")

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input,
        ln_weight,
        ln_bias,
        i2h_weight,
        i2h_bias,
        i2h_adapter_first,
        i2h_adapter_second,
        h2o_weight,
        h2o_bias,
        h2o_adapter_first,
        h2o_adapter_second,
        sandwich_ln_weight,
        sandwich_ln_bias,
        activation,
        dropout,
        training,
        ln_eps,
    ):
        ctx._activation, ctx._dropout, ctx._training, ctx._ln_eps = activation, dropout, training, ln_eps
        ctx._use_sandwich = sandwich_ln_weight is not None
        ctx._intermediate_size = h2o_weight.shape[1]
        dropout_mask, pre_sandwich = None, None  # optional tensors to save

        input = input.to(torch.float32, copy=False)  # accumulate residuals to fp32; no-op if already fp32
        input_2d = input.view(-1, input.shape[-1])

        input_ln = F.layer_norm(input_2d, input.shape[-1:], ln_weight, ln_bias, ln_eps)

        pre_activation, i2h_tensors = _GeneralizedLinear._forward_impl(
            input_ln, i2h_weight, i2h_bias, i2h_adapter_first, i2h_adapter_second
        )

        hid_act = _LeanFFN._apply_activation(pre_activation, ctx._activation, h2o_weight.shape[1])

        out, h2o_tensors = _GeneralizedLinear._forward_impl(
            hid_act, h2o_weight, h2o_bias, h2o_adapter_first, h2o_adapter_second
        )

        if ctx._use_sandwich:
            pre_sandwich = out
            out = F.layer_norm(pre_sandwich, pre_sandwich.shape[-1:], sandwich_ln_weight, sandwich_ln_bias, eps=ln_eps)

        out = F.dropout(out, dropout, training, inplace=True)
        if training and dropout:
            dropout_mask = (out == 0.0).to(torch.int8)

        out = out.add_(input_2d) if out.dtype == input_2d.dtype else torch.add(input_2d, out)

        assert i2h_tensors[0] is input_ln and h2o_tensors[0] is hid_act  # we can rematerialize these tensors
        tensors_to_save = [
            input,
            pre_activation,
            ln_weight,
            ln_bias,
            pre_sandwich,
            sandwich_ln_weight,
            sandwich_ln_bias,
            dropout_mask,
        ]
        tensors_to_save.extend((*i2h_tensors[1:], *h2o_tensors[1:]))
        ctx.save_for_backward(*tensors_to_save)
        ctx._num_i2h_tensors = len(i2h_tensors)
        ctx._num_h2o_tensors = len(h2o_tensors)
        return out.view(*input.shape)

    @staticmethod
    def _h2o_backward(ctx, grad_output: torch.Tensor, hid_act: torch.Tensor):
        h2o_tensors = ctx.saved_tensors[-ctx._num_h2o_tensors + 1 :]
        needs_input_grad = (hid_act.requires_grad, *ctx.needs_input_grad[7:11])
        return _GeneralizedLinear._backward_impl(grad_output, hid_act, *h2o_tensors, needs_input_grad=needs_input_grad)

    @staticmethod
    def _i2h_backward(ctx, grad_output: torch.Tensor, input_ln: torch.Tensor):
        i2h_tensors = ctx.saved_tensors[-ctx._num_i2h_tensors - ctx._num_h2o_tensors + 2 : -ctx._num_h2o_tensors + 1]
        needs_input_grad = (input_ln.requires_grad, *ctx.needs_input_grad[3:7])
        return _GeneralizedLinear._backward_impl(grad_output, input_ln, *i2h_tensors, needs_input_grad=needs_input_grad)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        grad_input = grad_ln_weight = grad_ln_bias = grad_sandwich_ln_weight = grad_sandwich_ln_bias = None
        (
            input,
            pre_activation,
            ln_weight,
            ln_bias,
            pre_sandwich,
            sandwich_ln_weight,
            sandwich_ln_bias,
            dropout_mask,
            *_,
        ) = ctx.saved_tensors
        grad_output_2d = grad_output.view(-1, grad_output.shape[-1])

        # backward(... -> sandwich_norm -> dropout -> residual)
        grad_residual_2d = grad_output_2d
        if dropout_mask is not None:
            grad_output_2d = grad_output_2d.mul(dropout_mask.to(grad_output_2d.dtype))
        if ctx._use_sandwich:
            assert pre_sandwich is not None
            with torch.enable_grad():
                required_grad = pre_sandwich.requires_grad
                pre_sandwich.requires_grad_(True)
                sandwich_out = F.layer_norm(
                    pre_sandwich, pre_sandwich.shape[-1:], sandwich_ln_weight, sandwich_ln_bias, eps=ctx._ln_eps
                )
                grad_output, grad_sandwich_ln_weight, grad_sandwich_ln_bias = torch.autograd.grad(
                    sandwich_out, [pre_sandwich, sandwich_ln_weight, sandwich_ln_bias], grad_outputs=grad_output_2d
                )
                pre_sandwich.requires_grad_(required_grad)
                del pre_sandwich

        # backward(... -> nonlinearity -> intermediate_layernorm -> linear_h2o -> ...)
        input_2d = input.view(-1, input.shape[-1])
        grad_h2o_output_2d = grad_output.view(-1, grad_output.shape[-1])

        with torch.enable_grad():
            # rematerialize activation
            pre_activation.requires_grad_(True)
            hid_act = _LeanFFN._apply_activation(pre_activation, ctx._activation, ctx._intermediate_size)

            with torch.no_grad():
                (
                    grad_hid_act,
                    grad_h2o_weight,
                    grad_h2o_bias,
                    grad_h2o_adapter_first,
                    grad_h2o_adapter_second,
                ) = _LeanFFN._h2o_backward(ctx, grad_h2o_output_2d, hid_act)

            (grad_hid,) = torch.autograd.grad(hid_act, pre_activation, grad_outputs=grad_hid_act)
            pre_activation.requires_grad_(False)

        # backward(... -> input_layernorm -> liner_i2h -> ...)
        with torch.enable_grad():
            # rematerialize input_ln
            input_2d.requires_grad_(True)
            input_ln_2d = F.layer_norm(input_2d, input.shape[-1:], ln_weight, ln_bias, ctx._ln_eps)

            with torch.no_grad():
                (
                    grad_input_ln_2d,
                    grad_i2h_weight,
                    grad_i2h_bias,
                    grad_i2h_adapter_first,
                    grad_i2h_adapter_second,
                ) = _LeanFFN._i2h_backward(ctx, grad_hid, input_ln_2d)

            if any(ctx.needs_input_grad[0:3]):
                partial_grad_input_2d, grad_ln_weight, grad_ln_bias = torch.autograd.grad(
                    outputs=input_ln_2d, inputs=[input_2d, ln_weight, ln_bias], grad_outputs=grad_input_ln_2d
                )

        # add up residual grads
        if ctx.needs_input_grad[0]:
            grad_input = partial_grad_input_2d.add_(grad_residual_2d).view(*input.shape)

        return (
            grad_input,
            grad_ln_weight,
            grad_ln_bias,
            grad_i2h_weight,
            grad_i2h_bias,
            grad_i2h_adapter_first,
            grad_i2h_adapter_second,
            grad_h2o_weight,
            grad_h2o_bias,
            grad_h2o_adapter_first,
            grad_h2o_adapter_second,
            grad_sandwich_ln_weight,
            grad_sandwich_ln_bias,
            None,
            None,
            None,
            None,
        )
