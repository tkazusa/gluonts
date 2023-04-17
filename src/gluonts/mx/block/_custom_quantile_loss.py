# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import List, Optional

from mxnet.gluon import nn
from mxnet.gluon.loss import Loss

from gluonts.core.component import validated
from gluonts.mx import Tensor


class CustomQuantileLoss(Loss):
    @validated()
    def __init__(
        self,
        quantiles: List[float],
        quantile_weights: Optional[List[float]] = None,
        weight: Optional[float] = None,
        batch_axis: int = 0,
        **kwargs,
    ) -> None:
        """
        Represent the quantile loss used to fit decoders that learn quantiles.

        Parameters
        ----------
        quantiles
            list of quantiles to compute loss over.
        quantile_weights
            weights of the quantiles.
        weight
            weighting of the loss.
        batch_axis
            indicates axis that represents the batch.
        """
        assert len(quantiles) > 0

        super().__init__(weight, batch_axis, **kwargs)

        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.quantile_weights = (
            quantile_weights
            if quantile_weights is not None
            else uniform_weights(quantiles)
        )

    def hybrid_forward(self, F, y_true: Tensor, y_pred: Tensor, sample_weight=None):
        """
        Compute the weighted sum of quantile losses.

        Parameters
        ----------
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.
        y_true
            ground truth values, shape (N1 x N2 x ... x Nk)
        y_pred
            predicted target, shape (N1 x N2 x ... x Nk x num_quantiles)
        sample_weight
            sample weights

        Returns
        -------
        Tensor
            weighted sum of the quantile losses, shape N1 x N1 x ... Nk
        """
        print("----------------chinko chinko chinko------------")
        print(F)
        print(y_true)
        print(y_pred)
        if self.num_quantiles > 1:
            y_pred_all = F.split(
                y_pred, axis=-1, num_outputs=self.num_quantiles, squeeze_axis=1
            )
        else:
            y_pred_all = [F.squeeze(y_pred, axis=-1)]

        print(y_pred_all)

        qt_loss = []
        for level, weight, y_pred_q in zip(
            self.quantiles, self.quantile_weights, y_pred_all
        ):
            print(level, weight, y_pred_q)
            qt_loss.append(
                weight * self.compute_quantile_loss(F, y_true, y_pred_q, level)
            )
        stacked_qt_losses = F.stack(*qt_loss, axis=-1)
        sum_qt_loss = F.mean(stacked_qt_losses, axis=-1)  # avg across quantiles
        if sample_weight is not None:
            return sample_weight * sum_qt_loss
        return sum_qt_loss

    @staticmethod
    def compute_quantile_loss(F, y_true: Tensor, y_pred_p: Tensor, p: float) -> Tensor:
        """
        Compute the quantile loss of the given quantile.

        Parameters
        ----------
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.
        y_true
            ground truth values to compute the loss against.
        y_pred_p
            predicted target quantile, same shape as ``y_true``.
        p
            quantile error to compute the loss.

        Returns
        -------
        Tensor
            quantile loss, shape: (N1 x N2 x ... x Nk x 1)
        """

        under_bias = p * F.maximum(y_true - y_pred_p, 0)
        over_bias = (1 - p) * F.maximum(y_pred_p - y_true, 0)

        qt_loss = 2 * (under_bias + over_bias)

        return qt_loss
