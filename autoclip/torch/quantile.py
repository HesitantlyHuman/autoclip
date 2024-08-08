from typing import Iterator, List, Dict, Union, Any
import torch

from autoclip.torch.clipper import Clipper, OptimizerWithClipping


class QuantileClip(Clipper):
    def __init__(
        self,
        parameters: Iterator[torch.nn.parameter.Parameter],
        quantile: float = 0.9,
        history_length: int = 1000,
        global_threshold: bool = False,
        lr_regularize: bool = False,
    ) -> None:
        self.global_threshold = global_threshold
        self.global_quantile = None
        self.global_history_length = None
        if self.global_threshold:
            self.global_quantile = quantile
            self.global_history_length = history_length

        if lr_regularize:
            raise ValueError(
                "Learning rate regularization can only be used if you wrap an optimizer. Try using `QuantileClip.as_optimizer` instead."
            )

        self._lr_regularizes = None

        super().__init__(
            parameters,
            {"quantile": quantile, "history_length": history_length},
        )

    @classmethod
    def as_optimizer(
        cls: "QuantileClip",
        optimizer: torch.optim.Optimizer,
        quantile: float = 0.9,
        history_length: int = 1000,
        global_threshold: bool = False,
        lr_regularize: bool = True,
    ) -> "OptimizerWithClipping":
        return super().as_optimizer(
            optimizer,
            quantile=quantile,
            history_length=history_length,
            global_threshold=global_threshold,
            lr_regularize=lr_regularize,
        )

    def verify_parameter_settings(self, settings: Dict[str, Any]) -> None:
        quantile = settings["quantile"]
        history_length = settings["history_length"]
        if not isinstance(quantile, (float, torch.Tensor)):
            raise TypeError(
                "`QuantileClip` quantile value must be a float or a tensor."
            )
        if not isinstance(history_length, int):
            raise TypeError("`QuantileClip` history_length must be an int.")
        if quantile < 0.0 or quantile > 1.0:
            raise ValueError(
                "`QuantileClip` quantile value must be between 0.0 and 1.0."
            )
        if history_length <= 0:
            raise ValueError("`QuantileClip` history length must be greater than zero.")

    def step(self, param_group_learning_rates: List[torch.Tensor] = None) -> None:
        if self._lr_regularizes is None:
            self._lr_regularizes = not param_group_learning_rates is None
        if self._lr_regularizes and param_group_learning_rates is None:
            raise ValueError(
                "`QuantileClip` history is regularized, but `step` did not receive `param_group_learning_rates`"
            )
        elif not self._lr_regularizes and not param_group_learning_rates is None:
            raise ValueError(
                "`QuantileClip` history is not regularized, but `step` received `param_group_learning_rates`"
            )
        if self.global_threshold:
            self._clip_global(param_group_learning_rates)
        else:
            self._clip_local(param_group_learning_rates)

    def _clip_local(self, param_group_learning_rates: List[torch.Tensor] = None):
        if param_group_learning_rates is None:
            param_group_learning_rates = [None for _ in self.param_groups]
        for param_group, param_group_learning_rate in zip(
            self.param_groups, param_group_learning_rates
        ):
            group_quantile = param_group["quantile"]
            group_history_length = param_group["history_length"]

            for parameter in param_group["params"]:
                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                if len(state) == 0:
                    state["history"] = torch.Tensor([]).to(parameter.device)
                    threshold = torch.inf
                else:
                    threshold = torch.quantile(state["history"], group_quantile)
                    if param_group_learning_rate is not None:
                        threshold = threshold * param_group_learning_rate
                old_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameter, max_norm=threshold
                )
                if param_group_learning_rate is not None:
                    old_grad_norm = old_grad_norm / param_group_learning_rate
                state["history"] = torch.hstack((state["history"], old_grad_norm))[
                    -group_history_length:
                ]

    def _clip_global(self, param_group_learning_rates: List[torch.Tensor] = None):
        parameters = []
        for param_group in self.param_groups:
            parameters = parameters + param_group["params"]

        if len(self.state["global_history"]) == 0:
            # Assumes all parameters are on the same device
            self.state["global_history"] = torch.Tensor([]).to(parameters[0].device)
            threshold = torch.inf
        else:
            threshold = torch.quantile(
                self.state["global_history"], self.global_quantile
            )

        if param_group_learning_rates is None:
            old_grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, max_norm=threshold
            )
        else:
            old_grad_norm_sum = 0
            for param_group, param_group_learning_rate in zip(
                self.param_groups, param_group_learning_rates
            ):
                parameters = param_group["params"]
                param_group_threshold = param_group_learning_rate * threshold
                old_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, max_norm=param_group_threshold
                )
                old_grad_norm = old_grad_norm / param_group_learning_rate
                old_grad_norm_sum += old_grad_norm
            old_grad_norm = old_grad_norm_sum / len(self.param_groups)

        self.state["global_history"] = torch.hstack(
            (self.state["global_history"], old_grad_norm)
        )[-self.global_history_length :]

    def add_param_group(
        self,
        param_group: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        quantile: float = None,
        history_length: int = None,
    ) -> None:
        param_group_args = {}
        if quantile is not None:
            param_group_args["quantile"] = quantile
        if history_length is not None:
            param_group_args["history_length"] = history_length
        return super().add_param_group(param_group, **param_group_args)
