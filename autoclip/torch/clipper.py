from typing import Callable, Iterator, Iterable, Dict, Mapping, Union, Any, List

import torch
from copy import deepcopy
from itertools import chain
from collections import defaultdict
from autoclip.torch.utils import deep_tensor_move


class Clipper:
    """
    Modeled after torch.optim.Optimizer
    """

    def __init__(
        self,
        parameters: Iterator[torch.nn.parameter.Parameter],
        defaults: Dict[str, Any],
    ) -> None:
        self.param_groups: List[Dict[str, torch.Tensor]] = []
        self.verify_parameter_settings(settings=defaults)
        self.defaults = defaults
        self.state = defaultdict(dict)

        if not isinstance(parameters, (Iterator, Iterable)):
            raise TypeError(
                "parameters argument given to the clipper should be "
                "an iterable of Tensors or dicts, but instead got "
                + torch.typename(parameters)
            )

        param_groups = list(parameters)
        if len(param_groups) == 0:
            raise ValueError(
                f"Clipper {type(self).__name__} got an empty parameter list"
            )
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group=param_group)

    @classmethod
    def as_optimizer(
        cls: "Clipper",
        optimizer: torch.optim.Optimizer,
        lr_regularize: bool = True,
        **kwargs,
    ) -> "OptimizerWithClipping":
        parameters = chain.from_iterable(
            [param_group["params"] for param_group in optimizer.param_groups]
        )
        clipper = cls(parameters=parameters, **kwargs)
        return OptimizerWithClipping(
            optimizer=optimizer, clipper=clipper, lr_regularize=lr_regularize
        )

    def verify_parameter_settings(self, settings: Dict[str, Any]) -> None:
        raise NotImplementedError()

    def step(self, param_group_learning_rates: List[torch.Tensor] = None) -> None:
        raise NotImplementedError()

    def state_dict(self) -> Dict[str, Any]:
        packed_param_groups = []
        for param_group in self.param_groups:
            packed_param_group = {
                key: value for key, value in param_group.items() if key != "params"
            }
            packed_param_group["params"] = [
                id(parameter) for parameter in param_group["params"]
            ]
            packed_param_groups.append(packed_param_group)

        packed_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }

        return {
            "state": packed_state,
            "param_groups": packed_param_groups,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        loaded_state_dict = deepcopy(state_dict)
        local_groups, saved_groups = (
            self.param_groups,
            loaded_state_dict["param_groups"],
        )

        if len(local_groups) != len(saved_groups):
            raise ValueError(
                f"Loaded state dict has {len(saved_groups)} parameter "
                f"groups, Clipper {type(self).__name__} has "
                f"{len(local_groups)} parameter groups"
            )
        local_lens = (len(g["params"]) for g in local_groups)
        saved_lens = (len(g["params"]) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(local_lens, saved_lens)):
            raise ValueError(
                "Loaded state dict contains a parameter group "
                "that doesn't match the size of Clipper "
                f"{type(self).__name__}'s group"
            )

        saved_id_to_parameter = {
            saved_id: parameter
            for saved_id, parameter in zip(
                chain.from_iterable([group["params"] for group in saved_groups]),
                chain.from_iterable([group["params"] for group in local_groups]),
            )
        }

        state = defaultdict(dict)
        for key, value in loaded_state_dict["state"].items():
            if key in saved_id_to_parameter:
                parameter = saved_id_to_parameter[key]
                state[parameter] = deep_tensor_move(value, parameter.device)
            else:
                state[key] = value

        new_param_groups = []
        for local_group, saved_group in zip(local_groups, saved_groups):
            saved_group["params"] = local_group["params"]
            new_param_groups.append(saved_group)

        self.state = state
        self.param_groups = new_param_groups

    def add_param_group(
        self,
        param_group: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        **kwargs,
    ) -> None:
        """Add a param_group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group
            specific optimization options.
        """
        if not isinstance(param_group, Mapping):
            param_group = {"params": param_group}

        parameters = param_group["params"]
        if isinstance(parameters, torch.Tensor):
            param_group["params"] = [parameters]
        elif isinstance(parameters, set):
            raise TypeError(
                "Clipping parameters must be ordered collections. "
                "The ordering of tensors in sets will change between runs."
                "Please use a list instead."
            )
        else:
            param_group["params"] = list(parameters)

        for parameter in param_group["params"]:
            if not isinstance(parameter, torch.Tensor):
                raise TypeError(
                    f"Clipper {type(self).__name__} can only clip Tensors, "
                    f"but one of the params is {torch.typename(parameter)}"
                )
            if not parameter.is_leaf:
                raise ValueError(
                    "Gradients to clip will only accumulate on leaf Tensors. "
                    f"{type(self).__name__} recieved non-leaf Tensor."
                )

        for name, default in self.defaults.items():
            param_group.setdefault(name, default)
        param_group.update(kwargs)
        self.verify_parameter_settings(param_group)

        parameters = param_group["params"]
        if len(parameters) != len(set(parameters)):
            raise ValueError(
                "Clipper contains a parameter group with duplicate parameters."
            )

        parameter_set = set()
        for group in self.param_groups:
            parameter_set.update(set(group["params"]))

        if not parameter_set.isdisjoint(set(param_group["params"])):
            raise ValueError(
                "Some clipping parameters appear in more than one parameter group"
            )

        self.param_groups.append(param_group)

    def __repr__(self):
        format_string = self.__class__.__name__ + " ("
        for i, group in enumerate(self.param_groups):
            format_string += "\n"
            format_string += "Parameter Group {0}\n".format(i)
            for key in sorted(group.keys()):
                if key != "params":
                    format_string += "    {0}: {1}\n".format(key, group[key])
        format_string += ")"
        return format_string


class OptimizerWithClipping(torch.optim.Optimizer):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        clipper: Clipper,
        lr_regularize: bool = True,
    ) -> None:
        self.optimizer = optimizer
        self.clipper = clipper
        self.lr_regularize = lr_regularize

    def step(
        self, closure: Union[Callable[[], float], None] = None
    ) -> Union[float, None]:
        if self.lr_regularize:
            param_group_learning_rates = [
                param_group["lr"] for param_group in self.optimizer.param_groups
            ]
            self.clipper.step(param_group_learning_rates=param_group_learning_rates)
        else:
            self.clipper.step()
        return self.optimizer.step(closure=closure)

    def add_param_group(
        self, param_group: Dict[str, Union[torch.Tensor, List[torch.Tensor]]], **kwargs
    ) -> None:
        self.optimizer.add_param_group(param_group=param_group)
        self.clipper.add_param_group(param_group=param_group, **kwargs)

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def __getstate__(self):
        return {
            "optimizer": self.optimizer,
            "clipper": self.clipper,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.optimizer._hook_for_profile()

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state dict of the optimizer and clipper as a :class:`dict`.

        It contains two entries:

        * optimizer - a dict holding the optimizer state dict. It will conain both
            state and param_groups, as described in the :class:`torch.optim.Optimizer` docs.
        * clipper - a dict holding the clipper state dict. It will contain its ow
            state and param_groups, as described in the :class:`autoclip.torch.Clipper` docs.
        """
        return {
            "optimizer": self.optimizer.state_dict(),
            "clipper": self.clipper.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict=state_dict["optimizer"])
        self.clipper.load_state_dict(state_dict=state_dict["clipper"])

    def __repr__(self) -> str:
        return f"OptimizerWithClipping (\n{self.optimizer}\n{self.clipper})"

    def __getattr__(self, attr):
        return getattr(self.optimizer, attr)
