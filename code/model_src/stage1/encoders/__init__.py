

from typing import Callable, Dict, Optional, Type, Union

ARCHS: Dict[str, Type] = {}
__all__ = ["ARCHS", "register_encoder"]


def _add_to_registry(name: str, cls: Type) -> Type:
    if name in ARCHS and ARCHS[name] is not cls:
        raise ValueError(f"Encoder '{name}' is already registered.")
    ARCHS[name] = cls
    return cls


def register_encoder(cls: Optional[Type] = None, *, name: Optional[str] = None) -> Union[Callable[[Type], Type], Type]:
    

    def decorator(inner_cls: Type) -> Type:
        encoder_name = name or inner_cls.__name__
        return _add_to_registry(encoder_name, inner_cls)

    if cls is None:
        return decorator

    return decorator(cls)



from . import dinov2  
