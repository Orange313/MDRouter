from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self) -> None:
        self._items: dict[str, Callable[..., T]] = {}

    def register(self, name: str, factory: Callable[..., T]) -> None:
        self._items[name] = factory

    def get(self, name: str) -> Callable[..., T]:
        if name not in self._items:
            available = ", ".join(sorted(self._items))
            raise KeyError(f"Unknown registry key '{name}'. Available: {available}")
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)

