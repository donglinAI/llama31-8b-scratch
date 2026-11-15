"""YAML 配置加载器。

设计目标：
1. 仅依赖 PyYAML，不引入 omegaconf/hydra 等重型配置框架；
2. 支持点号属性访问（``cfg.model.hidden_size``）与字典访问两种方式；
3. 支持 ``_base_`` 关键字实现配置继承（相对路径或路径列表，可多层嵌套），
   子配置覆盖父配置，字典做深度合并；
4. 支持命令行 ``key.subkey=value`` 形式的覆盖项。

该模块是纯标准库 + PyYAML 实现，可在无 GPU、无 torch 环境直接单测。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml


class ConfigError(ValueError):
    """配置文件解析或合并阶段的异常。"""


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """把 ``override`` 深度合并进 ``base``，返回新的字典。

    规则：
    - 两边同名字段都是 dict 时递归合并；
    - 其余情况以 override 的值为准（列表整体替换，不做元素级拼接）。
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _parse_scalar(text: str) -> Any:
    """把命令行传入的字符串标量解析成 Python 类型。"""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


class Config:
    """支持属性访问的配置对象。"""

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        if data is not None and not isinstance(data, dict):
            raise ConfigError(f"配置根节点必须是字典，实际为 {type(data)!r}")
        self._data: Dict[str, Any] = data or {}

    # ---------- 构造 ----------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        return cls(copy.deepcopy(data))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path).resolve()
        if not path.is_file():
            raise ConfigError(f"配置文件不存在: {path}")
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ConfigError(f"配置文件根节点必须是字典: {path}")

        merged = cls._resolve_bases(raw, path)
        return cls(merged)

    @classmethod
    def _resolve_bases(cls, raw: Dict[str, Any], current_path: Path) -> Dict[str, Any]:
        """递归处理 ``_base_`` 继承。"""
        bases = raw.pop("_base_", None)
        if bases is None:
            return raw

        if isinstance(bases, str):
            base_paths: Sequence[str] = [bases]
        elif isinstance(bases, list):
            base_paths = bases
        else:
            raise ConfigError(
                f"_base_ 必须是字符串或列表，实际为 {type(bases)!r}"
            )

        merged: Dict[str, Any] = {}
        for base_ref in base_paths:
            base_file = (current_path.parent / base_ref).resolve()
            base_config = cls.from_yaml(base_file)
            merged = deep_merge(merged, base_config.to_dict())
        return deep_merge(merged, raw)

    # ---------- 访问 ----------
    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted_key: str, value: Any) -> None:
        parts = dotted_key.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"路径 {dotted_key} 中间节点不是字典")
        node[parts[-1]] = value

    def __getattr__(self, name: str) -> Any:
        # 仅在常规属性查找失败时进入；避免与内部 _data 递归冲突
        try:
            value = self._data[name]
        except KeyError as exc:
            raise AttributeError(f"配置中不存在字段: {name}") from exc
        return _wrap(value)

    def __getitem__(self, key: str) -> Any:
        return _wrap(self._data[key])

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"Config({self._data!r})"

    def apply_overrides(self, overrides: Optional[List[str]] = None) -> "Config":
        """应用 ``a.b=value`` 形式的命令行覆盖，返回自身。"""
        for item in overrides or []:
            if "=" not in item:
                raise ConfigError(f"覆盖项必须是 key=value 形式: {item}")
            dotted_key, raw_value = item.split("=", 1)
            self.set(dotted_key.strip(), _parse_scalar(raw_value.strip()))
        return self


def _wrap(value: Any) -> Any:
    """字典递归包成 Config，列表/标量原样返回（列表内字典也包装）。"""
    if isinstance(value, dict):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


def load_config(
    path: str | Path, overrides: Optional[List[str]] = None
) -> Config:
    """加载 YAML 配置并应用命令行覆盖。"""
    config = Config.from_yaml(path)
    config.apply_overrides(overrides)
    return config
