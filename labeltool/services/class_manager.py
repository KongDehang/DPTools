from __future__ import annotations

from pathlib import Path
import ast
import re
from dataclasses import dataclass
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


@dataclass(slots=True)
class ClassManagerState:
    yaml_path: Path | None
    id_to_name: dict[int, str]
    name_to_id: dict[str, int]
    dirty: bool


class ClassManager:
    def __init__(self) -> None:
        self.root_dir: Path | None = None
        self.yaml_path: Path | None = None
        self.id_to_name: dict[int, str] = {}
        self.name_to_id: dict[str, int] = {}
        self._dirty = False

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def clear(self) -> None:
        self.yaml_path = None
        self.id_to_name.clear()
        self.name_to_id.clear()
        self._dirty = False

    def snapshot(self) -> ClassManagerState:
        return ClassManagerState(
            yaml_path=self.yaml_path,
            id_to_name=dict(self.id_to_name),
            name_to_id=dict(self.name_to_id),
            dirty=self._dirty,
        )

    def restore(self, state: ClassManagerState) -> None:
        self.yaml_path = state.yaml_path
        self.id_to_name = dict(state.id_to_name)
        self.name_to_id = dict(state.name_to_id)
        self._dirty = state.dirty

    def load_from_root(self, root_dir: str | Path) -> None:
        self.clear()
        self.root_dir = Path(root_dir)
        self.yaml_path = self._find_yaml_path(self.root_dir)
        if not self.yaml_path or not self.yaml_path.exists():
            return

        try:
            text = self.yaml_path.read_text(encoding="utf-8")
        except Exception:
            return

        mapping = self._parse_names_from_text(text)
        self._replace_mapping(mapping)
        self._dirty = False

    def _replace_mapping(self, mapping: dict[int, str]) -> None:
        self.id_to_name = {int(key): str(value) for key, value in sorted(mapping.items())}
        self.name_to_id = {value: key for key, value in self.id_to_name.items()}

    def _find_yaml_path(self, root_dir: Path) -> Path | None:
        keywords = ("data", "dataset", "label", "class", "classes")
        search_roots: list[tuple[Path, bool]] = [(root_dir, True)]
        search_roots.extend((parent, False) for parent in list(root_dir.parents)[:3])

        yaml_candidates: list[Path] = []
        for base, recursive in search_roots:
            if not base.exists():
                continue
            if recursive:
                yaml_candidates.extend(base.rglob("*.yaml"))
                yaml_candidates.extend(base.rglob("*.yml"))
            else:
                yaml_candidates.extend(base.glob("*.yaml"))
                yaml_candidates.extend(base.glob("*.yml"))

        if not yaml_candidates:
            return None

        unique_candidates: list[Path] = []
        seen: set[Path] = set()
        for candidate in yaml_candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            unique_candidates.append(candidate)

        def score(path: Path) -> tuple[int, int, int]:
            lower_name = path.name.lower()
            keyword_score = sum(1 for keyword in keywords if keyword in lower_name)
            try:
                depth = len(path.relative_to(root_dir).parts)
            except Exception:
                depth = len(path.parts)
            return keyword_score, -depth, -len(path.parts), -len(path.name)

        return max(unique_candidates, key=score)

    def _parse_names_from_text(self, text: str) -> dict[int, str]:
        if yaml is not None:
            try:
                data = yaml.safe_load(text)
                if isinstance(data, dict):
                    names = data.get("names")
                    mapping = self._coerce_names(names)
                    if mapping:
                        return mapping
            except Exception:
                pass

        return self._parse_names_without_yaml(text)

    def _coerce_names(self, names: Any) -> dict[int, str]:
        if isinstance(names, list):
            return {index: str(value) for index, value in enumerate(names)}
        if isinstance(names, dict):
            result: dict[int, str] = {}
            for key, value in names.items():
                try:
                    class_id = int(key)
                except Exception:
                    continue
                result[class_id] = str(value)
            return result
        return {}

    def _parse_names_without_yaml(self, text: str) -> dict[int, str]:
        lines = text.splitlines()
        for index, line in enumerate(lines):
            match_inline = re.match(r"^\s*names\s*:\s*(\[.*\]|\{.*\})\s*$", line)
            if match_inline:
                raw = match_inline.group(1)
                mapping = self._parse_inline_names(raw)
                if mapping:
                    return mapping

            match_block = re.match(r"^(?P<indent>\s*)names\s*:\s*$", line)
            if match_block:
                indent = len(match_block.group("indent"))
                block_lines: list[str] = []
                for tail_line in lines[index + 1 :]:
                    if not tail_line.strip():
                        block_lines.append(tail_line)
                        continue
                    tail_indent = len(tail_line) - len(tail_line.lstrip())
                    if tail_indent <= indent:
                        break
                    block_lines.append(tail_line)
                mapping = self._parse_block_names(block_lines)
                if mapping:
                    return mapping
        return {}

    def _parse_inline_names(self, raw: str) -> dict[int, str]:
        try:
            value = ast.literal_eval(raw)
        except Exception:
            value = None

        if isinstance(value, list):
            return {index: str(item) for index, item in enumerate(value)}
        if isinstance(value, dict):
            result: dict[int, str] = {}
            for key, item in value.items():
                try:
                    class_id = int(key)
                except Exception:
                    continue
                result[class_id] = str(item)
            return result

        stripped = raw.strip()[1:-1].strip()
        if not stripped:
            return {}
        if raw.strip().startswith("["):
            items = [item.strip().strip("'\"") for item in stripped.split(",") if item.strip()]
            return {index: item for index, item in enumerate(items)}
        return {}

    def _parse_block_names(self, block_lines: list[str]) -> dict[int, str]:
        items: dict[int, str] = {}
        current_list: list[str] = []
        dict_mode = False

        for line in block_lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("-"):
                current_list.append(stripped[1:].strip().strip("'\""))
            else:
                match = re.match(r"^(\d+)\s*:\s*(.+?)\s*$", stripped)
                if match:
                    dict_mode = True
                    items[int(match.group(1))] = match.group(2).strip().strip("'\"")

        if dict_mode and items:
            return items
        if current_list:
            return {index: value for index, value in enumerate(current_list)}
        return {}

    def next_available_id(self) -> int:
        return (max(self.id_to_name.keys()) + 1) if self.id_to_name else 0

    def get_name(self, class_id: int) -> str | None:
        return self.id_to_name.get(int(class_id))

    def get_id(self, class_name: str) -> int | None:
        return self.name_to_id.get(str(class_name))

    def ensure_name(self, class_name: str) -> int:
        name = str(class_name).strip()
        if not name:
            raise ValueError("Class name cannot be empty")
        existing_id = self.name_to_id.get(name)
        if existing_id is not None:
            return existing_id

        class_id = self.next_available_id()
        self.id_to_name[class_id] = name
        self.name_to_id[name] = class_id
        self._dirty = True
        return class_id

    def ensure_id(self, class_id: int, class_name: str | None = None) -> str:
        class_id = int(class_id)
        existing_name = self.id_to_name.get(class_id)
        provided_name = str(class_name).strip() if class_name is not None else ""

        if existing_name:
            if (
                provided_name
                and existing_name.startswith(("unknown_id_", "ID "))
                and provided_name not in self.name_to_id
            ):
                self.name_to_id.pop(existing_name, None)
                self.id_to_name[class_id] = provided_name
                self.name_to_id[provided_name] = class_id
                self._dirty = True
                return provided_name
            return existing_name

        assigned_name = provided_name or f"ID {class_id}"
        if assigned_name in self.name_to_id and self.name_to_id[assigned_name] != class_id:
            suffix = 1
            base_name = assigned_name
            while f"{base_name}_{suffix}" in self.name_to_id:
                suffix += 1
            assigned_name = f"{base_name}_{suffix}"

        self.id_to_name[class_id] = assigned_name
        self.name_to_id[assigned_name] = class_id
        self._dirty = True
        return assigned_name

    def resolve_label_token(self, token: str) -> str:
        text = str(token).strip()
        if not text:
            return self.ensure_id(self.next_available_id(), "object")
        if text.isdigit():
            return self.ensure_id(int(text))
        if text not in self.name_to_id:
            self.ensure_name(text)
        return text

    def rename_class(self, class_id: int, new_name: str) -> str:
        class_id = int(class_id)
        if class_id not in self.id_to_name:
            raise KeyError(f"Class id {class_id} does not exist")

        cleaned = str(new_name).strip()
        if not cleaned:
            raise ValueError("Class name cannot be empty")
        existing = self.name_to_id.get(cleaned)
        if existing is not None and existing != class_id:
            raise ValueError(f"Class name '{cleaned}' is already used by id {existing}")

        old_name = self.id_to_name[class_id]
        if old_name == cleaned:
            return old_name

        self.id_to_name[class_id] = cleaned
        self.name_to_id.pop(old_name, None)
        self.name_to_id[cleaned] = class_id
        self._dirty = True
        return old_name

    def delete_class(self, class_id: int) -> str:
        class_id = int(class_id)
        if class_id not in self.id_to_name:
            raise KeyError(f"Class id {class_id} does not exist")

        class_name = self.id_to_name.pop(class_id)
        self.name_to_id.pop(class_name, None)
        self._dirty = True
        return class_name

    def remap_class_id(
        self,
        old_id: int,
        new_id: int,
        *,
        swap_if_conflict: bool = False,
    ) -> tuple[str, str | None]:
        old_id = int(old_id)
        new_id = int(new_id)

        if old_id not in self.id_to_name:
            raise KeyError(f"Class id {old_id} does not exist")
        if old_id == new_id:
            return self.id_to_name[old_id], None

        moving_name = self.id_to_name[old_id]
        target_name = self.id_to_name.get(new_id)
        if target_name is not None and not swap_if_conflict:
            raise ValueError(f"Class id {new_id} is already used by '{target_name}'")

        if target_name is None:
            self.id_to_name.pop(old_id, None)
            self.id_to_name[new_id] = moving_name
        else:
            self.id_to_name[new_id] = moving_name
            self.id_to_name[old_id] = target_name

        self.name_to_id = {value: key for key, value in self.id_to_name.items()}
        self._dirty = True
        return moving_name, target_name

    def sorted_items(self) -> list[tuple[int, str]]:
        return sorted(self.id_to_name.items(), key=lambda item: item[0])

    def sync_to_yaml(self, force: bool = False) -> Path | None:
        if not self.root_dir and not self.yaml_path:
            return None
        if not force and not self._dirty and self.yaml_path and self.yaml_path.exists():
            return self.yaml_path

        target = self.yaml_path or (self.root_dir / "data.yaml")
        if target is None:
            return None

        names_mapping = {class_id: name for class_id, name in self.sorted_items()}
        payload: dict[str, Any]
        if target.exists() and yaml is not None:
            try:
                payload = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
                if not isinstance(payload, dict):
                    payload = {}
            except Exception:
                payload = {}
        else:
            payload = {}

        payload.setdefault("train", "images/train")
        payload.setdefault("val", "images/val")
        payload["nc"] = len(names_mapping)
        payload["names"] = names_mapping

        try:
            if yaml is not None:
                text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
            else:
                if names_mapping:
                    names_lines = "\n".join(
                        f"  {class_id}: {name!r}"
                        for class_id, name in sorted(names_mapping.items())
                    )
                else:
                    names_lines = "  {}"
                text = (
                    f"train: images/train\n"
                    f"val: images/val\n\n"
                    f"nc: {len(names_mapping)}\n"
                    f"names:\n{names_lines}\n"
                )
            target.write_text(text, encoding="utf-8")
            self.yaml_path = target
            self._dirty = False
            return target
        except Exception:
            return None
