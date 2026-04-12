# Utilities to save/load Pydantic models to/from JSONL files.
from pathlib import Path
from collections.abc import Iterable
from typing import TypeVar

from pydantic import BaseModel

def _ensure_parent(path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: str | Path, item: BaseModel) -> None:
	"""Save a Pydantic `BaseModel` instance to a JSON file.

	- Creates parent directories if needed.
	- Overwrites the file if it already exists.
	"""
	p = Path(path)
	_ensure_parent(p)
	with p.open("w", encoding="utf-8") as fh:
		fh.write(item.model_dump_json(indent=4))

def save_jsonl(path: str | Path, items: Iterable[BaseModel], append: bool = False) -> None:
	"""Save an iterable of Pydantic `BaseModel` instances to a JSONL file.

	- Each model is written as one JSON object per line.
	- Creates parent directories if needed.
	- If `append` is True, appends to the file, otherwise overwrites it.
	"""
	p = Path(path)
	_ensure_parent(p)
	mode = "a" if append else "w"
	with p.open(mode, encoding="utf-8") as fh:
		for item in items:
			if not isinstance(item, BaseModel):
				raise TypeError("save_jsonl expects Pydantic BaseModel instances")
			# compact separators to keep jsonl files small
			fh.write(item.model_dump_json() + "\n")

T = TypeVar("T", bound=BaseModel)

def load_jsonl(path: str | Path, model: type[T]) -> list[T]:
	"""Load a JSONL file and parse each line into `model` instances.

	Returns an empty list if the file does not exist.
	"""
	p = Path(path)
	if not p.exists():
		return []
	out: list[T] = []
	with p.open("r", encoding="utf-8") as fh:
		for raw in fh:
			raw = raw.strip()
			if not raw:
				continue
			out.append(model.model_validate_json(raw))
	return out


__all__ = [
	"save_json",
	"save_jsonl",
	"load_jsonl",
]

