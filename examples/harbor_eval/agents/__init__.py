from . import codex, mini_swe_agent, opencode, pi
from .codex import PrebuiltCodex
from .mini_swe_agent import PrebuiltMiniSweAgent
from .opencode import PrebuiltOpenCode
from .pi import PrebuiltPi
from .prebuilt import Prebuilt

PREBUILT: dict[str, Prebuilt] = {m.PREBUILT.name: m.PREBUILT for m in (mini_swe_agent, pi, opencode, codex)}

__all__ = ["PREBUILT", "Prebuilt", "PrebuiltCodex", "PrebuiltMiniSweAgent", "PrebuiltOpenCode", "PrebuiltPi"]
