"""Algorithm registry."""
from harl.algorithms.actors.happo import HAPPO
from harl.algorithms.actors.hatrpo import HATRPO
from harl.algorithms.actors.haa2c import HAA2C
from harl.algorithms.actors.haddpg import HADDPG
from harl.algorithms.actors.hatd3 import HATD3
from harl.algorithms.actors.hasac import HASAC
from harl.algorithms.actors.had3qn import HAD3QN
from harl.algorithms.actors.maddpg import MADDPG
from harl.algorithms.actors.mappo import MAPPO
from harl.algorithms.actors.hatrpo_CS import HATRPO_CS
from harl.algorithms.actors.matrpo import MATRPO
from harl.algorithms.actors.madpo import MADPO

# from harl.algorithms.actors.madac import MADAC

ALGO_REGISTRY = {
    "hatrpo_CS": HATRPO_CS,
    "happo": HAPPO,
    "hatrpo": HATRPO,
    "haa2c": HAA2C,
    "haddpg": HADDPG,
    "hatd3": HATD3,
    "hasac": HASAC,
    "had3qn": HAD3QN,
    "maddpg": MADDPG,
    "mappo": MAPPO,
    "matrpo": MATRPO,
    "madpo": MADPO,
    # "madac": MADAC
}
