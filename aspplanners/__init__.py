# Register both backends with the UP framework on import, so it's transparent
# to the user. Both adapters live in aspplanners.up_engines; add_engine takes the
# module/class as strings and imports them lazily on first use. Importing
# up_engines here is safe without `aspforaba` -- the ABA planner imports it
# lazily, only when the ABAPlanner engine actually solves.
import unified_planning as up

from aspplanners.up_engines import UPPLASPPlanner, UPABAPlanner  # noqa: F401

env = up.environment.get_environment()
env.factory.add_engine('ASPPlanner', 'aspplanners.up_engines', 'UPPLASPPlanner')
env.factory.add_engine('ABAPlanner', 'aspplanners.up_engines', 'UPABAPlanner')
