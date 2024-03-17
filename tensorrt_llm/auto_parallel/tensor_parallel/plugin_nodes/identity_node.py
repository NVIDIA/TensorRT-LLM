from ..identity_node import Identity
from ..plugin_node import PluginNode


class IdentityPlugin(Identity, PluginNode):

    def __init__(self, layer):
        PluginNode.__init__(self, layer)

    def _collect_strategies(self, device_mesh):
        return Identity._collect_strategies(self, device_mesh)
