import numpy as np

from ..modules.resource_allocation import ResourceAllocationBase
from ..modules.params import GeometryParams
from .parameters import SharedParams as AnomalySharedParams
from .parameters import WirelessParams as AnomalyWirelessParams


class TDDResourceAllocation(ResourceAllocationBase):
    """Class implementing dummy resource allocation for anomaly tracking, assuming a single transmitter per slot.

    It just takes the users scheduled for the pull and push sub-frame and generates the OFDM-like allocation.
    """
    def __init__(self, c: AnomalySharedParams, wireless: AnomalyWirelessParams, geo: GeometryParams):
        super().__init__(c, wireless, geo)

        # Impose single UE per resource
        self.rg.num_tx = 1


        # Allocation now
        self.pull_push_outcome = None

    def _rx_tx_association(self) -> np.ndarray:
        """Return the association between the receiver and transmitter nodes, assuming a single transmitter per slot.
        """
        return np.ones((1, 1))

    def slot_ue_association_mask(self) -> np.ndarray:
        """Transform allocation to the resource grid for channel masking and transmission module"""
        # General mask
        mask = np.zeros((self.c.num_slots, self.c.num_ue), dtype=bool)

        # Remove the -1 from the outcomes
        singleton_mask = self.pull_push_outcome != -1
        singleton_slots = np.where(singleton_mask)
        singleton_nodes = self.pull_push_outcome[singleton_mask]

        # Make true the singleton slots
        mask[singleton_slots, singleton_nodes] = True

        # TODO: the system should just generate the channel as the allocation commands, so explicit
        #  ra is not needed.

        return mask


    def __call__(self, frame: int, **kwargs):
        push = kwargs["push_outcome"]
        pull = kwargs["pull_outcome"]

        # Outcome is -1 if failed due to collision, index of the node otherwise
        self.pull_push_outcome = np.hstack((pull, push))
        return self.broadcast_slot_ue_mask()

