from abc import ABC, abstractmethod
import numpy as np
import sionna.phy as phy

from .. import utils as ut
from .params import SharedParams, WirelessParams, GeometryParams, EstimationParams
from .theory import eval_speb


class ResourceAllocationBase(ABC):
    """Abstract base class for OFDM resource allocation.

    Builds the sionna.phy `ResourceGrid` (time/frequency resource description), and a
    `StreamManagement` (rx/tx stream association). When called, it returns slot-ue and ap-ue association maps
    for current frame allocation.

    Subclasses implement the model-specific `slot_ue_association_mask`, and `__call__` methods. Optionally, custom
    `ap_ue_association_mask`, `_pilot_ofdm_symbol_indices`, and `rx_tx_association` methods can be implemented.
    """
    def __init__(self, c: SharedParams, wireless: WirelessParams,
                 geo: GeometryParams, estimate: EstimationParams):
        self.c = c
        self.wireless = wireless
        self.geo = geo
        self.estimate = estimate

        self.rg = phy.ofdm.ResourceGrid(
            num_ofdm_symbols=self.c.num_sym,
            fft_size=self.c.num_scs,
            subcarrier_spacing=self.wireless.sc_spacing,
            num_tx=self.c.num_ue, # TODO: check if it is possible to increase the number of transmitter w/o issues
            num_streams_per_tx=1,
            cyclic_prefix_length=self._cp_length_samples(),
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=self.pilot_ofdm_symbol_indices(),
        )

        self.sm = phy.mimo.StreamManagement(
            rx_tx_association=self._rx_tx_association(),
            num_streams_per_tx=1,
        )

        # Internal attributes
        self._last_allocated_ues = None  # np.zeros(self.c.num_slots, dtype=np.int32)
        self.risk0 = None
        self.risk1 = None
        self.delta_risk = None
        self.delta_queue = None
        self.forced = None

    def output_risks(self):
        if self.risk0 is None:
            risk0 = np.nan * np.zeros(self.c.num_ue)
        else:
            risk0 = self.risk0
        if self.risk1 is None:
            risk1 = np.nan * np.zeros(self.c.num_ue)
        else:
            risk1 = self.risk1
        return np.stack((risk0, risk1), axis=-1)


    def _cp_length_samples(self) -> int:
        """Cyclic prefix length in samples, consistent with `self.wireless.cp_time`."""
        return int(round(self.wireless.cp_time * self.wireless.sc_spacing * self.c.num_scs))

    def pilot_ofdm_symbol_indices(self) -> list:
        """Default two-pilot Kronecker pattern: one early, one late symbol in the slot
        (mirrors Sionna's own tutorial convention of [2, T-3])."""
        if self.c.num_sym < 6:
            return [0]
        return [2, self.c.num_sym - 3]

    def _rx_tx_association(self) -> np.ndarray:
        """As a default, every AP jointly receives from every UE (cell-free/distributed uplink combining).
        Due to centralized operations, it is like a single BS with L*M multiple antennas.
        """
        return np.ones((1, self.c.num_ue))

    @abstractmethod
    def slot_ue_association_mask(self) -> np.ndarray:
        """Build slot-UE allocation map.

        :returns: np.ndarray[bool], shape (S, U), True where UE u is allocated to slot s, else False.
        """
        raise NotImplementedError

    def broadcast_slot_ue_mask(self, mask: np.ndarray | None = None, materialize: bool = False) -> np.ndarray:
        """Reshape a (S, U) allocation mask so it broadcasts directly against a per-slot-batched
        channel tensor of shape (S, 1, L*M, U, 1, T, F).

        :param mask: np.ndarray | None, shape (S, U). If None, uses `self.slot_user_mask()`.
        :param materialize: bool, if True returns the fully expanded (S, 1, L*M, U, 1, T, F) array
            instead of the broadcastable (S, 1, 1, U, 1, 1, 1) one. Only useful if you need the
            dense mask itself (e.g. to inspect/save it).

        :returns: np.ndarray, shape (S, 1, 1, U, 1, 1, 1) by default, or (S, 1, L*M, U, 1, T, F) if
            `materialize=True`.
        """
        if mask is None:
            mask = self.slot_ue_association_mask()
        reshaped = mask[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]  # (S,1,1,U,1,1,1)
        if not materialize:
            return reshaped
        return np.broadcast_to(reshaped, (self.c.num_slots, 1, self.c.num_ap * self.c.num_ant,
                                          self.c.num_ue, 1, self.c.num_sym, self.c.num_scs)).copy()

    def scheduled_ues(self, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For each slot, how many UEs are actually scheduled and which ones.

        :param mask: np.ndarray[bool] | None, shape (S, U). If None, uses
            `self.slot_ue_association_mask()`.

        :returns: tuple (num_active, ue_indices) --
            num_active: np.ndarray, shape (S,), int, count of scheduled UEs per slot.
            ue_indices: list of length S, each entry an np.ndarray of the UE indices
                scheduled in that slot. Use these to index into TransmissionCFR's
                (S, U, ...) outputs (b, b_hat, x, x_hat) and discard everything else --
                unscheduled UEs' entries are noise-only decodes, not real transmissions.
        """
        if mask is None:
            mask = self.slot_ue_association_mask()
        num_scheduled = mask.sum(axis=1).astype(np.int32)
        # ue_indices = [np.flatnonzero(mask[s]) for s in range(mask.shape[0])]
        s_idx, u_idx = np.nonzero(mask)
        return num_scheduled, s_idx, u_idx

    def ap_ue_association_mask(self, association: np.ndarray | None = None) -> np.ndarray:
        """Which APs actually serve which UEs, for the merged-antenna virtual receiver. As a default, each UE is served
        by all APs.

        :param association: np.ndarray | None, shape (L, U), True where AP l serves UE u, else False.
            If None, defaults to fully joint (every AP serves every UE).

        :returns: np.ndarray, shape (L, U).
        """
        if association is None:
            return np.ones((self.c.num_ap, self.c.num_ue), dtype=bool)
        assert association.shape == (self.c.num_ap, self.c.num_ue)
        return association.astype(bool)

    def broadcast_ap_ue_mask(self, association: np.ndarray | None = None) -> np.ndarray:
        """Broadcast an (L, U) AP-UE association mask onto the merged L*M antenna axis, for a
        channel already reshaped/batched to (S, 1, L*M, U, 1, T, F).

        :returns: np.ndarray, shape (1, 1, L*M, U, 1, 1, 1) -- broadcasts against (S, 1, L*M, U, 1, T, F).
        """
        mask = self.ap_ue_association_mask(association)  # (L, U)
        # Repeat each AP's row M times: antenna index l*M+m belongs to AP l, matching the
        # ordering produced by channel.reshape(L*M, U, ...) (AP-major, antenna-minor).
        mask_expanded = np.repeat(mask, self.c.num_ant, axis=0)  # (L*M, U)
        return mask_expanded[np.newaxis, np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis]

    def eval_semantic_risk(self, frame, efim: np.ndarray, covariance_pri: np.ndarray) -> tuple:
        """Compute semantic risk based on UE position/region estimates.

        :param frame: int, index of the current frame.
        :param efim: np.ndarray, shape (..., U, d, d) representing the effective FIM of the position observations.
        :param covariance_pri: np.ndarray, shape (..., U, d, d) representing the covariance matrix of the prior model.
        :returns: tuple of (risk0, risk1, delta) with risk0 being a np.ndarray (U,) representing the semantic risk
            with no allocation (only prior is used); risk1 is a np.ndarray (U) representing the semantic risk of
            allocating users; delta is a np.ndarray (U) representing their difference.
        """
        covariance_post = np.linalg.inv(efim + np.linalg.inv(covariance_pri))
        bound_dist = np.zeros(self.c.num_ue)
        bound_normal = np.zeros((self.c.num_ue, 3))
        for r in range(self.c.num_regions):
            ue_mask = self.estimate.ue_region_hat[frame] == r
            if np.any(ue_mask):
                bound_dist[ue_mask], bound_normal[ue_mask] = ut.boundary_dist_normal(self.estimate.ue_pos_hat_prio[frame, ue_mask],
                                                                                     self.geo.semantic_regions.geoms[r],
                                                                                     exclude_polygon=self.geo.environment_shape)

        risk0 = ut.qfun(bound_dist / np.squeeze(np.sqrt(np.matrix_transpose(bound_normal[..., np.newaxis])
                                                        @ covariance_pri @ bound_normal[..., np.newaxis])))
        risk1 = ut.qfun(bound_dist / np.squeeze(np.sqrt(np.matrix_transpose(bound_normal[..., np.newaxis])
                                                        @ covariance_post @ bound_normal[..., np.newaxis]), axis=(-2,-1)))
        delta_risk = risk0 - risk1

        return risk0, risk1, delta_risk

    def eval_comm_risk(self, delta_queue: np.ndarray):
        forced = delta_queue < self.wireless.traffic_pdf.ppf(self.wireless.queue_reliability, scale=self.wireless.traffic_avg)
        return delta_queue, forced



    @abstractmethod
    def __call__(self, frame: int, **kwargs) -> np.ndarray:
        """Returns the ResourceGrid and StreamManagement for the current scenario and build the slot_ue_association_mask
        based on some prior information.

        :param frame: int, current scenario frame for retrieving current priors from estimate
        :returns: tuple containing, e.g., (slot_ue_association_mask: np.ndarray, ...)
        """
        raise NotImplementedError


class RoundRobinAllocation(ResourceAllocationBase):
    """RoundRobin OFDM resource allocation.

    Builds the sionna.phy `ResourceGrid` (time/frequency resource description),  a
    `StreamManagement` (rx/tx stream association).
    Moreover, it creates boolean np.ndarrays for `slot-ue allocation mask` and `ue-ap allocation masks`.

    :param c: SharedParams, shared scenario parameters (num_ue, num_ap, num_sym, num_scs, ...).
    :param wireless: WirelessParams, OFDM numerology parameters (sc_spacing, cp_time, ...).
    :param geo: GeometryParams, geometric configuration (not used by the base allocation, but
        available for subclasses that allocate resources based on UE position/region).
    :param estimate: EstimationParams, current position/region estimates (same rationale as `geo`).
    """

    def __init__(self, c: SharedParams, wireless: WirelessParams,
                 geo: GeometryParams, estimate: EstimationParams):
        super().__init__(c, wireless, geo, estimate)


    def slot_ue_association_mask(self) -> np.ndarray:
        """Orthogonal round-robin slot allocation: UE `u` is scheduled in slot `u % S` and accounting for the last
        allocation made.

        :returns: np.ndarray[bool], shape (S, U), True where UE u is allocated to slot s, else False.
        """
        try:
            first_ue = (self._last_allocated_ues[-1] + 1) % self.c.num_ue
        except TypeError:
            first_ue = 0
        mask = np.roll(np.eye(self.c.num_slots, self.c.num_ue, dtype=bool), first_ue, axis=1)
        # last_ue = first_ue + self.c.num_slots
        # if last_ue >= self.c.num_ue:
        #    ue_idx = np.arange(first_ue, self.c.num_ue)
        #    empty_slots = self.c.num_slots - len(ue_idx)
        #    ue_idx = np.hstack((ue_idx, np.arange(0, empty_slots)))
        # else:
        #    ue_idx = np.arange(first_ue, last_ue)
        # mask[np.arange(self.c.num_slots), ue_idx] = True
        # self._last_allocated_ues = ue_idx.astype(np.int32)
        self._last_allocated_ues = np.where(mask)[1]
        return mask.astype(bool)

    def __call__(self, frame: int, **kwargs) -> np.ndarray:
        """Call the association for the current frame.

        :returns: np.ndarray[bool], ue-slot-association mask.
        """
        if frame < 0:
            raise ValueError("Frame index cannot be negative.")
        # prior = self.estimate.ue_pos_hat[frame]
        return self.slot_ue_association_mask()


class SemanticRiskAllocation(ResourceAllocationBase):
    """Semantic risk-based OFDM allocation.

    Builds the sionna.phy `ResourceGrid` (time/frequency resource description),  a
    `StreamManagement` (rx/tx stream association).
    Moreover, it creates boolean np.ndarrays for `slot-ue allocation mask` and `ue-ap allocation masks`.
    This base class gives the simplest possible grid. Subclass this for other allocation strategies.

    :param c: SharedParams, shared scenario parameters (num_ue, num_ap, num_sym, num_scs, ...).
    :param wireless: WirelessParams, OFDM numerology parameters (sc_spacing, cp_time, ...).
    :param geo: GeometryParams, geometric configuration (not used by the base allocation, but
        available for subclasses that allocate resources based on UE position/region).
    :param estimate: EstimationParams, current position/region estimates (same rationale as `geo`).
    """
    def __init__(self, c: SharedParams, wireless: WirelessParams,
                 geo: GeometryParams, estimate: EstimationParams):
        super().__init__(c, wireless, geo, estimate)

    def slot_ue_association_mask(self) -> np.ndarray:
        """Orthogonal slot allocation based on predicted semantic risk and predicted SE.

        :returns: np.ndarray[bool], shape (S, U), True where UE u is allocated to slot s, else False.
        """
        # General mask
        mask = np.zeros((self.c.num_slots, self.c.num_ue), dtype=bool)


        # Allocate the forced set
        num_forced = np.sum(self.forced)
        idx_forced = np.nonzero(self.forced)[0]
        if  num_forced >= self.c.num_slots:
            mask[np.arange(self.c.num_slots), idx_forced[:self.c.num_slots]] = True
            return mask
        else:
            mask[np.arange(num_forced), idx_forced] = True
            # Order UEs by delta_risk (descending) but prioritizing lower delta_queue if there is a rare tie
            order = np.lexsort((self.delta_queue, -self.delta_risk))
            # Remove eventually already allocated UEs
            order = order[~np.isin(order, idx_forced)]
            # Remaining slots (cap to available slots)
            remaining_slots = min(self.c.num_slots - num_forced, order.size)
            # Take slot indexing of available slots
            slot_idx = np.arange(num_forced, num_forced + remaining_slots)
            # Schedule the remaining nodes (up to the remaining slots)
            mask[slot_idx[:remaining_slots], order[:remaining_slots]] = True
            return mask

    def __call__(self, frame: int, **kwargs) -> np.ndarray:
        """Call the association for the current frame.

        :returns: np.ndarray[bool], ue-slot-association mask.
        """
        if frame < 0:
            raise ValueError("Frame index cannot be negative.")

        self.risk0, self.risk1, self.delta_risk = self.eval_semantic_risk(frame, **ut.filter_kwargs(self.eval_semantic_risk, **kwargs))
        self.delta_queue, self.forced = self.eval_comm_risk(**ut.filter_kwargs(self.eval_comm_risk, **kwargs))
        return self.slot_ue_association_mask()


class PebBasedAllocation(ResourceAllocationBase):
    """PEB-based OFDM allocation.

    Builds the sionna.phy `ResourceGrid` (time/frequency resource description),  a
    `StreamManagement` (rx/tx stream association).
    Moreover, it creates boolean np.ndarrays for `slot-ue allocation mask` and `ue-ap allocation masks`.
    This base class gives the simplest possible grid. Subclass this for other allocation strategies.

    :param c: SharedParams, shared scenario parameters (num_ue, num_ap, num_sym, num_scs, ...).
    :param wireless: WirelessParams, OFDM numerology parameters (sc_spacing, cp_time, ...).
    :param geo: GeometryParams, geometric configuration (not used by the base allocation, but
        available for subclasses that allocate resources based on UE position/region).
    :param estimate: EstimationParams, current position/region estimates (same rationale as `geo`).
    """
    def __init__(self, c: SharedParams, wireless: WirelessParams,
                 geo: GeometryParams, estimate: EstimationParams):
        super().__init__(c, wireless, geo, estimate)

    def slot_ue_association_mask(self) -> np.ndarray:
        """Orthogonal slot allocation based on predicted semantic risk and predicted SE.

        :returns: np.ndarray[bool], shape (S, U), True where UE u is allocated to slot s, else False.
        """
        # General mask
        mask = np.zeros((self.c.num_slots, self.c.num_ue), dtype=bool)

        # Allocate the forced set
        num_forced = np.sum(self.forced)
        idx_forced = np.nonzero(self.forced)[0]
        if  num_forced >= self.c.num_slots:
            mask[np.arange(self.c.num_slots), idx_forced[:self.c.num_slots]] = True
            return mask
        else:
            mask[np.arange(num_forced), idx_forced] = True
            # Order UEs by delta_risk (descending) but prioritizing lower delta_queue if there is a rare tie
            order = np.lexsort((self.delta_queue, -self.delta_risk))
            # Remove eventually already allocated UEs
            order = order[~np.isin(order, idx_forced)]
            # Remaining slots (cap to available slots)
            remaining_slots = min(self.c.num_slots - num_forced, order.size)
            # Take slot indexing of available slots
            slot_idx = np.arange(num_forced, num_forced + remaining_slots)
            # Schedule the remaining nodes (up to the remaining slots)
            mask[slot_idx[:remaining_slots], order[:remaining_slots]] = True
            return mask

    def __call__(self, frame: int, **kwargs) -> np.ndarray:
        """Call the association for the current frame.

        :returns: np.ndarray[bool], ue-slot-association mask.
        """
        if frame < 0:
            raise ValueError("Frame index cannot be negative.")

        self.risk0, self.risk1, self.delta_risk = eval_speb(**ut.filter_kwargs(eval_speb, **kwargs))
        self.delta_queue, self.forced = self.eval_comm_risk(**ut.filter_kwargs(self.eval_comm_risk, **kwargs))
        return self.slot_ue_association_mask()


class SemanticRiskAllocationNoComm(ResourceAllocationBase):
    """Semantic risk-based OFDM allocation.

    Builds the sionna.phy `ResourceGrid` (time/frequency resource description),  a
    `StreamManagement` (rx/tx stream association).
    Moreover, it creates boolean np.ndarrays for `slot-ue allocation mask` and `ue-ap allocation masks`.
    This base class gives the simplest possible grid. Subclass this for other allocation strategies.

    :param c: SharedParams, shared scenario parameters (num_ue, num_ap, num_sym, num_scs, ...).
    :param wireless: WirelessParams, OFDM numerology parameters (sc_spacing, cp_time, ...).
    :param geo: GeometryParams, geometric configuration (not used by the base allocation, but
        available for subclasses that allocate resources based on UE position/region).
    :param estimate: EstimationParams, current position/region estimates (same rationale as `geo`).
    """

    def __init__(self, c: SharedParams, wireless: WirelessParams,
                 geo: GeometryParams, estimate: EstimationParams):
        super().__init__(c, wireless, geo, estimate)

    def slot_ue_association_mask(self) -> np.ndarray:
        """Orthogonal slot allocation based on predicted semantic risk and predicted SE.

        :returns: np.ndarray[bool], shape (S, U), True where UE u is allocated to slot s, else False.
        """
        # General mask
        mask = np.zeros((self.c.num_slots, self.c.num_ue), dtype=bool)
        # Order UEs by delta_risk (descending)
        order = np.argsort(-self.delta_risk)
        # Take slot indexing of available slots (cap to the no. of UEs)
        num_slots = min(self.c.num_slots, self.c.num_ue)
        # Schedule the nodes
        mask[np.arange(num_slots), order[:num_slots]] = True
        return mask

    def __call__(self, frame: int, **kwargs) -> np.ndarray:
        """Call the association for the current frame.

        :returns: np.ndarray[bool], ue-slot-association mask.
        """
        if frame < 0:
            raise ValueError("Frame index cannot be negative.")

        self.risk0, self.risk1, self.delta_risk = self.eval_semantic_risk(frame, **ut.filter_kwargs(self.eval_semantic_risk, **kwargs))
        return self.slot_ue_association_mask()


class PebBasedAllocationNoComm(ResourceAllocationBase):
    """PEB-based OFDM allocation only localization.

    Builds the sionna.phy `ResourceGrid` (time/frequency resource description),  a
    `StreamManagement` (rx/tx stream association).
    Moreover, it creates boolean np.ndarrays for `slot-ue allocation mask` and `ue-ap allocation masks`.
    This base class gives the simplest possible grid. Subclass this for other allocation strategies.

    :param c: SharedParams, shared scenario parameters (num_ue, num_ap, num_sym, num_scs, ...).
    :param wireless: WirelessParams, OFDM numerology parameters (sc_spacing, cp_time, ...).
    :param geo: GeometryParams, geometric configuration (not used by the base allocation, but
        available for subclasses that allocate resources based on UE position/region).
    :param estimate: EstimationParams, current position/region estimates (same rationale as `geo`).
    """
    def __init__(self, c: SharedParams, wireless: WirelessParams,
                 geo: GeometryParams, estimate: EstimationParams):
        super().__init__(c, wireless, geo, estimate)

    def slot_ue_association_mask(self) -> np.ndarray:
        """Orthogonal slot allocation based on predicted semantic risk and predicted SE.

        :returns: np.ndarray[bool], shape (S, U), True where UE u is allocated to slot s, else False.
        """
        # General mask
        mask = np.zeros((self.c.num_slots, self.c.num_ue), dtype=bool)
        # Order UEs by delta_risk (descending)
        order = np.argsort(-self.delta_risk)
        # Take slot indexing of available slots (cap to the no. of UEs)
        num_slots = min(self.c.num_slots, self.c.num_ue)
        # Schedule the nodes
        mask[np.arange(num_slots), order[:num_slots]] = True
        return mask

    def __call__(self, frame: int, **kwargs) -> np.ndarray:
        """Call the association for the current frame.

        :returns: np.ndarray[bool], ue-slot-association mask.
        """
        if frame < 0:
            raise ValueError("Frame index cannot be negative.")

        self.risk0, self.risk1, self.delta_risk = eval_speb(**ut.filter_kwargs(eval_speb, **kwargs))
        return self.slot_ue_association_mask()



class QueueBasedAllocation(ResourceAllocationBase):
    """Semantic risk-based OFDM allocation.

    Builds the sionna.phy `ResourceGrid` (time/frequency resource description),  a
    `StreamManagement` (rx/tx stream association).
    Moreover, it creates boolean np.ndarrays for `slot-ue allocation mask` and `ue-ap allocation masks`.
    This base class gives the simplest possible grid. Subclass this for other allocation strategies.

    :param c: SharedParams, shared scenario parameters (num_ue, num_ap, num_sym, num_scs, ...).
    :param wireless: WirelessParams, OFDM numerology parameters (sc_spacing, cp_time, ...).
    :param geo: GeometryParams, geometric configuration (not used by the base allocation, but
        available for subclasses that allocate resources based on UE position/region).
    :param estimate: EstimationParams, current position/region estimates (same rationale as `geo`).
    """

    def __init__(self, c: SharedParams, wireless: WirelessParams,
                 geo: GeometryParams, estimate: EstimationParams):
        super().__init__(c, wireless, geo, estimate)
        # Specific attributes
        self._last_allocated_ues = None  #np.zeros(self.c.num_slots, dtype=np.int32)
        self.delta_queue = None

    def slot_ue_association_mask(self) -> np.ndarray:
        """Orthogonal slot allocation based on predicted semantic risk and predicted SE.

        :returns: np.ndarray[bool], shape (S, U), True where UE u is allocated to slot s, else False.
        """
        # TODO: check how the mean value of the distribution would change the order, now it is assume same traffic mean
        # General mask
        mask = np.zeros((self.c.num_slots, self.c.num_ue), dtype=bool)
        # Order UE based the delta_queue (ascending)
        order = np.argsort(self.delta_queue)
        # Schedule the remaining nodes (up to the remaining slots)
        mask[np.arange(np.minimum(self.c.num_slots, self.c.num_ue)), order[:self.c.num_slots]] = True
        return mask

    def __call__(self, frame: int, **kwargs) -> np.ndarray:
        """Call the association for the current frame.

        :returns: np.ndarray[bool], ue-slot-association mask.
        """
        if frame < 0:
            raise ValueError("Frame index cannot be negative.")

        self.delta_queue, _ = self.eval_comm_risk(**ut.filter_kwargs(self.eval_comm_risk, **kwargs))
        return self.slot_ue_association_mask()



available_methods = {"MAF": RoundRobinAllocation,
                     "SRI": SemanticRiskAllocation,
                     "SRL": SemanticRiskAllocationNoComm,
                     "PBI": PebBasedAllocation,
                     "QBC": QueueBasedAllocation,
                     "PBL": PebBasedAllocationNoComm,
                     }