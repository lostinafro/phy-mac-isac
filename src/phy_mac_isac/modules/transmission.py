import sionna.phy as phy
from sionna.sys import PHYAbstraction, InnerLoopLinkAdaptation
import numpy as np
import torch
from . import params
from .resource_allocation import ResourceAllocationBase
from abc import ABC
from dataclasses import dataclass

class TransmissionBase(ABC):
    """Abstract base class for a physical-layer transmission scheme -- something that takes a
    channel realization plus a resource grid / stream management and produces a transmission
    outcome (bits sent, bits recovered, channel estimate, etc.).

    Subclasses must implement `__call__`. Instantiating `TransmissionBase` directly raises
    `TypeError`, as with any ABC.

    :param wireless: WirelessParams, OFDM numerology/noise parameters shared by any concrete
        transmission scheme.
    """

    def __init__(self,
                 wireless: params.WirelessParams,
                 ra_module: ResourceAllocationBase,
                 device: torch.device | str = "cpu"):
        self.wireless = wireless
        self.ra_module = ra_module
        self.device = torch.device(device)
        # TODO: set a default for device

    def __call__(self, channel: np.ndarray, tx_power: np.ndarray | float | None = None,
                 no: float | None = None, **kwargs) -> tuple:
        """Run one transmission over `channel` using `resource_grid`/`stream_management`.

        :returns: tuple of transmission outputs (exact contents defined by the subclass).
        """
        raise NotImplementedError

# TODO: future should implement per-UE modulation below for PHY accurate tests.
class TransmissionCFR(TransmissionBase):
    """Simulates one uplink OFDM transmission entirely in the frequency domain using
    sionna.phy blocks: bits -> LDPC encode -> QAM map -> resource grid -> apply CFR channel
    -> LS channel estimation -> LMMSE equalization -> demap -> LDPC decode.

    :param wireless: WirelessParams, used for a default noise level if `no` isn't given.
    :param modulation: str, one of {"BPSK","QPSK","16QAM","64QAM","256QAM"}. Default "QPSK".
    :param coderate: float, LDPC code rate (k/n). Default 0.5. Must fall within the range
        supported by Sionna's 5G LDPC encoder for the resulting `n` (roughly 1/5 to 0.95).        
    """

    _MOD_BITS = {"BPSK": 1, "QPSK": 2, "16QAM": 4, "64QAM": 6, "256QAM": 8}
    # TODO: set the class to be able to be called with different num_bit_per_sym, avoid naming the modulation, and output the empirical SE

    def __init__(self, wireless: params.WirelessParams, ra_module: ResourceAllocationBase, device: torch.device,
                 modulation: str = "QPSK", coderate: float = 0.5):
        super().__init__(wireless, ra_module, device)
        try:
            self.num_bits_per_symbol = self._MOD_BITS[modulation.upper()]
        except KeyError:
            raise ValueError(f"Unsupported modulation '{modulation}'. Supported: {list(self._MOD_BITS)}")
        self.coderate = coderate

        self.binary_source = phy.mapping.BinarySource()
        self.mapper = phy.mapping.Mapper("qam", self.num_bits_per_symbol)
        self.demapper = phy.mapping.Demapper("app", "qam", self.num_bits_per_symbol)
        self.apply_channel = phy.channel.ApplyOFDMChannel()

    def __call__(self,
                 channel: np.ndarray | torch.Tensor,
                 tx_power: np.ndarray | float | None = None,
                 no: float | None = None,
                 **kwargs) -> tuple:
        """Run one uplink OFDM transmission over a given frequency channel realization.

        :param channel: complex CFR for a *single slot*, shape (S, L, M, U, T, F) --
            (slots=S, num_rx=APs, num_rx_ant, num_tx=UEs, num_ofdm_symbols, fft_size). This matches
            the per-slot slice of `compute_cfr`'s [L, M, U, S, T, F] output using S as it where batches
        :param resource_allocation: ResourceAllocationBase, instance of the resource allocation algorithm used
         :param tx_power: np.ndarray | float | None, per-UE transmit power (linear), shape
            (U, ) or scalar. Defaults to `self.wireless.ue_power`.
        :param no: float | None, noise power spectral density (linear). If None, defaults to `self.wireless.n0b_ap`.
        :returns: tuple (h_hat, b, b_hat, x, x_hat) -- LS channel estimate, transmitted bits,
            decoded bits, transmitted frequency-domain samples, decoded frequency-domain samples.
        """
        if tx_power is None:
            tx_power = self.wireless.ue_power
        if no is None:
            no = self.wireless.nob_ap

        resource_grid = self.ra_module.rg
        stream_management = self.ra_module.sm

        n = int(resource_grid.num_data_symbols * self.num_bits_per_symbol)
        k = int(n * self.coderate)
        encoder = phy.fec.ldpc.LDPC5GEncoder(k, n)
        decoder = phy.fec.ldpc.LDPC5GDecoder(encoder, hard_out=True)

        rg_mapper = phy.ofdm.ResourceGridMapper(resource_grid)
        ls_est = phy.ofdm.LSChannelEstimator(resource_grid, interpolation_type="nn")
        mf_equ = phy.ofdm.MFEqualizer(resource_grid, stream_management)

        # Bits -> encoded bits -> QAM symbols -> resource grid, batch size 1
        b = self.binary_source([self.wireless.c.num_slots, resource_grid.num_tx, resource_grid.num_streams_per_tx, k])
        c = encoder(b)
        x_sym = self.mapper(c)
        x = rg_mapper(x_sym)                                            # [S, U, 1, T, F]

        # Scale to actual per-UE transmit power (Mapper produces unit-energy symbols by default)
        tx_power = np.reshape(np.broadcast_to(tx_power, (resource_grid.num_tx,)), (1, -1, 1, 1, 1))
        x = x * np.sqrt(tx_power)

        # Add the singleton tx-antenna axis and batch axis expected by ApplyOFDMChannel
        h = channel.astype(np.complex64)  # [S, 1, L*M, U, 1 (num_streams), T, F]
        y = self.apply_channel(x, h, no)                                 # [S, 1, L*M, T, F]

        h_hat, err_var = ls_est(y, no)
        x_hat, no_eff = mf_equ(y, h_hat, err_var, no)
        llr = self.demapper(x_hat, no_eff)
        b_hat = decoder(llr)

        return h_hat, b, b_hat, x, x_hat, no_eff


class SEuplinkOFDMpostMRC(TransmissionBase):
    """PHY-layer abstraction of an uplink OFDM transmission: computes the post-combining
    SINR (and Shannon spectral efficiency) from LS channel estimation and MR (matched-filter)
    combining, without any bit-level LDPC encoding/decoding or QAM mapping/demapping.
    Substantially cheaper than TransmissionCFR when only link-quality metrics are needed,
    in the spirit of Sionna SYS's PHY-abstraction functionality.

    :param wireless: WirelessParams, used for a default noise level if `no` isn't given.
    """

    def __init__(self, wireless: params.WirelessParams, ra_module: ResourceAllocationBase, device: torch.device,
                 compiled_core: bool = False):
        super().__init__(wireless, ra_module, device)
        self.apply_channel = phy.channel.ApplyOFDMChannel()
        self.rg_mapper = phy.ofdm.ResourceGridMapper(ra_module.rg)
        self.ls_est = phy.ofdm.LSChannelEstimator(ra_module.rg, interpolation_type="nn")
        self.mf_equ = phy.ofdm.MFEqualizer(ra_module.rg, ra_module.sm)

        # Moving the blocks to the right device
        for block in (self.rg_mapper, self.ls_est, self.mf_equ):
            if isinstance(block, torch.nn.Module):
                block.to(self.device)

        self._run_core = self._core

        # Compile the core method if possible (Eager mode was find slightly faster)
        if compiled_core:
            compile_mode = "reduce-overhead" if self.device.type == "cuda" else "default"
            self._run_core = torch.compile(self._core, mode=compile_mode)

    def _core(self, h: torch.Tensor, tx_power: torch.Tensor, no: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """All-tensor hot path -- this is the part that gets compiled in CUDA."""
        # Data REs left at zero -- LS estimation and no_eff depend only on pilots + the
        # channel/noise geometry, not on actual data content, so real symbols aren't needed.
        x_sym = torch.zeros((self.wireless.c.num_slots,
                             self.ra_module.rg.num_tx,
                             self.ra_module.rg.num_streams_per_tx,
                             self.ra_module.rg.num_data_symbols),
                            dtype=torch.complex64, device=self.device)

        x = self.rg_mapper(x_sym)  # (S, U, 1, T, F)

        power = torch.reshape(torch.broadcast_to(tx_power, (self.ra_module.rg.num_tx,)), (1, -1, 1, 1, 1))
        x = x * torch.sqrt(power)

        y = self.apply_channel(x, h, no)

        h_hat, err_var = self.ls_est(y, no)
        _, no_eff = self.mf_equ(y, h_hat, err_var, no)  # (S, U, 1, num_data_symbols)

        sinr = torch.squeeze(1.0 / no_eff, dim=-2)  # (S, U, num_data_symbols)

        se = torch.mean(torch.log2(1.0 + sinr), dim=-1)  # (S, U)

        return sinr, se

    @torch.no_grad()
    def __call__(self, channel: np.ndarray, tx_power: np.ndarray | float | None = None,
                 no: float | None = None, **kwargs) -> tuple:
        """Estimate per-(slot, UE) SINR and spectral efficiency, skipping bit-level encoding.

        :param channel: complex CFR, shape (S, 1, L*M, U, 1, T, F) -- same convention as
            TransmissionCFR.
        :param resource_allocation: ResourceAllocationBase, providing `.rg`/`.sm`.
        :param tx_power: np.ndarray | float | None, per-UE transmit power (linear), shape
            (U, ) or scalar. Defaults to `self.wireless.ue_power`.
        :param no: float | None, noise PSD (linear). Defaults to `self.wireless.n0b_ap`.

        :returns: tuple (sinr, se) --
            sinr: np.ndarray, shape (S, U, num_data_symbols), post-MR-combining SINR (linear).
            se: np.ndarray, shape (S, U), Shannon spectral efficiency (bits/s/Hz), averaged
                over the UE's data resource elements.
        """
        # Transform the input to torch
        no_t = torch.as_tensor(no if no is not None else self.wireless.nob_ap, dtype=torch.float32, device=self.device)
        tx_power_t = torch.as_tensor(tx_power if tx_power is not None else self.wireless.ue_power,
                                     dtype=torch.float32, device=self.device)
        h = torch.as_tensor(channel, dtype=torch.complex64, device=self.device)

        # Call the core method
        sinr, se = self._run_core(h, tx_power_t, no_t)
        return sinr.cpu().numpy(), se.cpu().numpy()


@dataclass(frozen=True)
# TODO: To be merged with the general results as metrics
class SisoTransmissionResult:
    """Compact slot results; every vector has shape [S]. SINR is [S,T,F,1,1]."""

    sinr: torch.Tensor
    mcs_index: torch.Tensor
    decoded_bits: torch.Tensor
    harq_feedback: torch.Tensor
    sinr_eff: torch.Tensor
    tbler: torch.Tensor
    bler: torch.Tensor

    @property
    def decoded(self) -> torch.Tensor:
        return self.harq_feedback == 1


class SisoPhyAbstraction(TransmissionBase):
    """One SYS transport block per independent SISO slot.

    With perfect CSI, SINR = P_RE * |h|**2 / noise_power_per_subcarrier.
    Optional estimated CSI drives ILLA separately from the physical channel.
    Residual estimation error is approximated as uncorrelated noise P_RE*MSE.
    This bypasses waveform decoding; SYS samples ACK/NACK from transport BLER.
    Reserved REs are masked to zero and excluded from transport-block sizing.
    """

    def __init__(self,
                 wireless: params.WirelessParams,
                 ra_module: ResourceAllocationBase,
                 device: torch.device | str = "cpu",
                 data_mask: torch.Tensor | None = None):
         #resource_grid, device: torch.device | str = "cpu", *,
         #bler_target: float = 0.1, direction: str = "uplink",
         #mcs_table_index: int = 1, data_mask=None):
        super().__init__(wireless, ra_module, device)

        # Just to let it less verbose
        self.rg = self.ra_module.rg
        
        # Check input
        if self.rg.num_tx != 1 or self.rg.num_streams_per_tx != 1:
            raise ValueError("The resource grid must have one transmitter and stream")
        
        # Resource grid shape
        shape = (self.rg.num_ofdm_symbols, self.rg.fft_size)

        # Create mask of data symbols from the resource grid
        # ResourceGrid type 0 identifies data (1 pilots, 2 guard, 3 DC).
        if data_mask is None:
            data_mask = self.rg.build_type_grid()[0, 0] == 0
        self.data_mask = torch.as_tensor(data_mask, dtype=torch.bool, device=self.device)
        #  Check if data contains REs
        if tuple(self.data_mask.shape) != shape or not self.data_mask.any():
            raise ValueError(f"data_mask must have shape {shape} and contain data REs")

        # Setup PhyAbstraction and LinkAdaptation modules
        self.posteq_sinr = phy.ofdm.LMMSEPostEqualizationSINR(self.ra_module.rg,
                                                              self.ra_module.sm,
                                                              device=str(self.device))
        self.phy_abstraction = PHYAbstraction(device=str(self.device))
        self.illa = InnerLoopLinkAdaptation(self.phy_abstraction,
                                            bler_target=self.wireless.bler_target,
                                            device=str(self.device))

    @torch.no_grad()
    def __call__(self,
                 channel: np.ndarray | torch.Tensor,
                 tx_power: np.ndarray | float | None = None,
                 no: float | None = None, *,
                 channel_estimate: torch.Tensor | None = None,
                 error_variance: float = 0.) -> tuple:
        """Accept [S,1,1,1,1,T,F]; tx_power/no are linear powers per RE.

        Both powers accept a scalar or an [S] vector. ``no`` is noise power
        over one subcarrier, not noise PSD or total bandwidth noise.
        ``channel_estimate`` has the same shape as channel and is used for
        MCS selection only; channel still controls reception. ``error_variance``
        is a scalar or [S] absolute complex-coefficient MSE. At nonzero MSE,
        both SINRs use noise + P_RE*MSE, an abstraction of estimation mismatch
        rather than a bit-level implementation of an imperfect equalizer.
        """
        # Check exact channel
        h = torch.as_tensor(channel, device=self.device)
        expected = (1, 1, 1, 1, self.rg.num_ofdm_symbols, self.rg.fft_size)
        if h.ndim != 7 or tuple(h.shape[1:]) != expected:
            raise ValueError(f"channel must have shape [S,{','.join(map(str, expected))}]")
        if not torch.is_complex(h) or not torch.isfinite(h).all():
            raise ValueError("channel must contain finite complex coefficients")

        # Check estimated channel
        if channel_estimate is not None:
            h_hat = torch.as_tensor(channel_estimate, device=self.device)
            if h_hat.shape != h.shape or not torch.is_complex(h_hat) or not torch.isfinite(h_hat).all():
                raise ValueError("channel_estimate must be finite, complex and have the same shape as channel")
        else:
            h_hat = None

        # Transform the other inputs to torch if needed
        no_t = torch.as_tensor(no if no is not None else self.wireless.nob_ap, dtype=torch.float32, device=self.device)
        tx_power_t = torch.as_tensor(tx_power if tx_power is not None else self.wireless.ue_power,
                                     dtype=torch.float32, device=self.device)

        results = self._core(h, tx_power_t, no_t, h_hat)
        return tuple([r.cpu().numpy() for r in results])

    def _core(self, h: torch.Tensor, tx_power: torch.Tensor, no: torch.Tensor, h_hat: torch.Tensor | None = None) -> tuple:
        """All-tensor hot path -- this is the part that gets compiled in CUDA eventually."""
        power = torch.reshape(torch.broadcast_to(tx_power, (self.ra_module.rg.num_tx,)), (1, -1, 1, 1, 1))
        h_eff = torch.sqrt(power) * h
        h_hat_eff = h_eff if h_hat is None else h_hat * torch.sqrt(power)

        # Compute SINR assuming LMMSE equalization
        sinr = self.posteq_sinr(h_eff, no, h_hat_eff)

        # Create Link adaptation and simulate transmission
        kwargs = dict(sinr=sinr,
                      mcs_table_index=self.wireless.mcs_table_index,
                      mcs_category=self.wireless.mcs_category)
        mcs_index = self.illa(**kwargs)
        decoded_bits, harq, sinr_eff, tbler, bler = self.phy_abstraction(mcs_index, **kwargs)

        return (sinr, mcs_index.squeeze(-1), decoded_bits.squeeze(-1), harq.squeeze(-1),
                sinr_eff.squeeze(-1), tbler.squeeze(-1), bler.squeeze(-1))

        slots = h.shape[0]
        powers = []
        for name, value, strictly_positive in (("tx_power", tx_power, False), ("no", no, True),
                                               ("error_variance", error_variance, False)):
            tensor = torch.as_tensor(value, dtype=h.real.dtype, device=self.device)
            if tensor.ndim > 1 or (tensor.ndim == 1 and tensor.shape[0] != slots):
                raise ValueError(f"{name} must be a scalar or a vector of length S")
            if not torch.isfinite(tensor).all() or (tensor < 0).any() or (
                    strictly_positive and (tensor == 0).any()):
                raise ValueError(f"{name} must be finite and {'positive' if strictly_positive else 'nonnegative'}")
            powers.append(torch.broadcast_to(tensor, (slots,)))
        power, noise, variance = powers
        effective_noise = noise + power * variance
        def make_sinr(cfr):
            gain = cfr[:, 0, 0, 0, 0].abs().square()
            raw = gain * (power / effective_noise)[:, None, None]
            return torch.where(self.data_mask, raw, 0)[..., None, None]

        sinr = make_sinr(h)
        predicted_sinr = sinr if channel_estimate is None else make_sinr(h_hat)
        if not torch.isfinite(sinr).all() or not torch.isfinite(predicted_sinr).all():
            raise ValueError("SINR overflow: check channel and power scaling")
        if slots == 0:
            integer = torch.empty(0, dtype=torch.int32, device=self.device)
            real = torch.empty(0, dtype=h.real.dtype, device=self.device)
            return SisoTransmissionResult(sinr, integer, integer.clone(), integer.clone(),
                                          real, real.clone(), real.clone())
        kwargs = dict(sinr=sinr, mcs_table_index=self.mcs_table_index,
                      mcs_category=self.mcs_category)
        mcs_index = self.illa(sinr=predicted_sinr, mcs_table_index=self.mcs_table_index,
                              mcs_category=self.mcs_category)
        decoded_bits, harq, sinr_eff, tbler, bler = self.phy_abstraction(mcs_index, **kwargs)

        # SYS interprets all-zero SINR as an unscheduled user. Here the batch
        # contains only occupied slots: a zero-power/deep-outage slot is a NACK.
        outage = ~torch.any(sinr > 0, dim=(1, 2, 3, 4))
        outage = outage[:, None]
        decoded_bits = torch.where(outage, 0, decoded_bits)
        harq = torch.where(outage, 0, harq)
        sinr_eff = torch.where(outage, 0, sinr_eff)
        tbler = torch.where(outage, 1, tbler)
        bler = torch.where(outage, 1, bler)
        return (sinr, mcs_index.squeeze(-1), decoded_bits.squeeze(-1), harq.squeeze(-1),
            sinr_eff.squeeze(-1), tbler.squeeze(-1), bler.squeeze(-1))
