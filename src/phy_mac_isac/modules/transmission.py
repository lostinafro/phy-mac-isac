import sionna.phy as phy
import numpy as np
import torch
from . import params
from . import resource_allocation as ra
from abc import ABC, abstractmethod

class TransmissionBase(ABC):
    """Abstract base class for a physical-layer transmission scheme -- something that takes a
    channel realization plus a resource grid / stream management and produces a transmission
    outcome (bits sent, bits recovered, channel estimate, etc.).

    Subclasses must implement `__call__`. Instantiating `TransmissionBase` directly raises
    `TypeError`, as with any ABC.

    :param wireless: WirelessParams, OFDM numerology/noise parameters shared by any concrete
        transmission scheme.
    """

    def __init__(self, wireless: params.WirelessParams, ra_module: ra.ResourceAllocationBase, device: torch.device):
        self.wireless = wireless
        self.ra_module = ra_module
        self.device = device

    @abstractmethod
    def __call__(self, channel: np.ndarray, tx_power: np.ndarray | float | None = None,
                 no: float | None = None, **kwargs) -> tuple:
        """Run one transmission over `channel` using `resource_grid`/`stream_management`.

        :returns: tuple of transmission outputs (exact contents defined by the subclass).
        """
        raise NotImplementedError

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

    def __init__(self, wireless: params.WirelessParams, ra_module: ra.ResourceAllocationBase, device: torch.device,
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

    def __call__(self, channel: np.ndarray, tx_power: np.ndarray | float | None = None,
                 no: float | None = None, **kwargs) -> tuple:
        """Run one uplink OFDM transmission over a given frequency channel realization.

        :param channel: complex CFR for a *single slot*, shape (S, L, M, U, T, F) --
            (slots=S, num_rx=APs, num_rx_ant, num_tx=UEs, num_ofdm_symbols, fft_size). This matches
            the per-slot slice of `compute_cfr`'s [L, M, U, S, T, F] output using S as it where batches
        :param resource_allocation: ResourceAllocationBase, instance of the resource allocation algorithm used
         :param tx_power: np.ndarray | float | None, per-UE transmit power (linear), shape
            (U, ) or scalar. Defaults to `self.wireless.ue_power`.
        :param no: float | None, noise power spectral density (linear). If None, defaults to
            `self.wireless.n0b_ap` -- check this scaling matches your link budget before relying
            on the default; it's a placeholder, not a validated noise-power derivation.

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

    def __init__(self, wireless: params.WirelessParams, ra_module: ra.ResourceAllocationBase, device: torch.device):
        super().__init__(wireless, ra_module, device)
        self.apply_channel = phy.channel.ApplyOFDMChannel()
        self.rg_mapper = phy.ofdm.ResourceGridMapper(ra_module.rg)
        self.ls_est = phy.ofdm.LSChannelEstimator(ra_module.rg, interpolation_type="nn")
        self.mf_equ = phy.ofdm.MFEqualizer(ra_module.rg, ra_module.sm)

        # Moving the blocks to the right device
        for block in (self.rg_mapper, self.ls_est, self.mf_equ):
            if isinstance(block, torch.nn.Module):
                block.to(self.device)

        # Compile the core method if possible
        compile_mode = "reduce-overhead" if torch.cuda.is_available() else "default"
        self._compiled_core = torch.compile(self._core, mode=compile_mode)

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
        sinr, se = self._core(h, tx_power_t, no_t)
        return sinr.cpu().numpy(), se.cpu().numpy()