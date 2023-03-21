import torch as th
import torch.nn as nn
from fractions import Fraction
import math
from torch.nn import functional as F

from transformer import CrossTransformerEncoder
from layers_demucs import rescale_module
from states import capture_init
from layers_hdemucs import pad1d, ScaledEmbedding, HEncLayer, MultiWrap, HDecLayer


class demucs_phase(nn.Module):
    def __init__(self, 
                sources,
                # Channels
                audio_channels=2,
                channels=48,
                channels_time=None,
                growth=2,
                # STFT
                nfft=4096,
                wiener_iters=0,
                end_iters=0,
                wiener_residual=False,
                cac=True,
                # Main structure
                depth=4,
                rewrite=True,
                # Frequency branch
                multi_freqs=None,
                multi_freqs_depth=3,
                freq_emb=0.2,
                emb_scale=10,
                emb_smooth=True,
                # Convolutions
                kernel_size=8,
                phase_stride=2,
                stride=4,
                context=1,
                context_enc=0,
                # Normalization
                norm_starts=4,
                norm_groups=4,
                # DConv residual branch
                dconv_mode=1,
                dconv_depth=2,
                dconv_comp=8,
                dconv_init=1e-3,
                # Before the Transformer
                bottom_channels=0,
                # Transformer
                t_layers=5,
                t_emb="sin",
                t_hidden_scale=4.0,
                t_heads=8,
                t_dropout=0.0,
                t_max_positions=10000,
                t_norm_in=True,
                t_norm_in_group=False,
                t_group_norm=False,
                t_norm_first=True,
                t_norm_out=True,
                t_max_period=10000.0,
                t_weight_decay=0.0,
                t_lr=None,
                t_layer_scale=True,
                t_gelu=True,
                t_weight_pos_embed=1.0,
                t_sin_random_shift=0,
                t_cape_mean_normalize=True,
                t_cape_augment=True,
                t_cape_glob_loc_scale=[5000.0, 1.0, 1.4],
                t_sparse_self_attn=False,
                t_sparse_cross_attn=False,
                t_mask_type="diag",
                t_mask_random_seed=42,
                t_sparse_attn_window=500,
                t_global_window=100,
                t_sparsity=0.95,
                t_auto_sparsity=False,
                # ------ Particuliar parameters
                t_cross_first=False,
                # Weight init
                rescale=0.1,
                # Metadata
                samplerate=44100,
                segment=10,
                use_train_segment=True,       
                        ):
        #cac: uses complex as channels, i.e. complex numbers are 2 channels each
              #  in input and output. no further processing is done before ISTFT.
        #use_train_segment: (bool) if True, the actual size that is used during the
                #training is used during inference.
        super(demucs_phase, self).__init__()
        self.cac = cac
        self.wiener_residual = wiener_residual
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.bottom_channels = bottom_channels
        self.channels = channels
        self.samplerate = samplerate
        self.segment = segment
        self.use_train_segment = use_train_segment
        self.nfft = nfft
        self.hop_length = nfft // 4
        self.wiener_iters = wiener_iters
        self.end_iters = end_iters
        self.freq_emb = None
        
        self.encoder_z = nn.ModuleList()
        self.decoder_z = nn.ModuleList()
        
        self.encoder_p = nn.ModuleList()
        self.decoder_p = nn.ModuleList()
        
        chin_p = audio_channels
        chin_z = chin_p 
        if self.cac:
            chin_z *= 2
        chout_p = channels_time or channels
        chout_z = channels
        freqs = nfft // 2
        for index in range(depth):
            norm = index >= norm_starts
            freq = freqs > 1
            stri = stride
            ker = kernel_size
            
            if not freq:
                assert freqs == 1
                ker = phase_stride * 2
                stri = phase_stride
            
            pad = True
            last_freq = False
            if freq and freqs <= kernel_size:
                ker = freqs
                pad = False
                last_freq = True
    
    
            kw = {
                "kernel_size": ker,
                "stride": stri,
                "freq": freq,
                "pad": pad,
                "norm": norm,
                "rewrite": rewrite,
                "norm_groups": norm_groups,
                "dconv_kw": {
                    "depth": dconv_depth,
                    "compress": dconv_comp,
                    "init": dconv_init,
                    "gelu": True,
                },
            }
            kw_dec = dict(kw)
            multi = False
            if multi_freqs and index < multi_freqs_depth:
                multi = True
                kw_dec["context_freq"] = False

            if last_freq:
                chout_z = max(chout_p, chout_z)
                chout_p = chout_z
    
            enc_z = HEncLayer(
                chin_z, chout_z, dconv=dconv_mode & 1, context=context_enc, **kw
            )
    
            enc_p = HEncLayer(
                chin_p, chout_p, dconv=dconv_mode & 1, context=context_enc, **kw
            )
    
            if multi:
                enc_z = MultiWrap(enc_z, multi_freqs)
                
                enc_p = MultiWrap(enc_p, multi_freqs)
            
            self.encoder_z.append(enc_z)
            self.encoder_p.append(enc_p)
    
    
            if index == 0:
                chin_p = self.audio_channels * len(self.sources)
                chin_z = chin_p
#                 if self.cac:
#                     chin_z *= 2
#                     chin_p *= 2
    
            dec_z = HDecLayer(
                chout_z,
                chin_z,
                dconv=dconv_mode & 2,
                last=index == 0,
                context=context,
                **kw_dec
            )
            dec_p = HDecLayer(
                chout_p,
                chin_p,
                dconv=dconv_mode & 2,
                last=index == 0,
                context=context,
                **kw_dec
            )
        
    
            if multi:
                dec_z = MultiWrap(dec_z, multi_freqs)
                dec_p = MultiWrap(dec_p, multi_freqs)
    
            self.decoder_z.insert(0, dec_z)
            self.decoder_p.insert(0, dec_p)
            
            chin_p = chout_p
            chin_z = chout_z
            chout_p = int(growth * chout_p)
            chout_z = int(growth * chout_z)
            
            #
            if freq:
                if freqs <= kernel_size:
                    freqs = 1
                else:
                    freqs //= stride
            if index == 0 and freq_emb:
                self.freq_emb = ScaledEmbedding(
                    freqs, chin_z, smooth=emb_smooth, scale=emb_scale
                )
                self.freq_emb_scale = freq_emb
                
            #
                self.phase_emb = ScaledEmbedding(
                    freqs, chin_p, smooth=emb_smooth, scale=emb_scale
                )
                self.phase_emb_scale = freq_emb
            #
            
            if rescale:
                rescale_module(self, reference=rescale)
            
            transformer_channels = channels * growth ** (depth - 1)
            
            if bottom_channels:
                self.channel_upsampler = nn.Conv1d(transformer_channels, bottom_channels, 1)
                self.channel_downsampler = nn.Conv1d(
                    bottom_channels, transformer_channels, 1
                )
                self.channel_upsampler_t = nn.Conv1d(
                    transformer_channels, bottom_channels, 1
                )
                self.channel_downsampler_t = nn.Conv1d(
                    bottom_channels, transformer_channels, 1
                )

                transformer_channels = bottom_channels
            
            if t_layers > 0:
                self.crosstransformer = CrossTransformerEncoder(
                    dim=transformer_channels,
                    emb=t_emb,
                    hidden_scale=t_hidden_scale,
                    num_heads=t_heads,
                    num_layers=t_layers,
                    cross_first=t_cross_first,
                    dropout=t_dropout,
                    max_positions=t_max_positions,
                    norm_in=t_norm_in,
                    norm_in_group=t_norm_in_group,
                    group_norm=t_group_norm,
                    norm_first=t_norm_first,
                    norm_out=t_norm_out,
                    max_period=t_max_period,
                    weight_decay=t_weight_decay,
                    lr=t_lr,
                    layer_scale=t_layer_scale,
                    gelu=t_gelu,
                    sin_random_shift=t_sin_random_shift,
                    weight_pos_embed=t_weight_pos_embed,
                    cape_mean_normalize=t_cape_mean_normalize,
                    cape_augment=t_cape_augment,
                    cape_glob_loc_scale=t_cape_glob_loc_scale,
                    sparse_self_attn=t_sparse_self_attn,
                    sparse_cross_attn=t_sparse_cross_attn,
                    mask_type=t_mask_type,
                    mask_random_seed=t_mask_random_seed,
                    sparse_attn_window=t_sparse_attn_window,
                    global_window=t_global_window,
                    sparsity=t_sparsity,
                    auto_sparsity=t_auto_sparsity,
                )
            else:
                self.crosstransformer = None
    
    
    def __spectr(self, x, n_fft=512, hop_length=None, pad=0): # spectr tra 
        *other, length = x.shape
        x = x.reshape(-1, length)
        z = th.stft(x,
                n_fft * (1 + pad),
                hop_length or n_fft // 4,
                window=th.hann_window(n_fft).to(x),
                win_length=n_fft,
                normalized=True,
                center=True,
                return_complex=True,
                pad_mode='reflect')
        _, freqs, frame = z.shape
        return z.view(*other, freqs, frame)
    
    
    def __ispectr(self, z, hop_length=None, length=None, pad=0):
        *other, freqs, frames = z.shape
        n_fft = 2 * freqs - 2
        z = z.view(-1, freqs, frames)
        win_length = n_fft // (1 + pad)
        x = th.istft(z,
                 n_fft,
                 hop_length,
                 window=th.hann_window(win_length).to(z.real),
                 win_length=win_length,
                 normalized=True,
                 length=length,
                 center=True)
        _, length = x.shape
        return x.view(*other, length)
    
    
    def _spectr(self, x):
        hl = self.hop_length
        nfft = self.nfft
        x0 = x  # noqa

        # We re-pad the signal in order to keep the property
        # that the size of the output is exactly the size of the input
        # divided by the stride (here hop_length), when divisible.
        # This is achieved by padding by 1/4th of the kernel size (here nfft).
        # which is not supported by torch.stft.
        # Having all convolution operations follow this convention allow to easily
        # align the time and frequency branches later on.
        assert hl == nfft // 4
        le = int(math.ceil(x.shape[-1] / hl))
        pad = hl // 2 * 3
        x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")

        z = self.__spectr(x, nfft, hl)[..., :-1, :]
        
        assert z.shape[-1] == le + 4, (z.shape, x.shape, le)
        z = z[..., 2: 2 + le]
        return z
    
    
    def _ispectr(self, z, length=None, scale=0):
        hl = self.hop_length // (4**scale)
        z = F.pad(z, (0, 0, 0, 1))
        z = F.pad(z, (2, 2))
        pad = hl // 2 * 3
        le = hl * int(math.ceil(length / hl)) + 2 * pad
        x = self.__ispectr(z, hl, length=le)
        x = x[..., pad: pad + length]
        return x
    
    
    def _magnitude(self, z):
        if self.cac:
            B, C, Fr, T = z.shape
            m = th.view_as_real(z).permute(0, 1, 4, 2, 3)
            m = m.reshape(B, C * 2, Fr, T)
        else:
            m = z.abs()
        return m
    
    
    def forward(self, mix):
        length = mix.shape[-1]
        length_pre_pad = None
        
        if self.use_train_segment:
            if self.training:
                self.segment = Fraction(mix.shape[-1], self.samplerate)
            else:
                training_length = int(self.segment * self.samplerate)
                if mix.shape[-1] < training_length:
                    length_pre_pad = mix.shape[-1]
                    mix = F.pad(mix, (0, training_length - length_pre_pad))
        
        z = self._spectr(mix)
       
        x_z = self._magnitude(z)     # fft_amp
        x_p = th.atan2(z.imag, z.real) # fft_pha
        
        
        mean_z = x_z.mean(dim=(1, 2, 3), keepdim=True)
        std_z = x_z.std(dim=(1, 2, 3), keepdim=True)
        x_z = (x_z - mean_z) / (1e-5 + std_z)
        
        
        mean_p = x_p.mean(dim=(1, 2, 3), keepdim=True)
        std_p = x_p.std(dim=(1, 2, 3), keepdim=True)
        x_p = (x_p - mean_p) / (1e-5 + std_p)
        
        B, C, Fq, T = x_z.shape
        
        saved_z = []  # skip connections, freq.
        saved_p = []  # skip connections, phase.
        lengths_z = []  # saved lengths to properly remove padding, freq branch.
        lengths_p = []  # saved lengths for phase branch.
        
        for idx in range(len(self.encoder_z)):
            
            lengths_z.append(x_z.shape[-1])
            inject = None
            
            encode_z = self.encoder_z[idx]
            
            lengths_p.append(x_p.shape[-1])
            encode_p = self.encoder_p[idx]
            
            x_p = encode_p(x_p)
                
            if not encode_p.empty:
                # save for skip connection
                if idx == 0 and self.phase_emb is not None:
                    frs_p = th.arange(x_p.shape[-2], device=x_p.device)
                    emb_p = self.phase_emb(frs_p).t()[None, :, :, None].expand_as(x_p)
                    x_p = x_p + self.phase_emb_scale * emb_p
              
            else:
                inject = x_p
    
            saved_p.append(x_p) 
    
            x_z = encode_z(x_z, inject)   
            
            if idx == 0 and self.freq_emb is not None:

                frs = th.arange(x_z.shape[-2], device=x_z.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x_z)
                x_z = x_z + self.freq_emb_scale * emb

            saved_z.append(x_z)
            
        if self.crosstransformer:

            if self.bottom_channels:
                b, c, f, t = x_z.shape
                x_z = rearrange(x_z, "b c f t-> b c (f t)")
                x_z = self.channel_upsampler(x_z)
                x_z = rearrange(x_z, "b c (f t)-> b c f t", f=f)

                x_p = rearrange(x_p, "b c f t-> b c (f t)")
                x_p = self.channel_upsampler(x_p)
                x_p = rearrange(x_p, "b c (f t)-> b c f t", f=f)
            x_z, x_p = self.crosstransformer(x_z, x_p)
                    
        for idx in range(len(self.decoder_z)):
            decode_z = self.decoder_z[idx]
            skip_z = saved_z.pop(-1)
            x_z, pre = decode_z(x_z, skip_z, lengths_z.pop(-1))
            
            decode_p = self.decoder_p[idx]
            skip_p = saved_p.pop(-1)
            x_p, pre = decode_p(x_p, skip_p, lengths_p.pop(-1))
           
        assert len(saved_p) == 0
        assert len(lengths_p) == 0
        assert len(saved_z) == 0
        assert len(lengths_z) == 0

        S = len(self.sources)
        
        x_z = x_z.view(B, S, -1, Fq, T)
        x_z = x_z * std_z[:, None] + mean_z[:, None]

        x_p = x_p.view(B, S, -1, Fq, T)
        x_p = x_p * std_p[:, None] + mean_p[:, None]
        
        imag = x_z * th.sin(x_p)
        real = x_z * th.cos(x_p)
        x = th.complex(real, imag)
    
        if self.use_train_segment:
            if self.training:
                x = self._ispectr(x, length)
            else:
                x = self._ispectr(x, training_length)
        else:
            x = self._ispectr(x, length)
        if length_pre_pad:
            x = x[..., :length_pre_pad]
        return x
