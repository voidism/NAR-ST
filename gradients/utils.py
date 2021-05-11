import sys
import pdb
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sentencepiece as spm
from typing import Optional, Callable
from matplotlib.pyplot import cm

from espnet.nets.pytorch_backend.nets_utils import to_device

from espnet.nets.pytorch_backend.transducer.vgg import VGG2L
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling

class SPMModel():
    def __init__(self, model_path, token_list):
        self.model = spm.SentencePieceProcessor()
        self.model.Load(model_path)
        self.token_list = token_list
    
    def encode(self, sample):
        tokens = self.model.SampleEncodeAsPieces(sample, 1, 0.0)
        return [self.token_list.index(t) for t in tokens]
    
#     def detok(self, x):
#         return "".join(x).replace("▁", " ")
    
#     def decode(self, x, detok=True):
#         tokens = [self.token_list[t] for t in x]
#         return self.detok(tokens) if detok else tokens
    def decode(self, x):
        return self.model.DecodeIds(x)
            

def plot_align(
    aln,
    titles: Optional[list]=None,
    x_tokens: Optional[list]=None,
    y_tokens: Optional[list]=None,
    norm_fn: Optional[Callable]=lambda x:x,
    columns: Optional[int]=1,
    tick_size: Optional[int]=10,
    fig_size: Optional[tuple]=(12,8),
    save: Optional[str]=None,
    cmap=cm.Blues,
):
    """Function to plot the alignment with tokens
    """
    
    n_graphs = len(aln)
    rows = 1+(n_graphs // columns)
    xlen, ylen = aln[0].shape
    
    if titles is None:
        titles = [f"layer {i}" for i in range(n_graphs)]
    if x_tokens is None:
        x_tokens = [range(xlen)]*n_graphs
    if y_tokens is None:
        y_tokens = [range(ylen)]*n_graphs    
    
    fig = plt.figure(figsize=(fig_size[0]*columns,fig_size[1]*rows), dpi=100) 
    for i,a in enumerate(aln):
        ax = fig.add_subplot(rows, columns, i+1)
        ax.imshow(norm_fn(a), cmap=cmap)
        ax.set_title(titles[i]) # title
        for tick in ax.get_xticklabels(): # diagonal xtick
            tick.set_rotation(45)
        ax.set_xticks(range(ylen)) 
        ax.set_xticklabels(y_tokens[i%len(y_tokens)], fontsize=tick_size)
        ax.set_yticks(range(xlen)) 
        ax.set_yticklabels(x_tokens[i%len(x_tokens)], fontsize=tick_size)
     
    if save is not None:
        fig.savefig(save)
            
def x_norm(a):
    return a/a.sum(dim=0)
def sharp_norm(a, tau=1.):
    a = a**(1/tau)
    return x_norm(a)

def ctc_collapse(ctc_output, alignment, reduction='max'):
    """ collapse ctc repeat tokens and remove blank"""
    ctc_decode = []
    new_align = []
    raw_ctc_output = ctc_output.tolist()[0]
    for i in range(len(raw_ctc_output)):
        if raw_ctc_output[i] == 0:
            continue
        if i > 0 and raw_ctc_output[i-1] == raw_ctc_output[i]:
            if reduction == 'max':
                prev = new_align[-1]
                mask = prev > alignment[:,i]
                new_align[-1] = mask*prev + (~mask)*alignment[:,i]
            elif reduction in ('mean', 'sum'):
                new_align[-1] += alignment[:,i]
            continue
        ctc_decode.append(raw_ctc_output[i])
        new_align.append(alignment[:,i].clone())
    return ctc_decode, torch.stack(new_align, dim=1)


def find_multilayer_jacobian(
    model, 
    feat, 
    modeltype = "mt", # "mt" "st", "st_seg"
    ctc_output = None,
    tgt_lang = None,
    other_forward_encoder = None,
):
    """ """

    def forward_encoder(self, xs, masks):
        """Encode input sequence.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        extras = []
        if isinstance(self.embed, (Conv2dSubsampling, VGG2L)):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)
        for i,layer in enumerate(self.encoders):
            xs.retain_grad()
            extras.append(xs)
            xs, masks = layer(xs, masks)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks, extras

    
    if modeltype == "st_buf":
        """ st model """
        x = to_device(model, torch.as_tensor(feat)).unsqueeze(0)
        enc_output, _, layer_ins = other_forward_encoder(model.encoder, x, None)

        h = enc_output
    elif modeltype == "st_seg":
        """ st model """
        x = to_device(model, torch.as_tensor(feat)).unsqueeze(0)
        enc_output0, _, layer_ins0 = forward_encoder(model.encoder_src, x, None)
        enc_output1, _, layer_ins1 = forward_encoder(model.encoder_tgt, enc_output0, None)

        layer_ins = layer_ins0 + layer_ins1

        h = enc_output1

        ## asr 
        asr_output = model.ctc_src.argmax(enc_output0)

    elif modeltype == "st":
        """ st model """
        x = to_device(model, torch.as_tensor(feat)).unsqueeze(0)
        enc_output, _, layer_ins = forward_encoder(model.encoder, x, None)

        h = enc_output
    else:
        """ mt model """

        xs = to_device(model, torch.from_numpy(np.asarray(feat)))
        xs_pad = xs.unsqueeze(0)    
        xs_pad, _ = model.target_forcing(xs_pad, tgt_lang=tgt_lang)
        enc_output, _, layer_ins = forward_encoder(model.encoder, xs_pad, None)
    
        h = enc_output
        batch_size = xs_pad.size(0)
        h = h.view(batch_size, -1, model.adim//2)
        h = model.to_ctc_linear(h)

    if ctc_output is None:
        ctc_output = model.ctc.argmax(h)

    # we use logits instead of logp to mitigate influence from other classes
    logits = model.ctc.ctc_lo(h) # model.ctc.log_softmax(h)
    losses = F.nll_loss(
        logits.view(-1, logits.size(-1)), 
        ctc_output.view(-1), 
        reduction='none'
    ) # technically this is negative logits
    
    nlayer_jacobian = [] # (layer,xlen,ylen)
    for token_loss in losses:
        # clean grads
        model.zero_grad()
        for p in layer_ins:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        token_loss.backward(retain_graph=True)
        jacobian = [] # (layer,xlen)
        for hidden in layer_ins:
            input_grads = hidden.grad.norm(dim=-1) # (1,7,256) -> (1,7)
            jacobian.append(input_grads) # (1,xlen)
        nlayer_jacobian.append(torch.cat(jacobian, dim=0))

    outputs = (torch.stack(nlayer_jacobian, dim=-1), ctc_output)
    if modeltype == "st_seg":
        outputs = outputs + (asr_output,)
    return outputs

def plot_transition(
    aln,
    x_tokens: Optional[list]=None,
    y_tokens: Optional[list]=None,
    norm_fn: Optional[Callable]=lambda x:x,
    columns: Optional[int]=1,
    tick_size: Optional[int]=10,
    fig_size: Optional[tuple]=(12,8),
    mask: Optional[list]=None,
    ds: Optional[int]=1, # downsample for rightside label (y)
    yshift: Optional[int]=0,
    save: Optional[str]=None,    
    cmap=cm.Blues
):
    """Function to plot the transition of with tokens
    """
    n_layers, xlen, ylen = aln.shape
    transition = aln.transpose(0, 2)
    
    n_graphs = ylen
    rows = 1+(n_graphs // columns)
    
    if x_tokens is None:
        x_tokens = [range(xlen)]*n_graphs
    if y_tokens is None:
        y_tokens = [range(ylen)]*n_graphs
            
    xlen, ylen = transition[0].shape
    fig = plt.figure(figsize=(fig_size[0]*columns,fig_size[1]*rows), dpi=100) 
    
    p = 0
    for i,a in enumerate(transition):
        if mask is not None and not mask[i]:
            continue
        p = p+1        
        ax = fig.add_subplot(rows, columns, p)
        ax.imshow(norm_fn(a), aspect='auto', cmap=cmap)
        ax.set_title(f"token: {y_tokens[0][i]}({i})") # title
        for tick in ax.get_xticklabels(): # diagonal xtick
            tick.set_rotation(45)
        
        ax.set_xlabel('layers')
        ax.set_xticks(range(n_layers)) 
#         ax.set_xticklabels(y_tokens[i%len(y_tokens)], fontsize=tick_size)
        ax.set_yticks(range(xlen)) 
        ax.set_yticklabels(x_tokens[i%len(x_tokens)], fontsize=tick_size)
        
        ax_r = ax.twinx()
        ax_r.imshow(norm_fn(a), aspect='auto', cmap=cmap)
        ax_r.set_yticks(range(xlen)) 
        right_labels = [ "" for j in range(xlen)]
        right_labels[(yshift+i)//ds] = y_tokens[i%len(y_tokens)][i]
        ax_r.set_yticklabels(right_labels, fontsize=tick_size)
    
    fig.tight_layout()
    if save is not None:
        fig.savefig(save)