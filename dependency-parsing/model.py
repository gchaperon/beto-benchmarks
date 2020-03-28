import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

# WIP

POS_EMBED_SIZE = 100

N_MLP_ARC = 500
N_MLP_REL = 100


class MLP(nn.Module):
    def __init__(self, config, output_size):
        super(MLP, self).__init__()

        self.linear = nn.Linear(
            config.hidden_size + POS_EMBED_SIZE, output_size
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x


class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(
            torch.Tensor(n_out, n_in + bias_x, n_in + bias_y)
        )
        self.reset_parameters()

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum("bxi,oij,byj->boxy", x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)

        return s


class BertForDependencyParsing(BertPreTrainedModel):
    def __init__(self, config, *, n_pos_tags, n_rels):
        super().__init__(config)
        # breakpoint()
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        # TODO: tomarse en serio esta parte
        # turbo jarcodeaos, son 17 POS tags distintos y 768 es el hidden_size,
        # pa poder concatenarlos con la salida de bert
        self.pos_embeds = nn.Embedding(n_pos_tags, POS_EMBED_SIZE)

        self.mlp_arc_h = MLP(config, N_MLP_ARC)
        self.mlp_arc_d = MLP(config, N_MLP_ARC)
        self.mlp_rel_h = MLP(config, N_MLP_REL)
        self.mlp_rel_d = MLP(config, N_MLP_REL)

        self.arc_attn = Biaffine(n_in=N_MLP_ARC, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(
            n_in=N_MLP_REL, n_out=n_rels, bias_x=True, bias_y=True
        )

    def forward(self, *, pos_tags_ids, prediction_mask, **kwargs):
        # concat salida con pos tags
        output = self.bert(**kwargs)[0]
        pos_embeds = self.pos_embeds(pos_tags_ids)
        output = torch.cat((output, pos_embeds), dim=2)
        # apply MLPs to the BiLSTM output states
        arc_h = self.mlp_arc_h(output)
        arc_d = self.mlp_arc_d(output)
        rel_h = self.mlp_rel_h(output)
        rel_d = self.mlp_rel_d(output)

        s_arc = self.arc_attn(arc_d, arc_h)
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)

        # -inf para las palabras que no pueden ser cabeza
        s_arc.masked_fill_(~prediction_mask.unsqueeze(1).bool(), float("-inf"))
        # breakpoint()
        return s_arc, s_rel
