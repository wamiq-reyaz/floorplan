import warnings
warnings.filterwarnings('ignore', '', FutureWarning)
import torch
from torch import Tensor, device, dtype, nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from transformers.activations import  ACT2FN
from transformers.modeling_utils import (
    Conv1D,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer)

class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super().__init__()

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        if config.is_causal:
            self.register_buffer(
                "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
            )
        else:
            self.register_buffer(
                "bias", torch.ones((n_ctx, n_ctx), dtype=torch.uint8).view(1, 1, n_ctx, n_ctx)
            )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_head, self.split_size // self.n_head, self.pruned_heads
        )
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / (float(v.size(-1)) ** 0.5)
        nd, ns = w.size(-2), w.size(-1)
        mask = self.bias[:, :, ns - nd : ns, :ns]
        w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(
        self, x, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False
    ):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        else:
            present = (None,)

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super().__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(
        self, x, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False,
    ):
        output_attn = self.attn(
            self.ln_1(x),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        a = output_attn[0]  # output_attn: a, present, (attentions)

        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m

        outputs = [x] + output_attn[1:]
        return outputs  # x, present, (attentions)

class CrossAttnBlock(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super().__init__()
        nx = config.n_embd
        self.config = config

        self.cross_attn = nn.TransformerDecoderLayer(d_model=nx,
                                                     nhead=config.n_head,
                                                     dim_feedforward=4*nx,
                                                     activation='gelu'
                                                     )

    def forward(
        self, enc, dec, self_attn_mask=None, cross_attn_mask=None
    ):
        """
        :param enc: The encoder embeddings of shape (S, N, E) 
        :param dec: The decoder embeddings of shape (T, N, E)
        :param self_attn_mask: The decoder self attention mask of shape (N x T) for excluding padding
        :param cross_attn_mask: The decoder cross attention mask of shape (N, S) for excluding padding in encoder
        :return: output (output, 0): the attention+MLP output
        """
        output = self.cross_attn(tgt=dec, # T, N, E
                                 memory=enc, # S, N, E
                                 tgt_mask=self._get_decoder_mask(dec.shape[0]).to(dec.device), # TxT
                                 tgt_key_padding_mask=self._convert_mask_to_bool(self_attn_mask), # N x T
                                 memory_key_padding_mask=self._convert_mask_to_bool(cross_attn_mask) # N x S
                                 )

        return (output, torch.tensor(0, device=output.device))


    def _get_decoder_mask(self, nt):
        # nt = self.config.n_ctx
        if self.config.is_causal:
            mask = torch.tril(torch.ones(nt, nt, dtype=torch.uint8))
        else:
            mask = torch.ones(nt, nt, dtype=torch.uint8)
        return mask.logical_not()

    def _convert_mask_to_bool(self, mask: torch.Tensor):
        """
        :param mask: A torch array of shape (N, X). This is an arbitrary attention mask, which should have 1s in
                places we want to attend to, and 0's in places we don't. As PyTorch uses a different scheme, we invert.
        :return:
        """
        mask_ = mask.type(torch.uint8)
        return torch.logical_not(mask_)


class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size+2, config.n_embd)
        self.wtte = nn.Embedding(config.n_types, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size+2)

        self.init_weights()

    def init_weights(self):
        """ Initialize and prunes weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.
            If `past` is used only the last hidden-state of the sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True``) is passed or when ``config.output_hidden_states=True``:
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=inputs_embeds.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = head_mask = [None] * self.config.n_layer #self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0

        #type_embeds
        num_ids = input_ids.shape[-1]
        type_ids = torch.arange(num_ids, dtype=torch.long, device=inputs_embeds.device) % self.config.n_types
        type_ids = type_ids.repeat((batch_size, 1))
        type_embeds = self.wtte(type_ids)

        # print(inputs_embeds.shape, position_embeds.shape, type_embeds.shape)
        hidden_states = inputs_embeds + position_embeds + token_type_embeds + type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

            if output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if use_cache is True:
            outputs = outputs + (presents,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)

        if self.config.is_encoder:
            return (hidden_states,)

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs


        return outputs  # last hidden state, (presents), (all hidden_states), (attentions)


class GPT2Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size+2, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.n_embd)

        try:
            self.id_embed = self.config.id_embed
        except:
            self.id_embed = False

        try:
            self.pos_id = self.config.use_pos_emb
        except:
            self.pos_id = False

        try:
            self.passthrough = self.config.passthrough
        except:
            self.passthrough = False

        if self.id_embed:
            self.ide = nn.Embedding(config.vocab_size+2, config.n_embd)

        if self.pos_id:
            self.pde = nn.Embedding(config.n_ctx, config.n_embd)
            self.wtte = nn.Embedding(config.n_types, config.n_embd)

        self.init_weights()

    def init_weights(self):
        """ Initialize and prunes weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.
            If `past` is used only the last hidden-state of the sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True``) is passed or when ``config.output_hidden_states=True``:
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        elif input_ids is not None:
            input_shape = input_ids.size()[:-1]
            # input_shape = torch.Size)
            # input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)

        # print(next(self.parameters()).device)


        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=input_ids.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = head_mask = [None] * self.config.n_layer #self.get_head_mask(head_mask, self.config.n_layer)

        # print('Before embeddings')
        if inputs_embeds is None:
            if self.id_embed:
                # print(input_ids.shape)
                # print(input_ids[:, :, 0].shape)

                id_embeds = self.ide(input_ids[:, :, 0].contiguous())
                # print(id_embeds)
                inputs_embeds = self.wte(input_ids[:, :, 1:].contiguous())

                # if input_ids.dim() == 3:
                #     id_
                # print(input_ids.shape)
                # print(id_embeds.shape)
                # print(inputs_embeds.shape)
            else:
                id_embeds = 0
                inputs_embeds = self.wte(input_ids)

            # print('/INside encoder printing dim of inputs_embeds', inputs_embeds.dim())
            if inputs_embeds.dim() == 4: # ie the verts are 2d
                inputs_embeds = torch.sum(inputs_embeds, dim=-2) + id_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0

        if self.pos_id:
            if position_ids is None:
                n_ctx = inputs_embeds.shape[1]
                bs = inputs_embeds.shape[0]
                position_ids = torch.arange(0, n_ctx, dtype=torch.long, device=inputs_embeds.device).reshape(1, -1).repeat(bs, 1)
                type_ids = torch.arange(n_ctx, dtype=torch.long, device=inputs_embeds.device).reshape(1, -1).repeat(bs, 1) % self.config.n_types
            # print(pos_tokens.shape)
            pos_embeds = self.pde(position_ids)
            type_embeds = self.wtte(type_ids)
            inputs_embeds = inputs_embeds + pos_embeds + type_embeds
            print(inputs_embeds.shape)
            print('hello')


        hidden_states = inputs_embeds
        hidden_states = self.drop(hidden_states)

        if self.passthrough:
            return hidden_states

        # print('inside encoder printing hidden_states.shape', hidden_states.shape)
        output_shape = input_shape + (hidden_states.size(-1),)
        # print('inside encoder printing output shape', output_shape)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

            if output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)
        hidden_states = self.lm_head(hidden_states)

        # hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        # print('for outputs', outputs[0].shape)
        if use_cache is True:
            outputs = outputs + (presents,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)

        # print('Last in encoder shape', hidden_states.shape)
        return (hidden_states,)

class GPT2Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.n_embd)

        self.init_weights()

    def init_weights(self):
        """ Initialize and prunes weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.
            If `past` is used only the last hidden-state of the sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True``) is passed or when ``config.output_hidden_states=True``:
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache




        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=inputs_embeds.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = head_mask = [None] * self.config.n_layer #self.get_head_mask(head_mask, self.config.n_layer)

        position_embeds = self.wpe(position_ids)
        # print('In decoder,', position_embeds.shape)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        # print('in decoder, first hidden state', hidden_states.shape)


        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

            if output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)
        hidden_states = self.lm_head(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if use_cache is True:
            outputs = outputs + (presents,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)

        # print('Last in decoder, ', hidden_states.shape)
        return (hidden_states, )


class GPT2ConditionalDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size+2, config.n_embd)
        self.wtte = nn.Embedding(config.n_types, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([CrossAttnBlock(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size+2)

        self.init_weights()

    def init_weights(self):
        """ Initialize and prunes weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(
        self,
        input_ids=None, # the target sequence
        past=None,
        self_attention_mask=None,
        cross_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        enc_seq=None, # the encoder sequence
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache


        input_shape = input_ids.size()
        batch_size = input_ids.shape[0]

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            # print(position_ids)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        if token_type_ids is None:
            num_ids = input_ids.shape[-1]
            type_ids = torch.arange(num_ids, dtype=torch.long, device=input_ids.device) % self.config.n_types
            type_ids = type_ids.repeat((batch_size, 1))


        # Get the input and position embeds
        # NxT -> N x T x dim
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        type_embeds = self.wtte(type_ids)

        hidden_states = inputs_embeds + position_embeds + type_embeds
        hidden_states = self.drop(hidden_states)

        # print(enc_seq)
        # (N , T) + (dim)
        output_shape = input_shape + (hidden_states.size(-1),)

        enc_seq_transposed = enc_seq.transpose(0, 1) # S, N, E
        hidden_states = hidden_states.transpose(0, 1) # T, N, E
        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(enc=enc_seq_transposed,
                            dec=hidden_states,
                            self_attn_mask=self_attention_mask,
                            cross_attn_mask=cross_attention_mask)
            # print(outputs[0])
            # print(type(outputs))
            # sys.exit()

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

            if output_attentions:
                all_attentions.append(outputs[2])

        # untranspose
        hidden_states = hidden_states.transpose(0, 1) # N, T, E
        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if use_cache is True:
            outputs = outputs + (presents,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)

        hidden_states = self.lm_head(hidden_states)

        return (hidden_states, )

class GraphGPTModel(nn.Module):
    def __init__(self, enc_config, dec_config):
        super().__init__()
        self.encoder = GPT2Encoder(enc_config)
        self.decoder = GPT2Decoder(dec_config)
        self.embed_dim = enc_config.n_embd


    def forward(self,
                node,
                edg,
                labels,
                attention_mask=None,
                vert_attn_mask=None):
        # embed the nodes first
        node_embed = self.encoder(input_ids=node, attention_mask=vert_attn_mask)[0] # bs, len, embd
        node_embed *= vert_attn_mask[..., None]
        # print(edg.shape)
        edg = edg.unsqueeze(-1)
        indexer = edg.repeat((1, 1, self.embed_dim)).long() # bs, edg_len, embd
        # print('In graph, node, indexer', node_embed.shape, indexer.shape)
        # print('Unique in indexer', torch.unique(indexer))

        edg_embed = torch.gather(node_embed, 1, indexer)

        # print('In graph, edge_embeddings', edg_embed.shape)
        pointers = self.decoder(inputs_embeds=edg_embed, attention_mask=attention_mask)[0] # bs, edg_len, embd

        inner_prod = torch.matmul(pointers, node_embed.permute(0, 2, 1))

        # node_embed = node_embed.permute(1, 0, 2) # len, bs, embd
        # pointers = pointers.permute(1, 0, 2) # edg_len, bs, embd
        # pointers = pointers.unsqueeze(0) # 1, edg_len, bs, embd
        # # print(pointers.shape)
        # pointers = pointers.permute(1, 0, 2, 3) # edg_len, 1, bs, embd
        #
        # # print(pointers.shape, node_embed.shape)
        # inner_prod = pointers * node_embed
        # inner_prod = torch.sum(inner_prod, dim=-1) #edg_len, len, bs
        # inner_prod = inner_prod.permute(2, 0, 1) # bs, edg_len, len
        #
        # # print(inner_prod.shape)
        if labels is not None:
            logits = inner_prod[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()

            loss_fct = CrossEntropyLoss()
            # print(logits.size())
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        else:
            loss = 0

        # print(inner_prod.shape)
        return loss, inner_prod


class EncDecGPTModel(nn.Module):
    def __init__(self, enc_config, dec_config):
        super().__init__()
        self.encoder = GPT2Encoder(enc_config)
        self.decoder = GPT2ConditionalDecoder(dec_config)
        self.embed_dim = enc_config.n_embd


    def forward(self,
                enc_seq,
                dec_seq,
                enc_attn_mask=None,
                dec_attn_mask=None,
                labels=None):

        enc_embed = self.encoder(input_ids=enc_seq,
                                 attention_mask=enc_attn_mask)[0]

        # print(enc_embed)
        logits = self.decoder(input_ids=dec_seq,
                              self_attention_mask=dec_attn_mask,
                              cross_attention_mask=enc_attn_mask,
                              enc_seq=enc_embed)[0]

        outputs = (logits,)
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = outputs + (loss,) # tuple concat

        return outputs


if __name__ == '__main__':

    from transformers.configuration_gpt2 import  GPT2Config

    config = GPT2Config(
        vocab_size=65,
        n_positions=200,
        n_ctx=200,
        n_embd=264,
        n_layer=12,
        n_head=12,
        is_causal=False,
        is_encoder=False,
        n_types=5,
        pos_id=True
    )

    dec_config = GPT2Config(
        vocab_size=65,
        n_positions=100,
        n_ctx=100,
        n_embd=264,
        n_layer=12,
        n_head=12,
        is_causal=True,
        is_encoder=False,
        n_types=5,
        dec_n=100
    )

    model = EncDecGPTModel(config, dec_config)

    bs = 3
    node_ids = torch.arange(200, dtype=torch.long) % 63
    node_ids = node_ids.repeat((bs, 1))

    node_ids2 = torch.arange(100, dtype=torch.long) % 63
    node_ids2 = node_ids2.repeat((bs, 1))

    cross_attn_mask = torch.zeros((3, 200))
    cross_attn_mask[:, :100] = 1
    self_attn_mask = torch.zeros((3, 100))
    self_attn_mask[:, :150] = 1

    # print(node_ids.shape)
    ret_tuple = model(enc_seq=node_ids,
                   dec_seq=node_ids2,
                   enc_attn_mask=cross_attn_mask,
                   dec_attn_mask=self_attn_mask,
                   labels=node_ids2)

    # print(ret_tuple[0].shape, ret_tuple[1])
    # print(ret_tuple[0])
    # def forward(self,
    #             enc_seq,
    #             dec_seq,
    #             enc_attn_mask=None,
    #             dec_attn_mask=None,
    #             labels=None):

    # cond_dec_config = GPT2Config(is_causal=True,
    #                              n_types=5,
    #                              dec_n=1024
    #                                 )
    # bs = 3
    # node_ids = torch.arange(1024, dtype=torch.long)
    # node_ids = node_ids.repeat((bs, 1))
    #
    # cross_attn_mask = torch.ones((3, 1024))
    # self_attn_mask = torch.zeros((3, 1024))
    # embeddings = torch.rand(3, 1024, 768)
    #
    # model = GPT2ConditionalDecoder(cond_dec_config)
    #
    # logits = model(input_ids=node_ids,
    #                self_attention_mask=self_attn_mask,
    #                cross_attention_mask=cross_attn_mask,
    #                enc_seq=embeddings)
    #
    # print(logits[0].shape)


    # enc = GPT2Config(is_causal=False)
    # dec = GPT2Config(is_causal=True)

    # model = GraphGPTModel(enc, dec)
    #
    # bs = 3
    # node_ids = torch.arange(1024, dtype=torch.long)
    # node_ids = node_ids.repeat((bs, 1))
    #
    # edg_ids = torch.arange(512, dtype=torch.long)
    # edg_ids = edg_ids.repeat((bs, 1))
    #
    # attention_mask = torch.ones(512, dtype=torch.float).reshape((1, -1))
    # attention_mask = attention_mask.repeat((bs, 1))
    #
    # loss, logits = model(node_ids, edg_ids, edg_ids, attention_mask)

    # config = GPT2Config(is_causal=False)
    # # print(config)
    # # print(config.is_causal)
    #
    # model = GPT2Encoder(config)
    #
    # bs = 3
    # input_ids = torch.arange(1024, dtype=torch.long)
    # input_ids = input_ids.repeat((bs, 1))
    #
    # position_ids = position_ids = torch.arange(1024, dtype=torch.long)
    # position_ids = position_ids.repeat((bs, 1))
    #
    # attention_mask = torch.ones(1024, dtype=torch.float).reshape((1, -1))
    # attention_mask = attention_mask.repeat((bs, 1))
    # print(attention_mask.shape)
    #
    # output = model(input_ids=input_ids,
    #                position_ids=position_ids,
    #                attention_mask=attention_mask)
    #
    # print(output[0].shape)
