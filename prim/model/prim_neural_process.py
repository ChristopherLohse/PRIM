"""
PRIM with a single concatenated obs+int sample sequence]
+ explicit (int - obs) differencing with an attentive, neural-process-style
readout (Sec. 3.2.1 of Ke et al., 2022, "Learning to Induce Causal Structure").

Design differences from prim.PRIM / MACETransformerLayer:
  - Observational and interventional samples are concatenated into one
    sequence per node instead of being kept in two groups joined only by a
    one-directional (int -> obs) cross-attention. Every sample can now attend
    to every other sample, in both directions, at every layer.
  - The observational/interventional type and the optional anomaly
    (target-node) indicator are encoded as embeddings tied to a token's
    *content* (its type, its node id) rather than to its position in the
    sample sequence, so they are unaffected by the permutation invariance/
    equivariance of the attention layers over the sample axis. Unlike a
    typical additive positional embedding, they are CONCATENATED into their
    own dedicated, non-overlapping channels of the embedding vector rather
    than summed into the same channels as the value/node content - matching
    how CSIvA Sec. 3.2.1 itself builds its per-token embedding ("half the
    vector for value, half for node identity" is a concatenation, not a sum).
    This guarantees the tag is never diluted or overwritten by whatever the
    rest of the network happens to compute in that subspace.
  - The mask indicating the observed anomaly (target_mask) is folded into the
    positional/content encoding shared by both encoders instead of being
    added only to the interventional branch after encoding; it remains
    optional (an explicit "mask not provided" category is used when none is
    given, since a concatenated slice can't simply be left empty).
  - The type and anomaly channels are re-written (not added-to) before every
    layer (not just once at the input) so the obs/int identity of a token
    stays exactly the known tag at every layer instead of relying on it
    surviving many rounds of attention, LayerNorm and MLP through the
    residual stream. The value/node content channels are left alone between
    layers and evolve normally, carrying the residual stream.
  - Mean pooling over samples, followed by an explicit int_pooled - obs_pooled
    difference, is replaced by an attentive neural-process-style readout
    (QueryPooling). A learned per-node query is concatenated onto the sample
    axis as an extra token *before the first layer*, so it participates in
    every node-level and sample-level attention alongside the real obs/int
    samples and evolves together with them - exactly like the "column S+1"
    summary token in CSIvA Sec. 3.2.1. Only at the very end is it split back
    off and used as the query in one final cross-attention against the
    evolved real samples, mapping the dataset D directly to a single summary
    embedding per node. Because self-attention with no absolute positional
    encoding is permutation-equivariant, this stays exactly invariant to the
    order of the real samples throughout.
"""

import math

import torch
from torch import nn
from torch.nn.modules.transformer import MultiheadAttention, Linear, LayerNorm


class Decoder(nn.Module):

    def __init__(
        self,
        embedding_size: int,
        mlp_hidden_size: int,
        num_outputs: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            Linear(embedding_size, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(mlp_hidden_size, mlp_hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(mlp_hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, max_nodes, embedding_size)
        Returns:
            logits: (batch_size, max_nodes)
        """
        x = self.mlp(x)  # (batch_size, max_nodes, 1)
        return x.squeeze(-1)  # (batch_size, max_nodes)


def sinusoidal_positional_embedding(num_positions: int,
                                    dim: int) -> torch.Tensor:
    """
    Fixed, non-learned sinusoidal positional embedding (Vaswani et al., 2017):
    PE(pos, 2i) = sin(pos / 10000^(2i/dim)), PE(pos, 2i+1) = cos(pos / 10000^(2i/dim)).
    Returns a (num_positions, dim) tensor with no gradient and no learned
    parameters, used for node identity - unlike a typical additive positional
    embedding, this only supplies the CONTENT for its own dedicated,
    concatenated node_dim channels (see UnifiedDataEncoder).
    """
    position = torch.arange(num_positions, dtype=torch.float32).unsqueeze(
        1)  # (num_positions, 1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) *
        (-math.log(10000.0) / dim))
    pe = torch.zeros(num_positions, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
    return pe


class PRIMNeuralProcess(nn.Module):
    """Drop-in replacement for PRIM with the same constructor/forward signature."""

    def __init__(
        self,
        embedding_size: int,
        num_attention_heads: int,
        mlp_hidden_size: int,
        num_layers: int,
        max_nodes: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.max_nodes = max_nodes

        self.encoder = UnifiedDataEncoder(embedding_size,
                                          max_nodes,
                                          dropout=dropout)

        self.layers = nn.ModuleList([
            AlternatingAttentionLayer(
                embedding_size,
                num_attention_heads,
                mlp_hidden_size,
                sample_level=(i % 2 == 1),
                dropout=dropout,
            ) for i in range(num_layers)
        ])

        # Pre-norm blocks leave the residual stream unnormalized between
        # layers, so - as is standard for pre-norm transformers - apply one
        # final LayerNorm to the whole stream before reading out of it.
        self.final_norm = LayerNorm(embedding_size)

        self.readout = QueryPooling(
            embedding_size,
            self.encoder.content_dim,
            max_nodes,
            num_attention_heads,
            dropout=dropout,
        )
        self.decoder = Decoder(embedding_size,
                               mlp_hidden_size,
                               max_nodes,
                               dropout=dropout)

    def half(self):
        """
        Casts the model to float16 for speed/memory, EXCEPT the attention
        sublayers (every AlternatingAttentionLayer.attn and
        QueryPooling.attn), which are pinned to float32: softmax over scaled
        dot products is prone to overflow/underflow in fp16, so those
        submodules' own weights stay fp32-resident while everything else
        (value/node/type/anomaly embeddings, MLPs, LayerNorms, the decoder)
        runs in fp16. forward() upcasts activations to fp32 right at the
        boundary of each fp32-pinned attention call and casts the result
        back down, so no dtype mismatch occurs and no per-call weight
        toggling is needed (which would otherwise risk confusing optimizer
        state, e.g. Adam's fp16-shaped moment buffers).
        """
        super().half()
        for layer in self.layers:
            layer.attn.float()
        self.readout.attn.float()
        return self

    def forward(self, obs_data, int_data, target_mask=None, n_nodes=None):
        """
        Args:
            obs_data: (batch_size, n_obs, max_nodes)
            int_data: (batch_size, n_int, max_nodes)
            target_mask: (batch_size, max_nodes) - optional binary mask, 1 for
                the target/symptom node whose root cause is being queried.
            n_nodes: int - number of real nodes; positions >= n_nodes are
                padding. If None, inferred from data (all-zero columns).

        Returns:
            logits: (batch_size, max_nodes)
        """
        batch_size, n_obs, k = obs_data.shape
        n_int = int_data.shape[1]
        device = obs_data.device

        # Value + node-identity CONTENT channels only (concatenated, not
        # summed - see UnifiedDataEncoder). Type/anomaly channels are kept
        # separate so they can be attached fresh at every layer instead of
        # being fused in once.
        obs_content = self.encoder.encode_value_node(obs_data, sample_type=0)
        int_content = self.encoder.encode_value_node(int_data, sample_type=1)
        samples_content = torch.cat(
            [obs_content, int_content],
            dim=1)  # (batch, n_obs+n_int, max_nodes, content_dim)

        # Concatenate the readout's learned per-node query CONTENT as one
        # extra "sample" token, *before* the first layer, so it takes part in
        # every node-level and sample-level attention op and evolves
        # alongside the real data instead of only being introduced after the
        # stack.
        query_content = self.readout.query.expand(
            batch_size, -1,
            -1).unsqueeze(1)  # (batch, 1, max_nodes, content_dim)
        content = torch.cat(
            [samples_content, query_content],
            dim=1)  # (batch, n_obs+n_int+1, max_nodes, content_dim)
        n_real = n_obs + n_int  # index where the query token starts along dim 1
        n_total = n_real + 1

        # type_slice carries a third id for the query token so every layer can
        # tell it apart from real obs/int samples; anomaly_slice is per-node
        # and gets broadcast identically over every sample slot (including
        # the query), letting the query start "aware" of which node is under
        # query too.
        type_slice = self.encoder.type_slice(
            n_obs, n_int, device,
            include_query=True)  # (1, n_total, 1, type_dim)
        anomaly_slice = self.encoder.anomaly_slice(
            target_mask, batch_size, k,
            device)  # (batch, 1, max_nodes, anomaly_dim)

        if n_nodes is not None:
            padding_mask = torch.arange(k,
                                        device=device).unsqueeze(0) >= n_nodes
            padding_mask = padding_mask.expand(batch_size,
                                               -1)  # (batch, max_nodes)
        else:
            padding_mask = (obs_data == 0).all(dim=1)  # (batch, max_nodes)

        # NOT hardcoded to .float(): if the model is running in fp16
        # (PRIMNeuralProcess.half()), a hardcoded float32 mask here would
        # silently upcast every `t * node_mask` product back to float32 via
        # PyTorch's type-promotion rules, defeating the point.
        node_mask_bool = ~padding_mask  # (batch, max_nodes)

        def attach_tags(c):
            # c: (batch, n_total, max_nodes, content_dim) - CONCATENATES the
            # type/anomaly channels onto the content, overwriting whatever
            # was there before rather than adding to it.
            type_part = type_slice.expand(
                batch_size, -1, k, -1)  # (batch, n_total, max_nodes, type_dim)
            anomaly_part = anomaly_slice.expand(
                -1, n_total, -1,
                -1)  # (batch, n_total, max_nodes, anomaly_dim)
            t = torch.cat(
                [c, type_part, anomaly_part],
                dim=-1)  # (batch, n_total, max_nodes, embedding_size)
            node_mask = (node_mask_bool.unsqueeze(1).unsqueeze(-1).to(t.dtype)
                         )  # (batch, 1, max_nodes, 1)
            return t * node_mask

        # Dropout only at the input embedding stage; each layer already
        # applies its own dropout internally, so re-attaching the tags at
        # every layer should stay noise-free.
        x = self.encoder.dropout(attach_tags(content))

        content_dim = self.encoder.content_dim
        for layer in self.layers:
            x = layer(x, padding_mask)
            # Keep whatever the layer computed for the content channels, but
            # re-write (not add to) the type/anomaly channels so they always
            # hold exactly the known tag.
            x = attach_tags(x[..., :content_dim])

        x = self.final_norm(x)
        x = x * node_mask_bool.unsqueeze(1).unsqueeze(-1).to(
            x.dtype
        )  # normalize the pre-norm residual stream before reading out

        # Split the evolved query token back off and use it as the query in
        # one final cross-attention against the evolved real samples, mapping
        # the dataset D directly to a single summary embedding per node -
        # replacing mean pooling + explicit int-minus-obs differencing.
        samples_out = x[:, :n_real]  # (batch, n_real, max_nodes, emb)
        query_out = x[:,
                      n_real]  # (batch, max_nodes, emb) - integer index drops the sample dim
        node_summary = self.readout(samples_out,
                                    query_out)  # (batch, max_nodes, emb)
        logits = self.decoder(node_summary)
        return logits


class UnifiedDataEncoder(nn.Module):
    """
    Encodes a batch of samples (observational or interventional) into
    embeddings. The embedding vector is split into four dedicated,
    non-overlapping, CONCATENATED slices - never summed together - so no
    signal can dilute or overwrite another:
      - value_dim:   the sample's value, via a linear layer specific to obs
                      or int (they're different distributions, so distinct
                      weights, not just a shared layer plus an additive tag).
      - node_dim:    node identity, via a FIXED sinusoidal positional
                      embedding over node index (Vaswani et al., 2017) rather
                      than a learned table - node index is an arbitrary,
                      randomly-permuted label with no intrinsic distance
                      semantics for the model to exploit, and this also lets
                      the encoding generalize beyond the configured
                      max_nodes without retraining.
      - type_dim:    observational (0) / interventional (1) / query-token (2).
      - anomaly_dim: non-target (0) / target node (1) / mask-not-provided (2).
    value_dim + node_dim matches CSIvA Sec. 3.2.1's own value/node split
    (half the vector each, out of the budget left after reserving type_dim
    and anomaly_dim). type_dim and anomaly_dim are small, fixed slices
    (embedding_size // 8 each) that PRIMNeuralProcess.forward re-writes
    (never adds to) after every layer, so those channels always hold exactly
    the known tag - not a mixture with whatever the network computed there.
    """

    def __init__(self,
                 embedding_size: int,
                 max_nodes: int,
                 dropout: float = 0.1):
        super().__init__()
        self.type_dim = max(embedding_size // 8, 1)
        self.anomaly_dim = max(embedding_size // 8, 1)
        content_dim = embedding_size - self.type_dim - self.anomaly_dim
        self.value_dim = content_dim // 2
        self.node_dim = content_dim - self.value_dim
        self.content_dim = content_dim

        self.obs_value_embedding = nn.Linear(1, self.value_dim)
        self.int_value_embedding = nn.Linear(1, self.value_dim)
        # Fixed (non-learned) sinusoidal table, sized to the configured
        # max_nodes; encode_value_node slices it to the actual runtime node
        # count. persistent=False: it's cheap and deterministic to recompute,
        # no need to bloat checkpoints with it.
        self.register_buffer(
            "node_position_embedding",
            sinusoidal_positional_embedding(max_nodes, self.node_dim),
            persistent=False,
        )
        self.type_embedding = nn.Embedding(
            3,
            self.type_dim)  # 0=observational, 1=interventional, 2=query token
        self.anomaly_embedding = nn.Embedding(
            3, self.anomaly_dim)  # 0=non-target, 1=target, 2=mask not provided
        self.dropout = nn.Dropout(dropout)

    def encode_value_node(self, x: torch.Tensor,
                          sample_type: int) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_samples, max_nodes)
            sample_type: 0 for observational, 1 for interventional - selects
                which linear value-embedding layer to use.
        Returns:
            (batch_size, n_samples, max_nodes, content_dim) - value
            concatenated with node identity (no type/anomaly channels yet).
        """
        batch_size, n_samples, max_nodes = x.shape
        device = x.device

        value_layer = (self.obs_value_embedding
                       if sample_type == 0 else self.int_value_embedding)
        value_emb = value_layer(
            x.unsqueeze(-1))  # (batch, n_samples, max_nodes, value_dim)

        # Slice the precomputed sinusoidal table to the actual runtime node
        # count (<= the configured max_nodes it was built for).
        node_emb = (self.node_position_embedding[:max_nodes].to(device).view(
            1, 1, max_nodes, -1))
        node_emb = node_emb.expand(
            batch_size, n_samples, -1,
            -1)  # (batch, n_samples, max_nodes, node_dim)

        return torch.cat([value_emb, node_emb],
                         dim=-1)  # (batch, n_samples, max_nodes, content_dim)

    def type_slice(self,
                   n_obs: int,
                   n_int: int,
                   device,
                   include_query: bool = False) -> torch.Tensor:
        """
        Per-slot type ids for a concatenated [obs; int; query?] sample
        sequence, meant to be CONCATENATED (not added) onto the content
        channels at every layer, so every token's type stays exactly legible
        to the attention layers no matter how deep the stack is.

        Args:
            include_query: if True, appends one more id (2 = query token) for
                the readout query token concatenated onto the sample axis.

        Returns: (1, n_obs + n_int [+ 1], 1, type_dim)
        """
        ids = [
            torch.zeros(n_obs, dtype=torch.long, device=device),
            torch.ones(n_int, dtype=torch.long, device=device),
        ]
        if include_query:
            ids.append(torch.full((1, ), 2, dtype=torch.long, device=device))
        type_ids = torch.cat(ids)
        return self.type_embedding(type_ids).view(1, type_ids.shape[0], 1, -1)

    def anomaly_slice(self, target_mask: torch.Tensor, batch_size: int,
                      max_nodes: int, device) -> torch.Tensor:
        """
        Per-node anomaly ids, meant to be CONCATENATED (not added) onto the
        content channels at every layer, alongside type_slice. When
        target_mask is None, every node gets id=2 ("mask not provided") so
        the reserved channel always carries a well-defined value - a
        concatenated slice can't simply be left empty the way an additive
        term can be skipped.

        Args:
            target_mask: (batch_size, max_nodes) binary mask, 1 for the
                target/symptom node, or None.
        Returns:
            (batch_size, 1, max_nodes, anomaly_dim)
        """
        if target_mask is None:
            ids = torch.full((batch_size, max_nodes),
                             2,
                             dtype=torch.long,
                             device=device)
        else:
            ids = target_mask.long()
        return self.anomaly_embedding(ids).unsqueeze(
            1)  # (batch, 1, max_nodes, anomaly_dim)


class AlternatingAttentionLayer(nn.Module):
    """
    One block of the alternating-attention scheme from Sec. 3.2.1: layers
    alternate between attention across samples (for a fixed node) and
    attention across nodes (for a fixed sample). Because obs and int samples
    now live in a single concatenated sequence, sample-level attention lets
    every sample see every other sample regardless of type.

    Standard pre-norm transformer block: LayerNorm -> sublayer -> residual
    add, for both the attention sublayer and the MLP sublayer.
    """

    def __init__(
        self,
        embedding_size,
        nhead,
        mlp_hidden_size,
        sample_level: bool,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.sample_level = sample_level
        self.attn = MultiheadAttention(embedding_size,
                                       nhead,
                                       dropout=dropout,
                                       batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = LayerNorm(embedding_size)
        self.norm2 = LayerNorm(embedding_size)
        self.mlp = nn.Sequential(
            Linear(embedding_size, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(mlp_hidden_size, embedding_size),
            nn.Dropout(dropout),
        )

    def forward(self, x, padding_mask=None):
        """
        Args:
            x: (batch, n_samples, max_nodes, embedding_size)
            padding_mask: (batch, max_nodes) - True for padded node positions
        """
        batch_size, n_samples, max_nodes, embedding_size = x.shape

        # Pre-norm attention sublayer, residual added on the raw (un-normed) input.
        # self.attn's weights are always float32 (see PRIMNeuralProcess.half()),
        # so activations are upcast right before the call and downcast right
        # after - a no-op when the model is already float32.
        in_dtype = x.dtype
        x_norm = self.norm1(x)
        if self.sample_level:
            # Attention across samples, independently per node.
            flat = (x_norm.transpose(1, 2).reshape(batch_size * max_nodes,
                                                   n_samples,
                                                   embedding_size).float())
            attn_out = self.attn(flat, flat, flat)[0].to(in_dtype)
            attn_out = attn_out.reshape(batch_size, max_nodes, n_samples,
                                        embedding_size).transpose(1, 2)
        else:
            # Attention across nodes, independently per sample.
            flat = x_norm.reshape(batch_size * n_samples, max_nodes,
                                  embedding_size).float()
            key_padding_mask = None
            if padding_mask is not None:
                key_padding_mask = (padding_mask.unsqueeze(1).expand(
                    -1, n_samples, -1).reshape(batch_size * n_samples,
                                               max_nodes))
            attn_out = self.attn(
                flat, flat, flat,
                key_padding_mask=key_padding_mask)[0].to(in_dtype)
            attn_out = attn_out.reshape(batch_size, n_samples, max_nodes,
                                        embedding_size)
        x = x + self.dropout1(attn_out)

        # Pre-norm MLP sublayer, residual added on top.
        x = x + self.dropout2(self.mlp(self.norm2(x)))
        return x


class QueryPooling(nn.Module):
    """
    Neural-process-style attentive readout (analogous to the encoder-summary
    cross-attention in Sec. 3.2.1) that maps the dataset D of samples for each
    node directly to a single summary embedding, replacing mean pooling.

    `query` is a learned matrix with one row per node, but it only seeds the
    CONTENT (value/node) channels of the query token - PRIMNeuralProcess.
    forward concatenates it with fresh type/anomaly channels (id=2, "query
    token" / "mask not provided" as appropriate) before the first attention
    layer, then re-attaches those channels after every layer, so the full
    token participates in and evolves through every node-level/sample-level
    attention op alongside the real samples (like CSIvA's "column S+1"). The
    resulting fully-evolved per-node token is passed back in here as `query`
    for the final cross-attention against the (also evolved) real samples.
    Because self-attention with no absolute positional encoding is
    permutation-equivariant, and this final step is itself a standard
    attention op, the whole readout stays exactly invariant to the order of
    the real samples regardless of how obs/int samples are interleaved or
    reordered.

    Same transformer-block convention as AlternatingAttentionLayer: a
    residual connection is kept around the cross-attention sublayer (out =
    query + attn(...)), followed by LayerNorm.
    """

    def __init__(
        self,
        embedding_size: int,
        content_dim: int,
        max_nodes: int,
        nhead: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Seeds only the content channels; type/anomaly channels are always
        # attached externally (see PRIMNeuralProcess.forward), never summed
        # into this parameter.
        self.query = nn.Parameter(
            torch.randn(1, max_nodes, content_dim) * (content_dim**-0.5))
        self.attn = MultiheadAttention(embedding_size,
                                       nhead,
                                       dropout=dropout,
                                       batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(embedding_size)

    def forward(self, x: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_samples, max_nodes, embedding_size) - real samples
                (the query token, if concatenated upstream, must be split off
                before calling this).
            query: (batch, max_nodes, embedding_size) - the fully layer-
                evolved query token (content + type + anomaly channels).
        Returns:
            (batch, max_nodes, embedding_size)
        """
        batch_size, n_samples, max_nodes, embedding_size = x.shape

        # self.attn's weights are always float32 (see PRIMNeuralProcess.half());
        # upcast activations right before the call, downcast right after - a
        # no-op when the model is already float32.
        in_dtype = query.dtype
        kv = (x.transpose(1, 2).reshape(batch_size * max_nodes, n_samples,
                                        embedding_size).float())
        q_flat = query.reshape(batch_size * max_nodes, 1, embedding_size)

        attn_out, _ = self.attn(q_flat.float(), kv, kv)
        attn_out = attn_out.to(in_dtype)
        out = q_flat + self.dropout(
            attn_out)  # residual around the cross-attention sublayer
        out = self.norm(out)
        return out.reshape(batch_size, max_nodes, embedding_size)
