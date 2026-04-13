"""HELM-GNN configuration."""

from transformers import PretrainedConfig


class HELMGNNConfig(PretrainedConfig):
    """Configuration for HELM-GNN model.

    Extends the original HELM-BERT Transformer config with GPS encoder
    and graph distance bias parameters.
    """

    model_type = "helmgnn"

    def __init__(
        self,
        # Monomer vocab
        vocab_size: int = 3103,  # ~3098 monomers + 5 special tokens
        # GPS encoder
        gps_hidden_dim: int = 256,
        gps_num_layers: int = 3,
        gps_num_heads: int = 8,
        gps_dropout: float = 0.1,
        # Transformer encoder
        hidden_size: int = 768,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        # Disentangled attention
        max_relative_positions: int = 512,
        position_buckets: int = 256,
        pos_att_type: str = "c2p|p2c",
        share_att_key: bool = False,
        # Graph distance bias
        max_graph_distance: int = 32,
        num_graph_distance_buckets: int = 33,  # 0..32
        # nGiE
        ngie_kernel_size: int = 3,
        ngie_dropout: float = 0.1,
        # Special tokens
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        sep_token_id: int = 2,
        mask_token_id: int = 4,
        # Classification/regression
        num_labels: int = 2,
        problem_type: str = None,
        classifier_num_layers: int = 0,
        classifier_dropout: float = 0.1,
        # Monomer library
        monomer_library_path: str = "data/monomer_library/helm_monomer_library.csv",
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size

        # GPS
        self.gps_hidden_dim = gps_hidden_dim
        self.gps_num_layers = gps_num_layers
        self.gps_num_heads = gps_num_heads
        self.gps_dropout = gps_dropout

        # Transformer
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings

        # Disentangled attention
        self.max_relative_positions = max_relative_positions
        self.position_buckets = position_buckets
        self.pos_att_type = pos_att_type
        self.share_att_key = share_att_key

        # Graph distance
        self.max_graph_distance = max_graph_distance
        self.num_graph_distance_buckets = num_graph_distance_buckets

        # nGiE
        self.ngie_kernel_size = ngie_kernel_size
        self.ngie_dropout = ngie_dropout

        # Special tokens
        self.sep_token_id = sep_token_id
        self.mask_token_id = mask_token_id

        # Classification
        self.num_labels = num_labels
        self.problem_type = problem_type
        self.classifier_num_layers = classifier_num_layers
        self.classifier_dropout = classifier_dropout

        # Monomer library
        self.monomer_library_path = monomer_library_path
