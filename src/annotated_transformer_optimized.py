import gc
from os.path import exists
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.profiler import profile, record_function, ProfilerActivity

# Set to False to skip notebook execution (e.g. for debugging)
warnings.filterwarnings("ignore")
# Some convenience helper functions used throughout the notebook


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask, loss_compute, labels, ntokens):
        "Take in and process masked src and target sequences."
        logits = self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
        return loss_compute(logits, labels, ntokens)

    # def forward(self, src, tgt, src_mask, tgt_mask):
    #     "Take in and process masked src and target sequences."
    #     return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    # 20240312 modified
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.features = features
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        return F.layer_norm(x, (self.features,), self.a_2, self.b_2, self.eps)

    # def __init__(self, features, eps=1e-6):
    #     super(LayerNorm, self).__init__()
    #     self.a_2 = nn.Parameter(torch.ones(features))
    #     self.b_2 = nn.Parameter(torch.zeros(features))
    #     self.eps = eps

    # def forward(self, x):
    #     mean = x.mean(-1, keepdim=True)
    #     std = x.std(-1, keepdim=True)
    #     return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# class DecoderLayer(nn.Module):
#     "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

#     def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
#         super(DecoderLayer, self).__init__()
#         self.size = size
#         self.self_attn = self_attn
#         self.src_attn = src_attn
#         self.feed_forward = feed_forward
#         self.sublayer = clones(SublayerConnection(size, dropout), 3)

#     def forward(self, x, memory, src_mask, tgt_mask):
#         "Follow Figure 1 (right) for connections."
#         m = memory
#         x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
#         x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
#         return self.sublayer[2](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        self.memory_scaling = nn.Parameter(torch.ones(size))  # Trainable parameter

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory * self.memory_scaling  # Multiply memory by the trainable parameter
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


# def attention(query, key, value, mask=None, dropout=None):
#     "Compute 'Scaled Dot Product Attention'"
#     d_k = query.size(-1)
#     scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
#     if mask is not None:
#         scores = scores.masked_fill(mask == 0, -1e9)
#     p_attn = scores.softmax(dim=-1)
#     if dropout is not None:
#         p_attn = dropout(p_attn)
#     return torch.matmul(p_attn, value), p_attn


def attention(query, key, value, mask=None, dropout=None):
    # Compute 'Scaled Dot Product Attention' using PyTorch's optimized function
    scale = 1 / math.sqrt(query.size(-1))
    attn_mask = mask if mask is None or mask.dtype == torch.bool else mask.float()

    # Call PyTorch's scaled dot product attention function
    attn_output = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=0.0 if dropout is None else dropout.p, scale=scale
    )

    # Return attention output and weights (weights are not returned by the function)
    # If you need to track the attention weights, you'll have to compute them separately

    return attn_output, None  # Returning None for the weights


# class MultiHeadedAttention(nn.Module):
#     def __init__(self, h, d_model, dropout=0.1):
#         "Take in model size and number of heads."
#         super(MultiHeadedAttention, self).__init__()
#         assert d_model % h == 0
#         # We assume d_v always equals d_k
#         self.d_k = d_model // h
#         self.h = h
#         self.linears = clones(nn.Linear(d_model, d_model), 4)
#         self.attn = None
#         self.dropout = nn.Dropout(p=dropout)

#     def forward(self, query, key, value, mask=None):
#         "Implements Figure 2"
#         # 1) Do all the linear projections in batch from d_model => h x d_k
#         stream1 = torch.cuda.Stream()
#         stream2 = torch.cuda.Stream()
#         stream3 = torch.cuda.Stream()

#         # if mask is not None:
#         #     # Same mask applied to all h heads.
#         #     mask = mask.unsqueeze(1)
#         mask = mask.unsqueeze(1)
#         nbatches = query.size(0)

#         torch.cuda.synchronize()

#         with torch.cuda.stream(stream1):
#             query = self.linears[0](query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
#         with torch.cuda.stream(stream2):
#             key = self.linears[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
#         with torch.cuda.stream(stream3):
#             value = self.linears[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

#         torch.cuda.synchronize()

#         # destroy streams
#         del stream1, stream2, stream3
#         # print("Stream: ", query.shape, key.shape, value.shape, mask.shape)
#         # print(query_)
#         # 2) Apply attention on all the projected vectors in batch.
#         x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

#         # 3) "Concat" using a view and apply a final linear.
#         x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
#         # print("Stream: ", x)
#         del query
#         del key
#         del value
#         return self.linears[-1](x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        del query
        del key
        del value
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model  # .half()


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


def loss(x, crit):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
    return crit(predict.log(), torch.LongTensor([1])).data


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        return sloss.data * norm, sloss


def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
    pad_id=2,
):
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for _src, _tgt in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    # src_vocab(src_pipeline(_src)),
                    [src_vocab[tok] for tok in src_pipeline.encode(_src).tokens],
                    dtype=torch.int16,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    # tgt_vocab(tgt_pipeline(_tgt)),
                    [tgt_vocab[tok] for tok in tgt_pipeline.encode(_tgt).tokens],
                    dtype=torch.int16,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(F.pad(processed_src, (0, max_padding - len(processed_src)), value=pad_id))
        tgt_list.append(F.pad(processed_tgt, (0, max_padding - len(processed_tgt)), value=pad_id))

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)


from torch.utils.data import Dataset, DataLoader, DistributedSampler
import pickle


def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    tokenizer_src,
    tokenizer_tgt,
    batch_size=12000,
    max_padding=128,
    is_distributed=True,
):
    # def create_dataloaders(batch_size=12000):
    # def tokenize_de(text):
    #     return tokenize(text, spacy_de)

    # def tokenize_en(text):
    #     return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenizer_src,
            tokenizer_tgt,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=2,
        )

    # train_iter, valid_iter, test_iter = datasets.Multi30k(
    #     language_pair=("de", "en")
    # )

    class CustomDataset(Dataset):
        def __init__(self, src_data, tgt_data):
            self.src_data = src_data
            self.tgt_data = tgt_data
            assert len(self.src_data) == len(self.tgt_data), "Source and target datasets must have the same length"

        def __len__(self):
            return len(self.src_data)

        def __getitem__(self, idx):
            return self.src_data[idx], self.tgt_data[idx]

    def to_map_style_dataset(data):
        src_data, tgt_data = data[0], data[1]  # Assuming data is a list of tuples (src_sentence, tgt_sentence)
        return CustomDataset(src_data, tgt_data)

    torch.random.manual_seed(42)

    with open("./data/dataset_500K_train.pkl", "rb") as f:
        # with open("./data/dataset_500K_train_sorted.pkl", "rb") as f:
        train_iter = pickle.load(f)[:10_000]
    with open("./data/dataset_500K_val.pkl", "rb") as f:
        valid_iter = pickle.load(f)

    print(f"Train dataset length: {len(train_iter[0])}")
    print(f"Valid dataset length: {len(valid_iter[0])}")

    train_iter_map = to_map_style_dataset(train_iter)  # DistributedSampler needs a dataset len()
    train_sampler = DistributedSampler(train_iter_map) if is_distributed else None
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = DistributedSampler(valid_iter_map) if is_distributed else None

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        # shuffle=False,
        sampler=train_sampler,
        collate_fn=collate_fn,
        pin_memory=False,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        # shuffle=False,
        sampler=valid_sampler,
        collate_fn=collate_fn,
        pin_memory=False,
    )
    return train_dataloader, valid_dataloader


import concurrent.futures


def save_model_async(model_state, file_path):
    torch.save(model_state, file_path)


def train_worker(
    gpu,
    ngpus_per_node,
    vocab_src,
    vocab_tgt,
    tokenizer_src,
    tokenizer_tgt,
    config,
    is_distributed=False,
):
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)
    torch.cuda.empty_cache()
    gc.collect()

    torch.random.manual_seed(42)

    pad_idx = 2
    d_model = 512
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.cuda(gpu)
    module = model
    is_main_process = True
    if is_distributed:
        dist.init_process_group("nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node)
        model = DDP(model, device_ids=[gpu], gradient_as_bucket_view=True, bucket_cap_mb=100)
        module = model.module
        is_main_process = gpu == 1
        # is_main_process = gpu == 0

    criterion = LabelSmoothing(size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda(gpu)

    train_dataloader, valid_dataloader = create_dataloaders(
        gpu,
        vocab_src,
        vocab_tgt,
        tokenizer_src,
        tokenizer_tgt,
        batch_size=config["batch_size"] // ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
    )

    scaler = GradScaler()

    optimizer = torch.optim.Adam(model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, d_model, factor=1, warmup=config["warmup"]),
    )
    train_state = TrainState()

    # scaler = GradScaler()

    print(torch.cuda.memory_summary())
    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
            scaler=scaler,
            device=gpu,
        )


        GPUtil.showUtilization()
        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            # torch.save(module.state_dict(), file_path)
            model_state = module.state_dict()

            # 비동기 파일 저장
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(save_model_async, model_state, file_path)

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
            scaler=scaler,
            device=gpu,
        )
        print("Validation Loss: ", sloss)
        gc.collect()
        torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)
        gc.collect()
        torch.cuda.empty_cache()


def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=4,  # Assume we want to accumulate gradients over 4 steps
    train_state=TrainState(),
    scaler=GradScaler(),
    device="cuda",
):
    start_time = time.time()
    total_tokens = 0
    total_loss = torch.tensor(0.0, device=device)
    tokens = 0
    n_accum = 0
    logs = []
    model = DDP(model)

    if mode == "train" or mode == "train+log":
        optimizer.zero_grad(set_to_none=True)  # Initialize gradients

        
        for i, batch in enumerate(data_iter):
            with autocast():  # Apply mixed precision
                # Use no_sync() context for all but the last accumulation step
                if (i + 1) % accum_iter != 0:
                    with model.no_sync():  # Disable gradient synchronization
                        loss, loss_node = model(
                            batch.src,
                            batch.tgt,
                            batch.src_mask,
                            batch.tgt_mask,
                            loss_compute,
                            batch.tgt_y,
                            batch.ntokens,
                        )
                        # loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
                        # out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
                        # loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
                        scaler.scale(loss_node).backward()  # Accumulate gradients locally
                else:
                    loss, loss_node = model(
                        batch.src,
                        batch.tgt,
                        batch.src_mask,
                        batch.tgt_mask,
                        loss_compute,
                        batch.tgt_y,
                        batch.ntokens,
                    )
                    # out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
                    # loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
                    scaler.scale(loss_node).backward()  # This backward pass synchronizes gradients

            if (i + 1) % accum_iter == 0:
                scaler.step(optimizer)  # Update model parameters
                scaler.update()
                optimizer.zero_grad(set_to_none=True)  # Reset gradients
                n_accum += 1
                train_state.accum_step += 1
                scheduler.step()  # Adjust learning rate

            total_loss += loss.detach()
            total_tokens += batch.ntokens
            tokens += batch.ntokens

            if i % 50 == 1:
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start_time
                log_msg = f"Epoch Step: {i:6d} | Accumulation Step: {n_accum:3d} | Loss: {loss/batch.ntokens:6.2f} | Tokens / Sec: {tokens / elapsed:7.1f} | Learning Rate: {lr:6.1e} | Time Elapsed: {elapsed:6.2f}"
                logs.append(log_msg)
                print(log_msg, flush=True)
                # torch.cuda.empty_cache()

        # Output logs at the end of the epoch
        if mode == "train+log":
            for log in logs:
                print(log, flush=True)

    return total_loss.item() / total_tokens, train_state
