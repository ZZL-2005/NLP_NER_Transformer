from models.TransformerNER import TransformerNER
from models.Attention_Residual_Kimi import AttnResTransformerNER,AttnResTransformerNER_MY
from torch.utils.data import DataLoader
from dataset.dataset import NERDataset, collate_fn

def build_dataloaders(args, char2id, tag2id):
    train_dataset = NERDataset(
        data_path=args.train_path,
        tag_path=args.train_tag_path,
        char2id=char2id,
        tag2id=tag2id,
    )
    dev_dataset = NERDataset(
        data_path=args.dev_path,
        tag_path=args.dev_tag_path,
        char2id=char2id,
        tag2id=tag2id,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, dev_loader

def build_model(args, vocab_size, num_tags, pad_token_id):
    if args.model_name == "transformer":
        model = TransformerNER(
            vocab_size=vocab_size,
            num_tags=num_tags,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            num_layers=args.num_layers,
            max_len=args.max_len,
            dropout=args.dropout,
            pad_token_id=pad_token_id,
        )
    elif args.model_name == "attnres_transformer":
        model = AttnResTransformerNER(
            vocab_size=vocab_size,
            num_tags=num_tags,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            num_layers=args.num_layers,
            max_len=args.max_len,
            dropout=args.dropout,
            pad_token_id=pad_token_id,)
    elif args.model_name == "attnres_transformer_my":
        model = AttnResTransformerNER_MY(
            vocab_size=vocab_size,
            num_tags=num_tags,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            num_layers=args.num_layers,
            max_len=args.max_len,
            dropout=args.dropout,
            pad_token_id=pad_token_id,)
    else:
        raise ValueError(f"不支持的 model_name: {args.model_name}")

    return model
