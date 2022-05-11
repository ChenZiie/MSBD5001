config = {
    'prompt': 'Overall it was [MASK].',
    'use_prompt': True,
    'mask_location': 4,
    'prefix_pormpt': True,

    'regression_model' : False,

    'train_batch_size': 128,
    'test_batch_size': 64,
    "learn_rate": 0.00001,
    "lr_dc_step": 2,
    "lr_dc": 0.99,
    "dropout": 0.5,
    "train_epoch": 20,
    'max_len': 64,
    "pretrain_epoch": 1,
    "margine": 0.5,
    "seq_feature_dim": 768,  # 768#384
    "pretrain_model": "activebus/BERT-XD_Review",
    "English_Only": True
    #"allenai/reviews_roberta_base" 768
    #"activebus/BERT-XD_Review" 768
    #"activebus/BERT_Review" 768
    #
}
