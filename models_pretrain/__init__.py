from .rnn_pretrain import MyRNN
try:
    from .transformer_pretrain import MyTransformer
    from .transformerxl_pretrain import MyTransformerXL
except:
    pass


# load pretrained model
def get_pretrain(pretrain_stage_config):
    if pretrain_stage_config['pretrain_model'].lower() in {'rnn', 'lstm', 'gru'}:
        pretrained = MyRNN(pretrain_stage_config)
    elif pretrain_stage_config['pretrain_model'].lower() == 'transformer':
        pretrained = MyTransformer(pretrain_stage_config)
    elif pretrain_stage_config['pretrain_model'].lower() == 'transformerxl':
        pretrained = MyTransformerXL(pretrain_stage_config)
    return pretrained