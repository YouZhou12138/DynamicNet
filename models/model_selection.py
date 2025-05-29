
from models.DynamicNet_multi.model import DynamicNet_multi
from models.DynamicNet_single.dynamic_net_single import DynamicNet_sigle
from models.RNN.ct_rnn import CT_RNN

MODELS = {
    "DynamicNet_single": DynamicNet_sigle,
    "DynamicNet_multi": DynamicNet_multi,
    "CT_RNN": CT_RNN,
    # "AttenRNN": AttenRNN,
    # "RNN": RNN,
}