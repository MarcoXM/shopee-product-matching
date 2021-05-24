

# loss_module = 'cosface' # ,'arcface'  #'adacos'
loss_module = 'arcface'
batch_size = 64
model_params = {
    'n_classes':11014,
    'model_name':'tf_efficientnet_b3_ns',
    'use_fc':False,
    'fc_dim':512,
    'dropout':0.1,
    'loss_module':loss_module,
    's':30.0,
    'margin':0.50,
    'ls_eps':0.1,
    'theta_zero':0.785,
    'pretrained':True,
}

scheduler_params = {
        "lr_start": 1e-4,
        "lr_max": 1e-5 * batch_size,
        "lr_min": 1e-6,
        "lr_ramp_ep": 5,
        "lr_sus_ep": 0,
        "lr_decay": 0.8,
    }
