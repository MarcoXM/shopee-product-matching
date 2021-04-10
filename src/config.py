

# loss_module = 'cosface' # ,'arcface'  #'adacos'
loss_module = 'arcface'
batch_size = 16
model_params = {
    'n_classes':11014,
    'model_name':'tf_efficientnet_b4',
    'use_fc':False,
    'fc_dim':512,
    'dropout':0.0,
    'loss_module':loss_module,
    's':30.0,
    'margin':0.50,
    'ls_eps':0.0,
    'theta_zero':0.785,
    'pretrained':True

scheduler_params = {
        "lr_start": 1e-5,
        "lr_max": 1e-5 * 16,
        "lr_min": 1e-6,
        "lr_ramp_ep": 5,
        "lr_sus_ep": 0,
        "lr_decay": 0.8,
    }