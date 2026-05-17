
def get_args():
    args = {}
    args["deterministic_and_reproducible"] = False
    args["num_runs"] = 3
    args["seed_always"] = []
    args["data_root"] = r"./graph"
    args["batch_size"] = 128
    args["epochs"] = 1000
    args["lr"] = 1e-4
    args["patience"] = 100
    args["optimizer"] = "adam"
    args["weight_decay"] = 1e-6
    args["max_grad_norm"] = 100.0

    args["lr_patience"] = 50
    args["lr_factor"] = 0.5
    args["min_lr"] = 1e-50


    args["K_modes_ligand"] = 1
    args["K_u_ligand"] = 200
    args["K_modes_protein"] = 1
    args["K_u_protein"] = 200
    args["K_modes_complex"] = 1
    args["K_u_complex"] = 200
    
    args["use_warmup"] = False
    args["warmup_epochs"] = 5
    args["warmup_start_factor"] = 0.1

    args["normalize_targets"] = False
    args["target_norm_eps"] = 1e-8
    return args
