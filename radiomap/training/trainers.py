import torch
from torch import nn
from radiomap.data.geometry import geometry_region_stats, region_sum
from radiomap.flow.path import interpolate, velocity_target, clean_from_velocity
from radiomap.flow.time_sampler import history_times
from radiomap.training.losses import generation_loss_sum, region_errors, risk_loss_sum


def generator_optimizer(system, config, lr=None):
    decay, no_decay = [],[]
    generator_ids = {id(parameter) for parameter in system.generator_parameters()}
    for module in system.modules():
        for name,parameter in module.named_parameters(recurse=False):
            if id(parameter) not in generator_ids or not parameter.requires_grad:
                continue
            target = decay if isinstance(module,(nn.Conv2d,nn.Linear)) and name=="weight" else no_decay
            target.append(parameter)
    options = config["optimizer"]
    return torch.optim.AdamW([{"params":decay,"weight_decay":options["generator_weight_decay"]},
                              {"params":no_decay,"weight_decay":0.}],
                             lr=lr or config["train"]["base_lr"],betas=tuple(options["betas"]),eps=options["eps"])


def risk_optimizer(provider, config, lr=None):
    parameters = list(provider.parameters()) if provider is not None else []
    if not parameters:
        return None
    options = config["optimizer"]
    return torch.optim.AdamW(parameters,lr=lr or config["train"]["finetune_risk_lr"],
                             betas=tuple(options["betas"]),eps=options["eps"],weight_decay=0.)


def batch_losses(system, batch, random, config, operator, provider=None):
    """Return sums so micro-batches can be normalized using GLOBAL denominators."""
    target,noise,time = batch["x"].float(),random["noise"].float(),random["time"]
    context = operator.prepare(batch["building"],batch["tx_xy"])
    region_size,minimum = config["risk"]["region_size"],config["risk"]["min_valid_pixels"]
    stats = geometry_region_stats(batch["building"],context.maps,region_size)
    count = region_sum(context.free,region_size)
    trains_risk = provider is not None and provider.mode != "unit"
    target_reference = operator.apply(target,context).detach() if trains_risk else None
    amp = config["train"]["precision"] == "bf16"
    with torch.autocast(device_type=target.device.type,dtype=torch.bfloat16,enabled=amp):
        features = system.encode_condition(batch)
    risk_sums = []

    def predict(at, reference=None, confidence=None, track=False, previous_clean=None, previous_time=None,build_next=True):
        state = interpolate(target,noise,at)
        with torch.set_grad_enabled(track):
            with torch.autocast(device_type=target.device.type,dtype=torch.bfloat16,enabled=amp):
                velocity,hidden = system.predict(state,at,features,reference,confidence)
            clean = clean_from_velocity(state,velocity.float(),at)
        clean_reference = operator.apply(clean.detach(),context) if build_next or trains_risk else None
        error = region_errors(clean_reference,target_reference,context.free,region_size)[0] if trains_risk else None
        if provider is None or provider.mode=="unit":
            confidence_next = context.free
        else:
            raw = provider.raw(hidden.detach(),features[-1].detach(),at,stats,context.free,clean.detach(),
                               previous_clean,previous_time,oracle_error=error)
            confidence_next = provider.gate(raw,count,at,context.free) if build_next else None
            if provider.mode != "oracle":
                risk_sums.append(risk_loss_sum(raw,error,count,minimum)[0])
        return velocity,clean.detach(),clean_reference.detach() if clean_reference is not None else None,confidence_next.detach() if confidence_next is not None else None

    context_mode = config.get("context","same_time_two_pass")
    if context_mode=="single_pass":
        with torch.autocast(device_type=target.device.type,dtype=torch.bfloat16,enabled=amp):
            velocity,_ = system.predict(interpolate(target,noise,time),time,features)
    elif context_mode=="same_time_two_pass":
        _,_,reference,confidence = predict(time)
        keep = (~random["null"]).reshape(-1,1,1,1)
        velocity,_,_,_ = predict(time,reference*keep,confidence*keep,track=True,build_next=False)
    elif context_mode=="lagged_history_three_pass":
        previous,reference_time,now = history_times(time,random["history_nfe"])
        _,clean0,reference0,confidence0 = predict(previous)
        _,clean1,reference1,confidence1 = predict(reference_time,reference0,confidence0,
                                                previous_clean=clean0,previous_time=previous)
        keep = (~random["null"]).reshape(-1,1,1,1)
        velocity,_,_,_ = predict(now,reference1*keep,confidence1*keep,track=True,
                                 previous_clean=clean1,previous_time=reference_time,build_next=False)
    else:
        raise ValueError(f"Unsupported context mode {context_mode}")
    generation_sum,pixels = generation_loss_sum(velocity,velocity_target(target,noise))
    risk_sum = torch.stack(risk_sums).mean() if risk_sums else None
    return {"generation_sum":generation_sum,"pixels":pixels,"risk_sum":risk_sum,
            "per_sample_mse":(velocity.detach().float()-velocity_target(target,noise)).square().flatten(1).mean(1),
            "eligible_regions":int((count>=minimum).sum().item())}


def train_step(system,batch,random,config,operator,optimizer,step=0,provider=None,provider_optimizer=None,micro_batch=None):
    system.train()
    micro = micro_batch or config["train"]["micro_batch"] or batch["x"].shape[0]
    batch_size = batch["x"].shape[0]
    global_pixels = batch["x"].numel()
    free = (~batch["building"].bool()).float()
    counts = region_sum(free,config["risk"]["region_size"])
    global_regions = int((counts>=config["risk"]["min_valid_pixels"]).sum().item())
    finetune = config.get("context")=="lagged_history_three_pass"
    warmup = config["train"]["finetune_warmup" if finetune else "base_warmup"]
    base_lr = config["train"]["finetune_lr" if finetune else "base_lr"]
    for group in optimizer.param_groups:
        group["lr"] = base_lr*min((step+1)/warmup,1.)
    if provider_optimizer is not None:
        for group in provider_optimizer.param_groups:
            group["lr"] = config["train"]["finetune_risk_lr"]*min((step+1)/config["train"]["finetune_warmup"],1.)
    optimizer.zero_grad(set_to_none=True)
    if provider_optimizer is not None:
        provider_optimizer.zero_grad(set_to_none=True)
    generation_total,risk_total = 0.,0.
    bin_counts=[0]*10;bin_loss_sums=[0.]*10
    for start in range(0,batch_size,micro):
        stop = min(start+micro,batch_size)
        sub_batch = {key:value[start:stop] for key,value in batch.items() if isinstance(value,torch.Tensor)}
        sub_random = {key:value[start:stop] for key,value in random.items()}
        losses = batch_losses(system,sub_batch,sub_random,config,operator,provider)
        normalized = losses["generation_sum"]/global_pixels
        generation_total += float(normalized.detach())
        bins=(sub_random["time"]*10).long().clamp(0,9)
        counts_local=torch.bincount(bins,minlength=10).cpu().tolist()
        sums_local=torch.zeros(10,device=bins.device).scatter_add_(0,bins,losses["per_sample_mse"]).cpu().tolist()
        bin_counts=[a+b for a,b in zip(bin_counts,counts_local)]
        bin_loss_sums=[a+b for a,b in zip(bin_loss_sums,sums_local)]
        if provider_optimizer is not None and global_regions and losses["risk_sum"] is not None:
            risk = losses["risk_sum"]/global_regions
            normalized = normalized+risk
            risk_total += float(risk.detach())
        if not torch.isfinite(normalized):
            raise FloatingPointError("Non-finite complete training loss")
        normalized.backward()
    generator_norm = torch.nn.utils.clip_grad_norm_(list(system.generator_parameters()),config["optimizer"]["generator_global_grad_clip"],error_if_nonfinite=True)
    risk_norm = None
    if provider_optimizer is not None and global_regions:
        risk_norm = torch.nn.utils.clip_grad_norm_(list(provider.parameters()),config["optimizer"]["risk_global_grad_clip"],error_if_nonfinite=True)
    optimizer.step()
    if provider_optimizer is not None and global_regions:
        provider_optimizer.step()
    return {"generation_loss":generation_total,"risk_loss":risk_total,
            "generator_grad_norm":float(generator_norm),"risk_grad_norm":float(risk_norm) if risk_norm is not None else None,
            "generator_clipped":bool(generator_norm>config["optimizer"]["generator_global_grad_clip"]),
            "eligible_regions":global_regions,"risk_step_skipped":provider_optimizer is not None and global_regions==0,
            "time_bin_counts":bin_counts,"time_bin_mse_sums":bin_loss_sums,
            "lr":optimizer.param_groups[0]["lr"],"batch_size":batch_size,"micro_batch":micro}
