import torch
from radiomap.data.geometry import geometry_region_stats, region_sum
from radiomap.flow.path import clean_from_velocity


@torch.no_grad()
def sample(system, condition, noise, operator, nfe=25, provider=None, trace=None, precision="fp32"):
    if "x" in condition or "target" in condition:
        raise ValueError("Formal sample accepts conditions only, never ground truth")
    if provider is not None and provider.mode == "oracle":
        raise ValueError("Oracle is forbidden in the deployable sampler")
    if nfe < 1:
        raise ValueError("NFE must be positive")
    system.eval()
    amp = precision == "bf16"
    with torch.autocast(device_type=noise.device.type,dtype=torch.bfloat16,enabled=amp):
        features = system.encode_condition(condition)
    context = operator.prepare(condition["building"],condition["tx_xy"])
    reference, confidence, state = torch.zeros_like(noise),torch.zeros_like(noise),noise.float().clone()
    previous_clean, previous_time = None,None
    region_size = provider.region_size if provider is not None else 16
    stats = geometry_region_stats(condition["building"],context.maps,region_size)
    count = region_sum(context.free,region_size)
    for index in range(nfe):
        time = torch.full((noise.shape[0],),index/nfe,device=noise.device,dtype=torch.float32)
        input_reference,input_confidence = reference,confidence
        with torch.autocast(device_type=noise.device.type,dtype=torch.bfloat16,enabled=amp):
            velocity,hidden = system.predict(state,time,features,reference,confidence)
        clean = clean_from_velocity(state,velocity.float(),time)
        state = state+velocity.float()/nfe
        raw = None
        if index < nfe-1 and operator.mode != "none":
            reference = operator.apply(clean,context)
            if provider is None or provider.mode == "unit":
                confidence = context.free
            else:
                raw = provider.raw(hidden,features[-1],time,stats,context.free,clean,previous_clean,previous_time)
                confidence = provider.gate(raw,count,time,context.free)
        if trace is not None:
            trace({"index": index,"time": time,"clean": clean,"hidden": hidden,
                   "input_reference": input_reference,"input_confidence": input_confidence,
                   "next_reference": reference if index<nfe-1 else None,
                   "next_confidence": confidence if index<nfe-1 else None,"raw_risk": raw})
        previous_clean,previous_time = clean,time
    return state,{"nfe": nfe,"condition_encoder_calls": 1,
                 "reference_calls": nfe-1 if operator.mode!="none" else 0,
                 "risk_calls": nfe-1 if provider is not None and provider.mode!="unit" and operator.mode!="none" else 0}
