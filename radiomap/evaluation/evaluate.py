import csv
from pathlib import Path
import time
import torch
from torch.utils.data import default_collate
from radiomap.data.dataset import DPMDataset
from radiomap.flow.solver import sample
from radiomap.reference.operators import ReferenceOperator
from radiomap.utils.seed import evaluation_noise
from radiomap.utils.config import write_json
from radiomap.utils.profiling import to_device,synchronize


@torch.no_grad()
def evaluate(system,config,evaluation_samples,noise_seeds,output,provider=None,batch_size=8,nfe=25):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    device=next(system.parameters()).device
    scene_ids=list(dict.fromkeys(str(x["scene_id"]) for x in evaluation_samples))
    dataset=DPMDataset(config["data"]["root"],scene_ids,image_size=config["data"]["image_size"])
    indices=[scene_ids.index(str(s["scene_id"]))*80+int(s["tx_id"]) for s in evaluation_samples]
    operator=ReferenceOperator(**config["reference"])
    rows=[];synchronize(device);start=time.perf_counter()
    for noise_seed in noise_seeds:
        for offset in range(0,len(indices),batch_size):
            values=[dataset[index] for index in indices[offset:offset+batch_size]]
            batch=to_device(default_collate(values),device)
            condition={key:value for key,value in batch.items() if key!="x"}
            noise=torch.stack([evaluation_noise(noise_seed,item["scene_id"],item["tx_id"],tuple(item["x"].shape)) for item in values]).to(device)
            prediction,_=sample(system,condition,noise,operator,nfe=nfe,provider=provider,precision=config["train"]["precision"])
            error=prediction.double()-batch["x"].double()
            free=(~batch["building"].bool()).double()
            sse=error.square().flatten(1).sum(1).cpu().tolist()
            free_sse=(error.square()*free).flatten(1).sum(1).cpu().tolist()
            free_pixels=free.flatten(1).sum(1).cpu().tolist()
            absolute=error.abs().flatten(1).sum(1).cpu().tolist()
            bias=error.flatten(1).sum(1).cpu().tolist()
            below=(prediction<0).flatten(1).sum(1).cpu().tolist()
            above=(prediction>1).flatten(1).sum(1).cpu().tolist()
            for index,item in enumerate(values):
                rows.append({"scene_id":item["scene_id"],"tx_id":item["tx_id"],"noise_seed":int(noise_seed),
                             "sse":sse[index],"pixels":error[0].numel(),"free_sse":free_sse[index],
                             "free_pixels":int(free_pixels[index]),"absolute_error_sum":absolute[index],
                             "bias_sum":bias[index],"below_zero":int(below[index]),"above_one":int(above[index])})
    synchronize(device);elapsed=time.perf_counter()-start
    totals={key:sum(row[key] for row in rows) for key in rows[0] if key not in ("scene_id","tx_id","noise_seed")}
    metrics={**totals,"raw_rmse":(totals["sse"]/totals["pixels"])**.5,
             "free_raw_rmse":(totals["free_sse"]/totals["free_pixels"])**.5,
             "mae":totals["absolute_error_sum"]/totals["pixels"],"bias":totals["bias_sum"]/totals["pixels"],
             "below_zero_fraction":totals["below_zero"]/totals["pixels"],"above_one_fraction":totals["above_one"]/totals["pixels"],
             "images_per_repeat":len(evaluation_samples),"noise_seeds":list(noise_seeds),"nfe":nfe,
             "aggregate_repeats":"pool_sse_not_average_images","elapsed_seconds":elapsed,"batch_size":batch_size}
    scenes=[]
    for scene in scene_ids:
        items=[row for row in rows if row["scene_id"]==scene]
        scenes.append({"scene_id":scene,**{key:sum(row[key] for row in items) for key in totals}})
    with (output/"per_sample.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (output/"per_scene.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(scenes[0]));writer.writeheader();writer.writerows(scenes)
    write_json(output/"metrics.json",metrics,replace=True)
    return metrics
