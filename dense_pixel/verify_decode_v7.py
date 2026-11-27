"""Reload saved heads, rerun original RGB->DiT->pixel path, then held-out check."""
import gc
import json
import time
import torch
import run_probe as v1
import run_probe_v3b as native
from decode_unit_v5 import measure
from decode_nonlinear_v7 import RUN, video_from_patches, preview, accepted


@torch.no_grad()
def predict(weights,features):
    # Match the immutable training cache's feature dtype explicitly.
    f=features.cuda().to(torch.bfloat16).float()
    x=f[1:].reshape(-1,3072)
    first=f[0].reshape(-1,3072)@weights['first_weight']+weights['first_bias']
    base=x@weights['regular_weight']+weights['regular_bias']
    hidden=torch.nn.functional.gelu(x@weights['residual_weight1']+weights['residual_bias1'])
    regular=base+weights['residual_scale']*torch.nn.functional.linear(hidden,weights['residual_weight2'],weights['residual_bias2'])
    return video_from_patches(first,regular)


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    native.RUN=RUN
    row=v1.rows('overfit8')[0];cached=v1.load_sample(str(v1.cache_path(row)))
    model,vae,contract=native.load_native()
    rgb,gt,preprocess=v1.read_pixels(row)
    with torch.no_grad():
        source=torch.from_numpy(rgb.copy()).permute(3,0,1,2).cuda().to(torch.bfloat16)/127.5-1
        z=vae.encode([source])[0]
    del vae,source;gc.collect();torch.cuda.empty_cache()
    heads={task:torch.load(RUN/f'{task}_head.pt',map_location='cuda',weights_only=False) for task in v1.TASKS}
    results={}
    for task in v1.TASKS:
        prompt='output: depth' if task=='depth' else 'output: person alpha matte'
        result=native.rollout(model,z,task,contract['prompts'].index(prompt))
        raw=result['features'];features=raw.to(torch.bfloat16)
        delta=features.float()-cached['features'][task].float()
        pred=predict(heads[task],features)
        previous=torch.load(RUN/f'{task}_prediction.pt',map_location='cpu',weights_only=True)['prediction']
        metrics=measure(pred,gt[task].cuda(),task)
        results[task]=dict(metrics=metrics,clamped_metrics=v1.metrics(pred,gt[task].cuda(),task),
            native_feature_dtype=str(raw.dtype),head_input='native features -> BF16 cache contract -> FP32 pixel head',
            feature_max_difference=float(delta.abs().max()),feature_mse=float(delta.square().mean()),
            prediction_max_difference=float((pred.cpu()-previous).abs().max()),accepted=accepted(metrics,task),
            fresh_target_max_difference=float((gt[task]-cached['targets'][task]).abs().max()))
        print(json.dumps(dict(event='live_verification',task=task,**results[task])),flush=True)
        torch.save(pred.cpu(),RUN/f'{task}_live_prediction.pt')
        preview(cached,task,pred)
    del model,z;gc.collect();torch.cuda.empty_cache()
    # These cases were not used for whitening, fitting, or selecting checkpoints.
    holdout=[]
    for row in v1.rows('eval_candidates'):
        data=v1.load_sample(str(v1.cache_path(row)))
        record={'sample':row['sample'],'tasks':{}}
        for task in v1.TASKS:
            pred=predict(heads[task],data['features'][task])
            record['tasks'][task]=dict(unclamped=measure(pred,data['targets'][task].cuda(),task),
                clamped=v1.metrics(pred,data['targets'][task].cuda(),task),
                official_vae=v1.metrics(data['baseline'][task],data['targets'][task],task))
        holdout.append(record)
    holdout_mean={task:{key:sum(r['tasks'][task]['clamped'][key] for r in holdout)/len(holdout)
                       for key in holdout[0]['tasks'][task]['clamped']} for task in v1.TASKS}
    v1.atomic_json(RUN/'LIVE_VERIFIED.json',dict(tasks=results,all_passed=all(r['accepted'] for r in results.values()),
        source_checkpoint=native.SHA,code_sha256=v1.sha(__file__),preprocess=preprocess))
    v1.atomic_json(RUN/'HOLDOUT.json',dict(cases=holdout,mean_clamped=holdout_mean,
        scope='Single-clip fitted heads, no tuning on these cases'))
    assert all(r['accepted'] for r in results.values()),results
    print(json.dumps(dict(event='verified',holdout_mean_clamped=holdout_mean)),flush=True)


if __name__=='__main__':main()
