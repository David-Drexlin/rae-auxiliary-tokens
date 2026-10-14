from __future__ import annotations
import csv, hashlib, json, math, random
from pathlib import Path
SCHEMA="register-decoder-training-v1"

def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def canonical_sha256(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def seed_for(seed,*parts):
    text=json.dumps([int(seed),*parts],separators=(',',':'))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'little') & ((1<<63)-1)


def read_manifest(path,require_content_hash=True,unique_parents=True):
    root=Path(path).resolve().parent
    with Path(path).open(newline='') as f:rows=list(csv.DictReader(f))
    if not rows:raise ValueError('Empty training manifest')
    ids=set();parents=set()
    for row in rows:
        if not row.get('stable_id') or not row.get('path'):raise ValueError('Manifest requires stable_id,path,label columns')
        if row['stable_id'] in ids:raise ValueError('Duplicate stable_id in training manifest')
        ids.add(row['stable_id'])
        row['label']=int(row.get('label',0))
        row['parent_id']=row.get('parent_id') or row['stable_id']
        if unique_parents and row['parent_id'] in parents:raise ValueError('Training requires one canonical image per parent')
        parents.add(row['parent_id'])
        p=Path(row['path']);row['path']=str(p if p.is_absolute() else root/p)
        digest=row.get('sha256') or row.get('image_sha256')
        if require_content_hash and (not digest or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest.lower())):
            raise ValueError('Training manifest requires image sha256 for exact data binding')
        row['sha256']=digest.lower() if digest else None
    return rows


def ordered_indices(count,seed,epoch):
    order=list(range(count));random.Random(seed_for(seed,'epoch_order',epoch)).shuffle(order);return order


def rank_microbatches(count,seed,epoch,global_batch,accum,world,rank,start_update=0):
    if global_batch<=0 or accum<=0 or world<=0 or not 0<=rank<world or global_batch%(accum*world):
        raise ValueError('Global batch must divide accumulation times world size')
    local=global_batch//(accum*world)
    order=ordered_indices(count,seed,epoch)
    for update in range(start_update,count//global_batch):
        for micro in range(accum):
            start=update*global_batch+micro*world*local+rank*local
            yield [(index,epoch) for index in order[start:start+local]]


def learning_rate(completed_updates,steps_per_epoch,training):
    base=float(training.get('lr',2e-4));floor=float(training.get('min_lr',2e-5))
    warmup=int(training.get('warmup_epochs',1))*steps_per_epoch
    decay=int(training.get('scheduler_decay_end_epoch',training.get('reference_epochs',16)))*steps_per_epoch
    if completed_updates<warmup:return base*completed_updates/max(1,warmup)
    if completed_updates>=decay:return floor
    fraction=(completed_updates-warmup)/max(1,decay-warmup)
    return floor+.5*(base-floor)*(1+math.cos(math.pi*fraction))


def validate_config(config):
    if config.get('schema_version') != SCHEMA:
        raise ValueError('Invalid decoder training configuration')
    t = config['training']
    if t['epochs'] != 16 or t['reference_epochs'] != 16 or t['scheduler_decay_end_epoch'] != 16:
        raise ValueError('The training endpoint is 16 epochs')
    if t['global_batch_size'] % (t['world_size'] * t['grad_accum_steps']):
        raise ValueError('Batch size and distributed geometry do not agree')
    params = config['model']['params']
    interface = (params['decoder_aux_mode'], params['num_aux_tokens'])
    if interface not in (('discard', 0), ('prepend', 4)) or params['aux_token_source'] != 'register':
        raise ValueError('Expected zero or four native register inputs')
    if any(params.get(k, 0) != 0 for k in ('noise_tau', 'aux_noise_sigma', 'aux_dropout_prob')):
        raise ValueError('The configured recipe uses clean inputs')
    return config

def identity(config, allow_pending=False):
    source = Path(config['source_root'])
    files = {str(p.relative_to(source)): file_sha256(p) for p in sorted(source.rglob('*.py'))}
    assets = {}
    declared = dict(config['asset_bindings'])
    for key in ('train_manifest', 'monitor_manifest', 'final_manifest'):
        if config['data'].get(key):
            declared[key] = config['data'][key]
    for key, path in declared.items():
        assets[key] = {'path': str(Path(path).resolve()), 'sha256': file_sha256(path)}
    for key, digest in config['expected_asset_sha256'].items():
        if assets[key]['sha256'] != digest:
            raise ValueError('Asset checksum does not match: ' + key)
    result = {'config_sha256': canonical_sha256(config), 'source_files': files, 'source_sha256': canonical_sha256(files), 'assets': assets}
    result['binding_sha256'] = canonical_sha256(result)
    return result

def validate_calibration(config):
    sets = {}
    counts = {}
    for name in ('train', 'monitor', 'final'):
        rows = read_manifest(config['data'][name + '_manifest'], True)
        sets[name] = ({r['parent_id'] for r in rows}, {r['sha256'] for r in rows})
        counts[name] = len(rows)
    for a, b in (('train', 'monitor'), ('train', 'final'), ('monitor', 'final')):
        if sets[a][0] & sets[b][0] or sets[a][1] & sets[b][1]:
            raise ValueError('Image membership overlaps between ' + a + ' and ' + b)
    if counts['train'] != config['data']['expected_train_count'] or counts['final'] != 50000:
        raise ValueError('The configured image population does not match')
    return {'status': 'pass', 'image_counts': counts, 'final_evaluation_used': False}

def validate_runtime_admission(config, binding):
    return {'actual_trainer_admitted': False, 'scientific_training_admitted': False, 'scope': 'source_export'}
