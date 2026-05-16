"""
debug_compare.py
================
Step-by-step decode comparison between original and current Sticky KV.

Generates one token at a time and records: token_id, token_str, position_ids,
physical cache length, global_token_counter, tokens_since_last_review,
qcache_active.  The per-step state is captured via lightweight forward hooks
on layer-0 attention only, plus Q/K projection output hooks for norms.

USAGE
-----
    # Run original, save trace
    cd /path/to/qevict && python debug_compare.py --mode original --steps 30 --out trace_original.json

    # Run current
    python debug_compare.py --mode current --steps 30 --out trace_current.json

    # Run current with Q_RATIO=0  (disables q-cache → tests hypothesis H4)
    python debug_compare.py --mode current --q_ratio 0 --steps 30 --out trace_noq.json

    # Run current with OMEGA=9999 (disables decode eviction → tests hypothesis H2)
    python debug_compare.py --mode current --omega 9999 --steps 30 --out trace_noev.json

    # Compare two saved traces side-by-side
    python debug_compare.py --diff trace_original.json trace_current.json
"""

import sys, os, json, argparse, gc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ORIG_ROOT: directory containing the original Sticky KV source files.
# Probe by marker files so this works whether running from the main project,
# a sibling folder, or a .claude/worktrees/<name>/ worktree (3 levels deep).
def _find_orig_root():
    markers = ("configuration_sticky_llama.py", "sticky_config.py")
    candidates = [
        os.path.join(SCRIPT_DIR, "Qevict"),
        SCRIPT_DIR,
        os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "..")),
        os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..")),
    ]
    for c in candidates:
        if os.path.isdir(c) and all(os.path.isfile(os.path.join(c, m)) for m in markers):
            return c
    return os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

ORIG_ROOT = _find_orig_root()

# ──────────────────────────────────────────────────────────────────────────────
# ARGUMENTS
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--mode",      choices=["original", "current"], default="current")
parser.add_argument("--steps",     type=int, default=30, help="Max new tokens to generate")
parser.add_argument("--sample",    type=int, default=0,  help="LCC sample index (0-based from filtered set)")
parser.add_argument("--out",       type=str, default=None)
parser.add_argument("--q_ratio",   type=int, default=None, help="Override Q_RATIO (0 = no q-cache)")
parser.add_argument("--omega",     type=int, default=None, help="Override OMEGA (9999 = no decode eviction)")
parser.add_argument("--max_tokens", type=int, default=4500, help="Skip samples longer than this")
parser.add_argument("--diff",      nargs=2, metavar=("A","B"), help="Diff two saved traces")
args = parser.parse_args()

# torch is only needed for model runs, not for --diff
if not args.diff:
    import torch


# ──────────────────────────────────────────────────────────────────────────────
# DIFF MODE
# ──────────────────────────────────────────────────────────────────────────────
if args.diff:
    with open(args.diff[0]) as f: ta = json.load(f)
    with open(args.diff[1]) as f: tb = json.load(f)

    sa = {r["step"]: r for r in ta["steps"] if r.get("step", -2) >= 0}
    sb = {r["step"]: r for r in tb["steps"] if r.get("step", -2) >= 0}
    all_steps = sorted(set(sa) | set(sb))

    # TID_x = OUTPUT token generated at this step (first DIFF row = first divergence)
    hdr = (f"{'S':>4}  {'OUT_A':>8} {'OUT_B':>8} {'M':>4}  "
           f"{'POS_A':>6} {'POS_B':>6}  "
           f"{'PHB_A':>6} {'PHB_B':>6}  "
           f"{'TSR_A':>5} {'TSR_B':>5}  "
           f"{'QA_A':>5} {'QA_B':>5}  "
           f"{'GTC_A':>6} {'GTC_B':>6}")
    print(hdr)
    print("-" * len(hdr))

    first_div = None
    for step in all_steps:
        a = sa.get(step, {}); b = sb.get(step, {})
        ta_id = a.get("token_id","?"); tb_id = b.get("token_id","?")
        match = "OK" if ta_id == tb_id else "DIFF"
        if match == "DIFF" and first_div is None:
            first_div = step
        pos_a = (a.get("pos_ids") or ["?"])[0]
        pos_b = (b.get("pos_ids") or ["?"])[0]
        print(f"{step:>4}  {str(ta_id):>8} {str(tb_id):>8} {match:>4}  "  # OUT_A OUT_B
              f"{str(pos_a):>6} {str(pos_b):>6}  "
              f"{str(a.get('phys_before','?')):>6} {str(b.get('phys_before','?')):>6}  "
              f"{str(a.get('tokens_since','?')):>5} {str(b.get('tokens_since','?')):>5}  "
              f"{str(a.get('qcache_active','?')):>5} {str(b.get('qcache_active','?')):>5}  "
              f"{str(a.get('global_tc','?')):>6} {str(b.get('global_tc','?')):>6}")

    if first_div is None:
        print(f"\n>>> All {len(all_steps)} steps MATCH — no divergence found.")
    else:
        a = sa[first_div]; b = sb[first_div]
        print(f"\n>>> FIRST DIVERGENCE at step {first_div}")
        fields = ["token_id","token_str","input_token_id","pos_ids","cache_position",
                  "global_tc","global_tc_post",
                  "phys_before","phys_after","cache_seq_len",
                  "tokens_since","q_norm","k_norm","out_norm",
                  "qcache_active","q_windows"]
        for fld in fields:
            va = a.get(fld,"N/A"); vb = b.get(fld,"N/A")
            flag = "  " if va == vb else "!!"
            print(f"  {flag} {fld:22s}: A={va}  B={vb}")

        # Check one step BEFORE divergence for root cause
        if first_div > 0:
            prev_a = sa.get(first_div-1,{}); prev_b = sb.get(first_div-1,{})
            print(f"\n>>> Step {first_div-1} (one before divergence):")
            for fld in ["token_id","phys_before","phys_after","evicted_len",
                        "global_tc","global_tc_post","tokens_since","qcache_active"]:
                va = prev_a.get(fld,"N/A"); vb = prev_b.get(fld,"N/A")
                flag = "  " if va == vb else "!!"
                print(f"  {flag} {fld:22s}: A={va}  B={vb}")

    # Decoded text comparison
    print(f"\n>>> A generated: {repr(ta.get('generated_text','')[:200])}")
    print(f">>> B generated: {repr(tb.get('generated_text','')[:200])}")
    sys.exit(0)


# ──────────────────────────────────────────────────────────────────────────────
# SETUP
# ──────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, SCRIPT_DIR)

# Detect whether source files live under src/ or directly in SCRIPT_DIR.
_HAS_SRC = os.path.isdir(os.path.join(SCRIPT_DIR, "src"))

if args.mode == "current":
    if _HAS_SRC:
        import src.sticky_config as cfg
    else:
        import sticky_config as cfg
    if args.q_ratio is not None:
        cfg.Q_RATIO = args.q_ratio
        print(f"[override] Q_RATIO={args.q_ratio}", flush=True)
    if args.omega is not None:
        cfg.OMEGA = args.omega
        print(f"[override] OMEGA={args.omega}", flush=True)
    MODEL_PATH = cfg.MODEL_PATH
    DATA_DIR   = cfg.DATA_DIR
else:
    sys.path.insert(0, ORIG_ROOT)
    import sticky_config as orig_cfg
    if args.q_ratio is not None: orig_cfg.Q_RATIO = args.q_ratio
    if args.omega   is not None: orig_cfg.OMEGA   = args.omega
    MODEL_PATH = orig_cfg.MODEL_PATH
    DATA_DIR   = getattr(orig_cfg, "DATA_DIR",
                         "/kaggle/input/datasets/shaswatabhattacharya/longbench-12/1LongBenchData")

# ──────────────────────────────────────────────────────────────────────────────
# TOKENIZER + DATA
# ──────────────────────────────────────────────────────────────────────────────
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Locate lcc.jsonl
lcc_path = None
for root, dirs, files in os.walk(DATA_DIR):
    for fn in files:
        if fn.lower() in ("lcc.jsonl",):
            lcc_path = os.path.join(root, fn)
            break
    if lcc_path: break

if not lcc_path:
    print(f"ERROR: could not find lcc.jsonl under {DATA_DIR}", flush=True)
    sys.exit(1)

print(f"[data] LCC file: {lcc_path}", flush=True)
samples = []
with open(lcc_path) as f:
    for line in f:
        line = line.strip()
        if line:
            samples.append(json.loads(line))

# Filter (same logic as engine.py: skip >5000 tokens)
MAX_FILTER = 5000
MAX_PROMPT = args.max_tokens
_TEXT_FIELDS   = ("input", "prompt", "context", "text", "passage", "article")
_ANSWER_FIELDS = ("answers", "answer", "output", "label", "labels")

def _get_text(ex):
    for f in _TEXT_FIELDS:
        v = ex.get(f)
        if isinstance(v, str) and v.strip():
            return v
    # Last-resort: stringify first string-valued field found
    for v in ex.values():
        if isinstance(v, str) and v.strip():
            return v
    return ""

filtered = []
for ex in samples:
    text = _get_text(ex)
    ntok = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].shape[1]
    if ntok <= MAX_FILTER:
        filtered.append((ex, ntok))

if args.sample >= len(filtered):
    print(f"ERROR: only {len(filtered)} valid samples after filter, asked for index {args.sample}")
    sys.exit(1)

target_ex, prompt_ntok = filtered[args.sample]
prompt_text  = _get_text(target_ex)
ground_truth = next((target_ex[f] for f in _ANSWER_FIELDS if f in target_ex), "")

if not prompt_text:
    print(f"ERROR: sample {args.sample} has no recognised text field. Keys: {list(target_ex.keys())}")
    sys.exit(1)

print(f"[data] Using text field from keys: {list(target_ex.keys())}", flush=True)
input_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids

print(f"[data] Sample {args.sample}: {prompt_ntok} tokens  GT={repr(str(ground_truth)[:80])}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────────────────────────────────────
if args.mode == "current":
    if _HAS_SRC:
        from src.models.configuration_sticky_llama import LlamaConfig
        from src.models.sticky_llama_model import STICKYLlamaForCausalLM
    else:
        from configuration_sticky_llama import LlamaConfig
        from sticky_llama_model import STICKYLlamaForCausalLM

    model_config = LlamaConfig.from_pretrained(MODEL_PATH)
    if hasattr(model_config, "rope_scaling") and model_config.rope_scaling:
        rs = model_config.rope_scaling
        if "rope_type" in rs and "type" not in rs:
            rs["type"] = rs["rope_type"]
    model_config.rope_theta = getattr(model_config, "rope_theta", 500000.0)
    model_config.r_ratio   = cfg.R_RATIO
    model_config.p_ratio   = cfg.P_RATIO
    model_config.start_idx = cfg.S_IDX

    print(f"[model] Loading CURRENT from {MODEL_PATH}", flush=True)
    model = STICKYLlamaForCausalLM.from_pretrained(
        MODEL_PATH, config=model_config, torch_dtype=torch.bfloat16, device_map="auto"
    )

else:  # original
    cwd_backup = os.getcwd()
    os.chdir(ORIG_ROOT)
    from configuration_sticky_llama import LlamaConfig as OrigConfig
    from sticky_llama_model import STICKYLlamaForCausalLM as OrigModel

    model_config = OrigConfig.from_pretrained(MODEL_PATH)
    if hasattr(model_config, "rope_scaling") and model_config.rope_scaling:
        rs = model_config.rope_scaling
        if "rope_type" in rs and "type" not in rs:
            rs["type"] = rs["rope_type"]
    model_config.rope_theta = getattr(model_config, "rope_theta", 500000.0)
    model_config.r_ratio   = orig_cfg.R_RATIO
    model_config.p_ratio   = orig_cfg.P_RATIO
    model_config.start_idx = orig_cfg.S_IDX

    print(f"[model] Loading ORIGINAL from {MODEL_PATH}", flush=True)
    model = OrigModel.from_pretrained(
        MODEL_PATH, config=model_config, torch_dtype=torch.bfloat16, device_map="auto"
    )
    os.chdir(cwd_backup)

model.eval()
device = next(model.parameters()).device
input_ids = input_ids.to(device).long()
print(f"[model] Ready. device={device}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# HOOKS — capture per-step state via lightweight forward-hook on layer 0
# ──────────────────────────────────────────────────────────────────────────────
# We use a single dict that the hooks write to on every call.
# The main loop reads it after each decode step.

_per_step = {}   # overwritten each forward call by hooks below
_prefill_seen = [False]
_decode_idx   = [0]   # monotone counter for decode steps only

layer0_attn = model.model.layers[0].self_attn

# Hook on q_proj and k_proj to capture pre-RoPE norms
def _q_hook(_mod, _inp, out):
    _per_step["q_norm_pre"] = round(out.norm().item(), 5)

def _k_hook(_mod, _inp, out):
    _per_step["k_norm_pre"] = round(out.norm().item(), 5)

def _o_hook(_mod, _inp, out):
    _per_step["out_norm"] = round(out.norm().item(), 5)

layer0_attn.q_proj.register_forward_hook(_q_hook)
layer0_attn.k_proj.register_forward_hook(_k_hook)
layer0_attn.o_proj.register_forward_hook(_o_hook)

# Full forward hook captures position_ids, cache_position, and kv_cache state
_orig_l0_fwd = None

def _l0_forward_hook(module, args_in, kwargs_in, result):
    """post-forward hook on layer0_attn — runs AFTER the forward call"""
    hidden_states = args_in[0] if args_in else kwargs_in.get("hidden_states")
    if hidden_states is None:
        return

    bsz, q_len, _ = hidden_states.shape
    pkv = kwargs_in.get("past_key_value") or kwargs_in.get("past_key_values")
    is_decode = (q_len == 1) and (pkv is not None)

    if not is_decode:
        _prefill_seen[0] = True
        return

    if not _prefill_seen[0]:
        return

    # Position IDs
    pos = kwargs_in.get("position_ids")
    if pos is not None:
        _per_step["pos_ids"] = pos.detach().flatten().tolist()

    # cache_position (HF 4.57 passes this in kwargs)
    cp = kwargs_in.get("cache_position")
    if cp is not None and torch.is_tensor(cp):
        _per_step["cache_position"] = cp.detach().flatten().tolist()

    # KV cache state
    kvc = getattr(module, "kv_cache", None)
    if kvc is not None:
        gtc = getattr(kvc, "global_token_counter", None)
        if gtc is not None:
            _per_step["global_tc_post"] = int(gtc.item())
        em = getattr(kvc, "eviction_manager", None)
        if em is not None:
            _per_step["tokens_since"] = int(getattr(em, "tokens_since_last_review", -1))
        qm = getattr(kvc, "quantization_manager", None)
        if qm is not None:
            qck = getattr(qm, "q_cache_k_quant", None)
            _per_step["qcache_active"] = (qck is not None)
            qids = getattr(qm, "q_cache_ids", None)
            _per_step["q_windows"] = int(qids.shape[1]) if qids is not None else 0


# PyTorch register_forward_hook with kwargs support requires PyTorch >= 2.0
# and `with_kwargs=True`. Fall back gracefully if not available.
try:
    layer0_attn.register_forward_hook(_l0_forward_hook, with_kwargs=True)
except TypeError:
    # Older PyTorch: use a simpler hook without kwargs
    def _l0_fwd_simple(_mod, _inp, _out):
        _prefill_seen[0] = True   # best-effort; won't capture per-step details
    layer0_attn.register_forward_hook(_l0_fwd_simple)
    print("[warn] PyTorch <2.0 — per-step position/cache hooks disabled; "
          "phys_before/pos_ids will be N/A", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# PRE-FORWARD HOOK to capture phys_before BEFORE kv_cache increments things
# We wrap prepare_inputs_for_generation to also read physical cache length.
# ──────────────────────────────────────────────────────────────────────────────
_orig_prep = model.prepare_inputs_for_generation.__func__ if hasattr(
    model.prepare_inputs_for_generation, "__func__") else None

def _traced_prep(self_model, input_ids_arg, past_key_values=None, **kwargs):
    # Call the real prepare_inputs_for_generation
    if _orig_prep:
        result = _orig_prep(self_model, input_ids_arg,
                            past_key_values=past_key_values, **kwargs)
    else:
        result = type(self_model).prepare_inputs_for_generation(
            self_model, input_ids_arg, past_key_values=past_key_values, **kwargs)

    if past_key_values is not None:
        # Capture physical cache length BEFORE the forward call (i.e., before
        # the new token is appended).  We read from the cache object.
        phys = None
        try:
            pkv = past_key_values
            if hasattr(pkv, "layers") and len(pkv.layers) > 0:
                layer = pkv.layers[0]
                k = getattr(layer, "keys", None)
                if k is not None and k.numel() > 0:
                    phys = int(k.shape[-2])
            elif hasattr(pkv, "key_cache") and len(pkv.key_cache) > 0:
                k = pkv.key_cache[0]
                if k.numel() > 0:
                    phys = int(k.shape[-2])
            elif isinstance(pkv, tuple) and len(pkv) > 0:
                layer_kv = pkv[0]
                if isinstance(layer_kv, tuple) and len(layer_kv) >= 2:
                    phys = int(layer_kv[0].shape[-2])
                elif torch.is_tensor(layer_kv):
                    phys = int(layer_kv.shape[-2])
        except Exception:
            pass
        _per_step["phys_before_prep"] = phys

        # Also capture global_tc at this point (BEFORE forward increments it)
        try:
            l0_attn = self_model.model.layers[0].self_attn
            gtc = int(l0_attn.kv_cache.global_token_counter.item())
            _per_step["global_tc_prep"] = gtc
        except Exception:
            pass

        # Capture cache_position from model_inputs (after super() computed it, before our override)
        cp = result.get("cache_position")
        if cp is not None and torch.is_tensor(cp):
            _per_step["cache_pos_from_prep"] = cp.detach().flatten().tolist()
        pid = result.get("position_ids")
        if pid is not None and torch.is_tensor(pid):
            _per_step["pos_ids_from_prep"] = pid.detach().flatten().tolist()

        # What does DynamicCache.get_seq_length() return?
        # This is what super().prepare_inputs_for_generation() used to compute cache_position
        # BEFORE our override.  If this is wrong, HF's generated cache_position was wrong
        # (even though we override it afterwards).
        try:
            seq_len_reported = past_key_values.get_seq_length()
            _per_step["cache_get_seq_length"] = seq_len_reported
        except Exception:
            try:
                seq_len_reported = past_key_values.get_seq_length(0)
                _per_step["cache_get_seq_length"] = seq_len_reported
            except Exception:
                pass

    return result

import types
model.prepare_inputs_for_generation = types.MethodType(_traced_prep, model)


# ──────────────────────────────────────────────────────────────────────────────
# GENERATION LOOP  (manual, one token at a time)
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[gen] Prefilling {input_ids.shape[1]} tokens...", flush=True)

for layer in model.model.layers:
    attn = layer.self_attn
    if hasattr(attn, "_clean_cache"):
        attn._clean_cache()
    if hasattr(attn, "_dbg_count"):
        attn._dbg_count = 0
if hasattr(model, "_dbg_prepare_count"):
    model._dbg_prepare_count = 0

_TRACE = []

with torch.inference_mode():
    # Prefill
    _per_step.clear()
    out = model(input_ids=input_ids, use_cache=True, past_key_values=None, return_dict=True)
    pkv = out.past_key_values
    last_logits = out.logits[:, -1, :]
    next_tok = int(last_logits.argmax(dim=-1).item())
    all_ids = torch.cat([input_ids, torch.tensor([[next_tok]], device=device, dtype=torch.long)], dim=1)

    prefill_out_tok = next_tok   # first generated token (from prefill logits)
    print(f"[gen] Prefill done. First token: {next_tok} ({repr(tokenizer.decode([next_tok]))})", flush=True)

    # Decode loop
    # Terminology:
    #   input_tok  = token consumed as input at this decode step
    #   output_tok = token generated (output of logits) at this decode step
    #              = will become input_tok for the NEXT step
    # rec["token_id"] = output_tok, so that when two traces are diffed,
    # the first step where token_id differs is the EXACT step where outputs diverge.
    for step in range(args.steps):
        _per_step.clear()
        input_tok = next_tok  # the token this step processes as input

        model_inputs = model.prepare_inputs_for_generation(
            all_ids,
            past_key_values=pkv,
            attention_mask=torch.ones(1, all_ids.shape[1], device=device),
        )
        model_inputs["use_cache"] = True
        out = model(**model_inputs, return_dict=True)
        pkv = out.past_key_values
        new_logits = out.logits[:, -1, :]
        output_tok = int(new_logits.argmax(dim=-1).item())

        all_ids = torch.cat([all_ids, torch.tensor([[output_tok]], device=device, dtype=torch.long)], dim=1)

        rec = {
            "step":            step,
            "mode":            args.mode,
            # OUTPUT: token generated by this decode step (use for diff comparison)
            "token_id":        output_tok,
            "token_str":       tokenizer.decode([output_tok]),
            # INPUT: token that was fed into this decode step
            "input_token_id":  input_tok,
            "input_token_str": tokenizer.decode([input_tok]),
            # position info  (from _traced_prep hook)
            "pos_ids":         _per_step.get("pos_ids_from_prep"),
            "cache_position":  _per_step.get("cache_pos_from_prep"),
            # physical cache length BEFORE this step's forward
            "phys_before":     _per_step.get("phys_before_prep"),
            # what DynamicCache.get_seq_length() returned (used by HF to compute cache_position)
            "cache_seq_len":   _per_step.get("cache_get_seq_length"),
            # global_tc BEFORE the kv_cache increment
            "global_tc":       _per_step.get("global_tc_prep"),
            # global_tc AFTER the kv_cache increment (from post-forward hook)
            "global_tc_post":  _per_step.get("global_tc_post"),
            # eviction state (after kv_cache call)
            "tokens_since":    _per_step.get("tokens_since"),
            "qcache_active":   _per_step.get("qcache_active"),
            "q_windows":       _per_step.get("q_windows"),
            # norms at layer 0 (pre-RoPE, from projection output hooks)
            "q_norm":          _per_step.get("q_norm_pre"),
            "k_norm":          _per_step.get("k_norm_pre"),
            "out_norm":        _per_step.get("out_norm"),
        }

        # Physical cache length AFTER this step: read directly from pkv
        phys_after = None
        try:
            if hasattr(pkv, "layers") and len(pkv.layers) > 0:
                layer = pkv.layers[0]
                k = getattr(layer, "keys", None)
                if k is not None and k.numel() > 0:
                    phys_after = int(k.shape[-2])
            elif hasattr(pkv, "key_cache") and len(pkv.key_cache) > 0:
                k = pkv.key_cache[0]
                if k.numel() > 0:
                    phys_after = int(k.shape[-2])
            elif isinstance(pkv, tuple) and len(pkv) > 0:
                lkv = pkv[0]
                if isinstance(lkv, tuple):
                    phys_after = int(lkv[0].shape[-2])
        except Exception:
            pass
        rec["phys_after"] = phys_after

        _TRACE.append(rec)
        next_tok = output_tok  # becomes input for the next iteration

        if step < 20 or step % 5 == 0:
            print(f"  step {step:>3}: in={input_tok:>7}→out={output_tok:>7} "
                  f"({repr(tokenizer.decode([output_tok]))[:10]:>12})  "
                  f"pos={str(rec['pos_ids']):>10}  phys={str(rec['phys_before']):>5}→{str(rec['phys_after']):>5}  "
                  f"tslr={str(rec['tokens_since']):>3}  "
                  f"qact={str(rec['qcache_active']):>5}  "
                  f"gtc={str(rec['global_tc']):>6}", flush=True)

# gen_tokens: full sequence of generated tokens (prefill output + all decode outputs)
gen_tokens = [prefill_out_tok] + [r["token_id"] for r in _TRACE]
gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
print(f"\n[gen] Done. Generated ({len(gen_tokens)} tokens):\n  {repr(gen_text[:300])}", flush=True)

# Detect repetition loops
rep_start = None
for i in range(1, len(gen_tokens)):
    if gen_tokens[i] == gen_tokens[i-1] and rep_start is None:
        rep_start = i
        break
if rep_start is not None:
    print(f"\n[warn] REPETITION LOOP starts at generated token {rep_start} "
          f"(decode step {rep_start-1}): "
          f"tok={gen_tokens[rep_start]} ({repr(tokenizer.decode([gen_tokens[rep_start]]))})", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'S':>4}  {'TOK_ID':>8}  {'TOK_STR':>13}  "
      f"{'POS':>6}  {'PHY_B':>6} {'PHY_A':>6}  "
      f"{'TSR':>4}  {'QACT':>5}  "
      f"{'GTC':>6}  {'GTC+':>6}  "
      f"{'Q_NRM':>8}  {'K_NRM':>8}")
print("-" * 110)
for rec in _TRACE:
    pos = (rec.get("pos_ids") or ["?"])[0]
    print(f"{rec['step']:>4}  {str(rec['token_id']):>8}  "
          f"{repr(rec['token_str'])[:12]:>13}  "
          f"{str(pos):>6}  "
          f"{str(rec.get('phys_before','?')):>6} {str(rec.get('phys_after','?')):>6}  "
          f"{str(rec.get('tokens_since','?')):>4}  "
          f"{str(rec.get('qcache_active','?')):>5}  "
          f"{str(rec.get('global_tc','?')):>6}  "
          f"{str(rec.get('global_tc_post','?')):>6}  "
          f"{str(rec.get('q_norm','?')):>8}  "
          f"{str(rec.get('k_norm','?')):>8}")


# ──────────────────────────────────────────────────────────────────────────────
# SAVE
# ──────────────────────────────────────────────────────────────────────────────
out_path = args.out
if not out_path:
    suffix = ""
    if args.q_ratio  is not None: suffix += f"_q{args.q_ratio}"
    if args.omega    is not None: suffix += f"_omega{args.omega}"
    out_path = f"trace_{args.mode}{suffix}.json"

trace_data = {
    "mode":            args.mode,
    "sample":          args.sample,
    "prompt_tokens":   prompt_ntok,
    "q_ratio_override": args.q_ratio,
    "omega_override":   args.omega,
    "generated_tokens": gen_tokens,
    "generated_text":   gen_text,
    "steps":            _TRACE,
}
with open(out_path, "w") as f:
    json.dump(trace_data, f, indent=2, default=str)
print(f"\n[save] Trace saved to {out_path}", flush=True)
