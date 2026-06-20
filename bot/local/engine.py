""" Mortal Engine for 4p game"""
import torch
import numpy as np
from torch.distributions import Normal, Categorical
from bot.local.model import Brain, DQN

class MortalEngine:
    """ Mortal Engine for local Bot 4p"""
    def __init__(
        self,
        brain,
        dqn,
        is_oracle,
        version,
        device = None,
        stochastic_latent = False,
        enable_amp = False,
        enable_quick_eval = True,
        enable_rule_based_agari_guard = False,
        name = 'NoName',
        boltzmann_epsilon = 0,
        boltzmann_temp = 1,
        top_p = 1,
    ):
        self.engine_type = 'mortal'
        self.device = device or torch.device('cpu')
        assert isinstance(self.device, torch.device)
        self.brain = brain.to(self.device).eval()
        self.dqn = dqn.to(self.device).eval()
        self.is_oracle = is_oracle
        self.version = version
        self.stochastic_latent = stochastic_latent

        self.enable_amp = enable_amp
        self.enable_quick_eval = enable_quick_eval
        self.enable_rule_based_agari_guard = enable_rule_based_agari_guard
        self.name = name

        self.boltzmann_epsilon = boltzmann_epsilon
        self.boltzmann_temp = boltzmann_temp
        self.top_p = top_p

    def react_batch(self, obs, masks, invisible_obs):
        with (
            torch.autocast(self.device.type, enabled=self.enable_amp),
            torch.no_grad(),
        ):
            return self._react_batch(obs, masks, invisible_obs)

    def _react_batch(self, obs, masks, invisible_obs):
        # --- Debug logging: inspect incoming batch items to find problematic element ---
        try:
            try:
                from common.log_helper import LOGGER
            except Exception:
                LOGGER = None
            import numpy as _np

            # helper to format small repr
            def _fmt(x):
                try:
                    t = type(x)
                    is_nd = isinstance(x, _np.ndarray)
                    shape = getattr(x, "shape", None)
                    dtype = getattr(getattr(x, "dtype", None), "name", None)
                    # small repr (limit length)
                    r = repr(x)
                    if len(r) > 200:
                        r = r[:200] + "...(truncated)"
                    return f"type={t}, is_nd={is_nd}, shape={shape}, dtype={dtype}, repr={r}"
                except Exception as e:
                    return f"failed_fmt:{e}"

            if LOGGER is not None:
                LOGGER.debug("engine._react_batch incoming sizes: obs=%d, masks=%d, invisible=%s",
                             len(obs) if hasattr(obs, "__len__") else -1,
                             len(masks) if hasattr(masks, "__len__") else -1,
                             ("yes" if invisible_obs is not None else "no"))
                # log obs items
                for i, it in enumerate(obs):
                    LOGGER.debug("engine._react_batch obs[%d]: %s", i, _fmt(it))
                for i, it in enumerate(masks):
                    LOGGER.debug("engine._react_batch masks[%d]: %s", i, _fmt(it))
                if invisible_obs is not None:
                    for i, it in enumerate(invisible_obs):
                        LOGGER.debug("engine._react_batch invisible[%d]: %s", i, _fmt(it))
            else:
                # fallback prints
                print("engine._react_batch incoming sizes:", len(obs) if hasattr(obs, "__len__") else -1,
                      len(masks) if hasattr(masks, "__len__") else -1,
                      ("yes" if invisible_obs is not None else "no"))
        except Exception:
            # ensure debug instrumentation never breaks runtime
            try:
                if LOGGER is not None:
                    LOGGER.debug("engine._react_batch: debug logging failed")
            except Exception:
                pass
        # --- end debug logging ---

        # Safely convert obs/masks/invisible_obs into numeric numpy arrays before sending to torch.
        # This handles cases where an element may be a numpy.float32 scalar or uneven shapes.
        import numpy as _np

        def _safe_stack(list_of_items, dtype=None, bool_mask=False):
                # Convert each item to numpy array first
                arrs = []
                for item in list_of_items:
                    try:
                        a = _np.asarray(item)
                    except Exception:
                        a = _np.array(item, dtype=object)
                    arrs.append(a)

                # If all elements are scalar-like, return 1D numeric/bool array
                scalar_flags = [(_np.isscalar(a) or (isinstance(a, _np.ndarray) and getattr(a, 'shape', ()) == ())) for a in arrs]
                if all(scalar_flags):
                    vals = []
                    for a in arrs:
                        try:
                            v = float(a)
                        except Exception:
                            # fallback to 0.0 for numeric or False for mask
                            v = 0.0
                        vals.append(v)
                    if bool_mask:
                        return _np.array([bool(x) for x in vals], dtype=_np.bool_)
                    else:
                        target_dtype = _np.float32 if dtype is None else dtype
                        return _np.array(vals, dtype=target_dtype)

                # If shapes are identical, try a direct stack
                shapes = [getattr(a, 'shape', ()) for a in arrs]
                if len(set(shapes)) == 1:
                    try:
                        stacked = _np.stack(arrs, axis=0)
                        if dtype is not None and stacked.dtype != dtype:
                            stacked = stacked.astype(dtype)
                        if bool_mask:
                            stacked = stacked.astype(_np.bool_)
                        return stacked
                    except Exception:
                        pass

                # Handle common ragged case: 1D arrays of varying lengths -> pad to max length
                if all(isinstance(a, _np.ndarray) and a.ndim <= 1 for a in arrs):
                    maxlen = max((0 if a.shape == () else a.shape[0]) for a in arrs)
                    coerced = []
                    for a in arrs:
                        if getattr(a, 'shape', ()) == ():
                            v = _np.asarray([a], dtype=_np.float32)
                        else:
                            v = _np.asarray(a, dtype=_np.float32)
                        if v.shape[0] < maxlen:
                            pad = _np.zeros((maxlen - v.shape[0],), dtype=_np.float32)
                            v = _np.concatenate([v, pad])
                        elif v.shape[0] > maxlen:
                            v = v[:maxlen]
                        coerced.append(v)
                    stacked = _np.stack(coerced, axis=0)
                    if dtype is not None and stacked.dtype != dtype:
                        stacked = stacked.astype(dtype)
                    if bool_mask:
                        stacked = stacked.astype(_np.bool_)
                    return stacked

                # Fallback: try element-wise conversion to requested dtype then stack
                try:
                    target_dtype = _np.bool_ if bool_mask else (_np.float32 if dtype is None else dtype)
                    coerced = [_np.asarray(a, dtype=target_dtype) for a in arrs]
                    stacked = _np.stack(coerced, axis=0)
                    return stacked
                except Exception:
                    # Last resort: reduce to 1D numeric list (best-effort)
                    flat = []
                    for a in arrs:
                        try:
                            flat.append(float(a))
                        except Exception:
                            flat.append(0.0)
                    if bool_mask:
                        return _np.array([bool(x) for x in flat], dtype=_np.bool_)
                    else:
                        return _np.array(flat, dtype=_np.float32)

        # Convert obs -> torch tensor
        obs_np = _safe_stack(obs, dtype=_np.float32, bool_mask=False)
        # Convert masks -> boolean tensor
        masks_np = _safe_stack(masks, dtype=_np.bool_, bool_mask=True)

        invisible_np = None
        if self.is_oracle and invisible_obs is not None:
            try:
                invisible_np = _safe_stack(invisible_obs, dtype=_np.float32, bool_mask=False)
            except Exception:
                invisible_np = None

        # extra debug before converting to torch tensors
        try:
            LOGGER.debug("engine._react_batch post-stack obs_np: type=%s, shape=%s, dtype=%s, is_object=%s",
                         type(obs_np), getattr(obs_np, 'shape', None), getattr(getattr(obs_np, 'dtype', None), 'name', None),
                         getattr(obs_np, 'dtype', None) == _np.dtype('object'))
            LOGGER.debug("engine._react_batch post-stack masks_np: type=%s, shape=%s, dtype=%s, is_object=%s",
                         type(masks_np), getattr(masks_np, 'shape', None), getattr(getattr(masks_np, 'dtype', None), 'name', None),
                         getattr(masks_np, 'dtype', None) == _np.dtype('object'))
        except Exception:
            LOGGER.debug("engine._react_batch: failed to log post-stack shapes")

        try:
            # Ensure contiguous arrays and explicit dtypes before creating tensors
            obs_np_c = _np.ascontiguousarray(_np.asarray(obs_np, dtype=_np.float32))
            masks_np_c = _np.ascontiguousarray(_np.asarray(masks_np, dtype=_np.bool_))
            try:
                obs = torch.from_numpy(obs_np_c).to(self.device)
                masks = torch.from_numpy(masks_np_c).to(self.device)
            except Exception as e_inner:
                # fallback when torch.from_numpy not available or fails (e.g., "NumPy is not available")
                try:
                    if LOGGER is not None:
                        LOGGER.warning("engine._react_batch.from_numpy failed, falling back to torch.tensor(list): %s", e_inner)
                    else:
                        print("engine._react_batch.from_numpy failed, falling back:", e_inner)
                except Exception:
                    pass
                # Convert to Python lists and create tensors (this copies data)
                obs = torch.tensor(obs_np_c.tolist(), dtype=torch.float32, device=self.device)
                masks = torch.tensor(masks_np_c.tolist(), dtype=torch.bool, device=self.device)
        except Exception as e:
            # Log truncated reprs for diagnostics and re-raise
            try:
                ro = repr(obs_np)
                if len(ro) > 500:
                    ro = ro[:500] + '...(truncated)'
                rm = repr(masks_np)
                if len(rm) > 500:
                    rm = rm[:500] + '...(truncated)'
                if LOGGER is not None:
                    LOGGER.error("engine._react_batch tensor conversion failed: %s", e)
                    LOGGER.debug("obs_np repr: %s", ro)
                    LOGGER.debug("masks_np repr: %s", rm)
                else:
                    print("engine._react_batch tensor conversion failed:", e)
                    print("obs_np repr:", ro)
                    print("masks_np repr:", rm)
            except Exception:
                if LOGGER is not None:
                    LOGGER.debug("engine._react_batch: failed to capture reprs for failing conversion")
            raise
        # invisible_obs will be converted to tensor as needed by model path
        if invisible_np is not None:
            invisible_obs = torch.as_tensor(invisible_np, device=self.device)
        else:
            invisible_obs = None

        batch_size = obs.shape[0]

        match self.version:
            case 1:
                mu, logsig = self.brain(obs, invisible_obs)
                if self.stochastic_latent:
                    latent = Normal(mu, logsig.exp() + 1e-6).sample()
                else:
                    latent = mu
                q_out = self.dqn(latent, masks)
            case 2 | 3 | 4:
                phi = self.brain(obs)
                q_out = self.dqn(phi, masks)

        if self.boltzmann_epsilon > 0:
            is_greedy = torch.full((batch_size,), 1-self.boltzmann_epsilon, device=self.device).bernoulli().to(torch.bool)
            logits = (q_out / self.boltzmann_temp).masked_fill(~masks, -torch.inf)
            sampled = sample_top_p(logits, self.top_p)
            actions = torch.where(is_greedy, q_out.argmax(-1), sampled)
        else:
            is_greedy = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            actions = q_out.argmax(-1)

        # Convert outputs to pure Python native types to avoid numpy types leaking out
        try:
            actions_list = [int(x) for x in actions.tolist()]
        except Exception:
            actions_list = actions.tolist()

        try:
            q_out_np = q_out.detach().cpu().numpy()
            q_out_list = [[float(x) for x in row] for row in q_out_np]
        except Exception:
            q_out_list = q_out.tolist()

        try:
            masks_np_out = masks.detach().cpu().numpy()
            masks_list = [[bool(x) for x in row] for row in masks_np_out]
        except Exception:
            masks_list = masks.tolist()

        try:
            is_greedy_list = [bool(x) for x in is_greedy.tolist()]
        except Exception:
            is_greedy_list = is_greedy.tolist()

        return actions_list, q_out_list, masks_list, is_greedy_list

def sample_top_p(logits, p):
    if p >= 1:
        return Categorical(logits=logits).sample()
    if p <= 0:
        return logits.argmax(-1)
    probs = logits.softmax(-1)
    probs_sort, probs_idx = probs.sort(-1, descending=True)
    probs_sum = probs_sort.cumsum(-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.
    sampled = probs_idx.gather(-1, probs_sort.multinomial(1)).squeeze(-1)
    return sampled

def get_engine(model_file:str) -> MortalEngine:
    """ Create and return Mortal engine object
    params:
        model_file(str): Mortal model file path"""
    # check if GPU is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Get the path of control_state_file = current directory / control_state_file
    state = torch.load(model_file, map_location=device)

    mortal = Brain(version=state['config']['control']['version'],
        conv_channels=state['config']['resnet']['conv_channels'],
        num_blocks=state['config']['resnet']['num_blocks']).eval()
    dqn = DQN(version=state['config']['control']['version']).eval()
    mortal.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])

    engine = MortalEngine(
        mortal,
        dqn,
        is_oracle = False,
        device = device,
        enable_amp = False,
        enable_quick_eval = False,
        enable_rule_based_agari_guard = False,
        name = 'mortal',
        version = state['config']['control']['version'],
    )

    return engine
