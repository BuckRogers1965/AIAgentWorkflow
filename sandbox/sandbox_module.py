# --- OPTIONAL LIBRARY LOADING FOR CODEJAIL ---
try:
    from RestrictedPython import compile_restricted
    from RestrictedPython.Guards import safe_globals
    from RestrictedPython import safe_builtins, compile_restricted
    from RestrictedPython.Guards import safe_globals as rp_safe_globals
    from RestrictedPython.Eval import default_guarded_getiter
    from RestrictedPython.Guards import guarded_unpack_sequence
    from RestrictedPython import safe_builtins, compile_restricted
    from RestrictedPython.Eval import (
        default_guarded_getattr,
        default_guarded_getitem,
        default_guarded_getiter
    )
    from RestrictedPython.Guards import (
        guarded_iter_unpack_sequence,
        guarded_unpack_sequence
    )

    # Some versions use _apply_ internally:
    try:
        from RestrictedPython.Guards import guarded_apply
    except ImportError:
        guarded_apply = lambda func, *a, **kw: func(*a, **kw)

    RESTRICTEDPYTHON_AVAILABLE = True
    print("INFO: RestrictedPython library found. Sandboxed execution is available.")

except ImportError as e:
    RESTRICTEDPYTHON_AVAILABLE = False
    print("WARNING: RestrictedPython library not found. Sandboxed execution will be disabled.")
except Exception as e:
    # In case the failure is *not* an ImportError
    RESTRICTEDPYTHON_AVAILABLE = False
    print("ERROR while loading RestrictedPython or guards:", e)
# --- END OF OPTIONAL LOADING ---

import base64
import json
import logging
import argparse # Import argparse

def add_sandbox_args(parser):
    """
    Adds sandbox-specific command-line arguments to the given argparse parser.
    """
    parser.add_argument('--jail_dir', help='Path to the secure sandbox jail directory.')
    parser.add_argument('--jail_user', default='agent_worker',
                    help='The low-privilege user for the jail.')

def _execute_jailed(function_name: str, function_def: str,
                    updated_step_params: dict, jail_config: dict) -> tuple:
    jail_script = f'''
# Future-proof type hints evaluation
from __future__ import annotations
{function_def}
params = {repr(updated_step_params)}
try:
    result, status = {function_name}(**params)
    if isinstance(result, bytes):
        res_data, res_type = base64.b64encode(result).decode('utf-8'), 'bytes'
    else:
        res_data, res_type = json.dumps(result), 'json'
    output = {{'success': True, 'result_data': res_data, 'result_type': res_type, 'status': status}}
except Exception as e:
    output = {{'success': False, 'status': {{'status': {{'value': 1, 'reason': f'Jailed function error: {{repr(e)}}'}}}}}}
jail_output = output
'''

    logging.debug("*jail*script*:\n%s", jail_script)
    compiled_code = compile_restricted(jail_script, '<string>', 'exec')

    restricted_globals = jail_config['safe_globals'].copy()
    restricted_globals.update(jail_config['allowed_modules'])

    exec(compiled_code, restricted_globals)
    jail_output = restricted_globals['jail_output']

    if jail_output['success']:
        if jail_output['result_type'] == 'bytes':
            result = base64.b64decode(jail_output['result_data'])
        else:
            result = json.loads(jail_output['result_data'])
        status = jail_output['status']
    else:
        result, status = b'', jail_output['status']
    return result, status

def jailed_open(path, mode='r', *args, **kwargs):
    import builtins
    import os
    from pathlib import Path

    # True jail root (on macOS /tmp resolves to /private/tmp)
    base_dir = Path("/tmp").resolve()

    raw = Path(path)

    # Map absolute paths into the jail by stripping the leading slash
    # /Users/jrogers/hello -> /tmp/Users/jrogers/hello
    rel = raw if not raw.is_absolute() else Path(*raw.parts[1:])

    # Compose under jail root, then resolve to collapse any ".." or symlinks
    target = (base_dir / rel).resolve()

    # Enforce containment after resolution (prevents symlink traversal)
    try:
        # Python 3.9+: clean containment check
        target.relative_to(base_dir)
    except Exception:
        if not str(target).startswith(str(base_dir)):
            raise PermissionError(f"Access denied outside jail: {target}")

    # Read-only policy
    if any(flag in mode for flag in ('w', 'a', '+', 'x')):
        raise PermissionError(f"Write modes not allowed: {mode}")

    # Optional: debug
    print(f"[JAILED OPEN] cwd={{os.getcwd()}} base_dir={{base_dir}} raw={{raw}} -> target={{target}} mode={{mode}}")

    return builtins.open(target, mode, *args, **kwargs)

def setup_jail_config(args):
    if not hasattr(args, 'jail_dir') or args.jail_dir is None:
        return None
    if not hasattr(args, 'jail_user') or not args.jail_user:
        print("ERROR: --jail_dir was provided, but --jail_user is missing or empty.", file=sys.stderr)
        return None
    try:
        test_code = "result = 1 + 1"
        compiled_code = compile_restricted(test_code, '<string>', 'exec')
        test_globals = dict(safe_builtins)
        exec(compiled_code, test_globals)
        if test_globals.get('result') != 2:
            print("ERROR: RestrictedPython test failed", file=sys.stderr)
            return None
    except ImportError:
        print("ERROR: RestrictedPython not available. Install with: pip install RestrictedPython", file=sys.stderr)
        return None
    except Exception as e:
        print(f"ERROR: RestrictedPython test failed: {e}", file=sys.stderr)
        return None
    allowed_modules = {
        'json': __import__('json'), 'base64': __import__('base64'), 'sys': __import__('sys'),
        'math': __import__('math'), 'datetime': __import__('datetime'), 're': __import__('re'), }
    safe_globals = dict(safe_builtins) 
    safe_globals['open'] = jailed_open
    safe_globals.update({
        # Core guards/hooks that RestrictedPython might inject:
        '_getattr_': default_guarded_getattr,
        '_getitem_': default_guarded_getitem,
        '_getiter_': default_guarded_getiter,
        '_unpack_sequence_': guarded_unpack_sequence,
        '_iter_unpack_sequence_': guarded_iter_unpack_sequence,
        '_apply_': guarded_apply,
        # Core builtins/types to avoid NameError on annotations etc:
        'list': list, 'dict': dict, 'str': str, 'int': int,
        'float': float, 'bool': bool,'bytes': bytes,
        # Allow imports for already-injected modules:
        '__builtins__': {**safe_builtins, '__import__': __import__}
    })
    return {
        "enabled": True, "type": "restricted_python", "path": args.jail_dir, "user": args.jail_user,
        "allowed_modules": allowed_modules, "safe_globals": safe_globals,
        "restrictions": {
            "allow_imports": True,  # loosened so modules can lazy-load submodules
            "allow_file_access": False,
            "allow_network": False, } }
