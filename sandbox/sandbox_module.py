#!/usr/bin/env python3
"""
Modular sandboxing system for dynamic workflow agents.
Provides secure code execution using RestrictedPython.
"""

import sys
import json
import base64
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

class SandboxError(Exception):
    """Custom exception for sandbox-related errors"""
    pass

class SandboxManager:
    """Manages sandboxed code execution with different backend strategies"""
    
    def __init__(self):
        self.available_backends = {}
        self._discover_backends()
    
    def _discover_backends(self):
        """Discover and initialize available sandboxing backends"""
        # RestrictedPython backend
        try:
            backend = RestrictedPythonBackend()
            if backend.is_available():
                self.available_backends['restricted_python'] = backend
                logging.info("RestrictedPython sandbox backend available")
        except Exception as e:
            logging.warning(f"RestrictedPython backend failed to initialize: {e}")
        
        # Could add more backends here:
        # - Docker backend
        # - chroot backend  
        # - subprocess backend with ulimits
        
        if not self.available_backends:
            logging.warning("No sandbox backends available - execution will be unsandboxed")
    
    def get_backend(self, backend_type: str) -> Optional['SandboxBackend']:
        """Get a specific sandbox backend"""
        return self.available_backends.get(backend_type)
    
    def list_backends(self) -> list:
        """List all available sandbox backends"""
        return list(self.available_backends.keys())
    
    def execute_sandboxed(self, function_name: str, function_def: str, 
                         params: Dict[str, Any], config: Dict[str, Any]) -> Tuple[Any, Dict]:
        """Execute code using the configured sandbox backend"""
        backend_type = config.get('type', 'restricted_python')
        backend = self.get_backend(backend_type)
        
        if not backend:
            raise SandboxError(f"Sandbox backend '{backend_type}' not available")
        
        return backend.execute(function_name, function_def, params, config)

class SandboxBackend:
    """Base class for sandbox backends"""
    
    def is_available(self) -> bool:
        """Check if this backend is available"""
        raise NotImplementedError
    
    def execute(self, function_name: str, function_def: str, 
                params: Dict[str, Any], config: Dict[str, Any]) -> Tuple[Any, Dict]:
        """Execute code in sandbox"""
        raise NotImplementedError

class RestrictedPythonBackend(SandboxBackend):
    """RestrictedPython-based sandboxing backend"""
    
    def __init__(self):
        self._rp_available = False
        self._safe_globals = None
        self._setup_restricted_python()
    
    def _setup_restricted_python(self):
        """Initialize RestrictedPython components"""
        try:
            from RestrictedPython import compile_restricted, safe_builtins
            from RestrictedPython.Guards import safe_globals as rp_safe_globals
            from RestrictedPython.Eval import default_guarded_getiter, default_guarded_getattr, default_guarded_getitem
            from RestrictedPython.Guards import guarded_iter_unpack_sequence, guarded_unpack_sequence
            
            # Handle version differences
            try:
                from RestrictedPython.Guards import guarded_apply
            except ImportError:
                guarded_apply = lambda func, *a, **kw: func(*a, **kw)
            
            self.compile_restricted = compile_restricted
            self.safe_builtins = safe_builtins
            self.guarded_apply = guarded_apply
            self.default_guarded_getattr = default_guarded_getattr
            self.default_guarded_getitem = default_guarded_getitem
            self.default_guarded_getiter = default_guarded_getiter
            self.guarded_unpack_sequence = guarded_unpack_sequence
            self.guarded_iter_unpack_sequence = guarded_iter_unpack_sequence
            
            self._rp_available = True
            
        except ImportError as e:
            logging.warning(f"RestrictedPython not available: {e}")
        except Exception as e:
            logging.error(f"Error setting up RestrictedPython: {e}")
    
    def is_available(self) -> bool:
        return self._rp_available
    
    def _create_capability_aware_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create security overrides that check agent capabilities"""
        jail_path = config.get('path', '/tmp')
        agent_name = config.get('agent_name', 'unknown')
        
        overrides = {}
        
        # === FILE SYSTEM OVERRIDES ===
        def capability_aware_open(path, mode='r', *args, **kwargs):
            context = {'path': str(path), 'mode': mode}
            
            if 'r' in mode or mode == 'rb':
                if not capability_manager.check_capability(agent_name, 'file_read', context):
                    raise PermissionError(f"Agent '{agent_name}' denied file read access to {path}")
            
            if any(flag in mode for flag in ('w', 'a', '+', 'x')):
                if not capability_manager.check_capability(agent_name, 'file_write', context):
                    raise PermissionError(f"Agent '{agent_name}' denied file write access to {path}")
            
            # Proceed with jailed file access
            import builtins
            from pathlib import Path
            
            base_dir = Path(jail_path).resolve()
            raw = Path(path)
            rel = raw if not raw.is_absolute() else Path(*raw.parts[1:])
            target = (base_dir / rel).resolve()
            
            try:
                target.relative_to(base_dir)
            except Exception:
                if not str(target).startswith(str(base_dir)):
                    raise PermissionError(f"Access denied outside jail: {target}")
            
            return builtins.open(target, mode, *args, **kwargs)
        
        overrides['open'] = capability_aware_open
        
        # === NETWORK OVERRIDES ===
        def capability_aware_request(method, url, **kwargs):
            context = {'url': url, 'method': method}
            
            if not capability_manager.check_capability(agent_name, 'network_http', context):
                raise PermissionError(f"Agent '{agent_name}' denied network access to {url}")
            
            # If we get here, the request is allowed
            import requests
            return requests.request(method, url, **kwargs)
        
        def make_requests_module():
            """Create a requests module with capability checking"""
            return type('CapabilityAwareRequests', (), {
                'get': lambda url, **kw: capability_aware_request('GET', url, **kw),
                'post': lambda url, **kw: capability_aware_request('POST', url, **kw),
                'put': lambda url, **kw: capability_aware_request('PUT', url, **kw),
                'delete': lambda url, **kw: capability_aware_request('DELETE', url, **kw),
                'request': capability_aware_request
            })()
        
        # === IMPORT OVERRIDES ===
        def capability_aware_import(name, globals=None, locals=None, fromlist=(), level=0):
            context = {'module': name, 'fromlist': fromlist}
            
            if not capability_manager.check_capability(agent_name, 'module_import', context):
                raise ImportError(f"Agent '{agent_name}' denied import of module '{name}'")
            
            return __import__(name, globals, locals, fromlist, level)
        
        # === SYSTEM CONTROL OVERRIDES ===
        def capability_aware_exit(code=0):
            context = {'exit_code': code}
            
            if not capability_manager.check_capability(agent_name, 'system_exit', context):
                raise PermissionError(f"Agent '{agent_name}' denied sys.exit() access")
            
            # Convert to exception instead of actual exit
            raise SandboxError(f"Agent '{agent_name}' called sys.exit({code})")
        
        def capability_aware_eval(source, *args, **kwargs):
            context = {'source_preview': str(source)[:100]}
            
            if not capability_manager.check_capability(agent_name, 'code_eval', context):
                raise PermissionError(f"Agent '{agent_name}' denied eval() access")
            
            # Even if allowed, use restricted eval
            return eval(source, *args, **kwargs)
        
        def capability_aware_exec(*args, **kwargs):
            context = {'exec_attempt': True}
            
            if not capability_manager.check_capability(agent_name, 'code_eval', context):
                raise PermissionError(f"Agent '{agent_name}' denied exec() access")
            
            raise PermissionError("exec() not permitted even with capability")
        
        def capability_aware_remove(path):
            context = {'path': str(path)}
            
            if not capability_manager.check_capability(agent_name, 'file_delete', context):
                raise PermissionError(f"Agent '{agent_name}' denied file deletion access to {path}")
            
            import os
            return os.remove(path)
        
        # Build the overrides dictionary
        overrides.update({
            'open': capability_aware_open,
            'requests': make_requests_module(),
            '__import__': capability_aware_import,
            'exit': capability_aware_exit,
            'quit': capability_aware_exit,
            'eval': capability_aware_eval,
            'exec': capability_aware_exec,
        })
        
        # Create capability-aware os module
        import os
        safe_os = type('CapabilityAwareOS', (), {
            'path': os.path,
            'environ': dict(os.environ),
            'getcwd': os.getcwd,
            'listdir': lambda p: os.listdir(p) if capability_manager.check_capability(
                agent_name, 'file_read', {'path': p}) else None,
            'remove': capability_aware_remove,
            'unlink': capability_aware_remove,
            'system': lambda cmd: (
                os.system(cmd) if capability_manager.check_capability(
                    agent_name, 'process_exec', {'command': cmd})
                else (_ for _ in ()).throw(PermissionError(f"Agent '{agent_name}' denied os.system()"))
            ),
        })()
        overrides['os'] = safe_os
        
        return overrides
        """Create all security function overrides"""
        jail_path = config.get('path', '/tmp')
        restrictions = config.get('restrictions', {})
        
        overrides = {}
        
        # === FILE SYSTEM OVERRIDES ===
        def jailed_open(path, mode='r', *args, **kwargs):
            import builtins
            import os
            from pathlib import Path
            
            if not restrictions.get('allow_file_access', False):
                # Only allow read access in jail
                base_dir = Path(jail_path).resolve()
                raw = Path(path)
                
                # Map absolute paths into the jail
                rel = raw if not raw.is_absolute() else Path(*raw.parts[1:])
                target = (base_dir / rel).resolve()
                
                # Enforce containment
                try:
                    target.relative_to(base_dir)
                except Exception:
                    if not str(target).startswith(str(base_dir)):
                        raise PermissionError(f"Access denied outside jail: {target}")
                
                # Read-only policy unless explicitly allowed
                if any(flag in mode for flag in ('w', 'a', '+', 'x')):
                    if not restrictions.get('allow_write', False):
                        raise PermissionError(f"Write modes not allowed: {mode}")
                
                logging.debug(f"Jailed open: {path} -> {target} (mode: {mode})")
                return builtins.open(target, mode, *args, **kwargs)
            else:
                raise PermissionError("File access not permitted in sandbox")
        
        overrides['open'] = jailed_open
        
        # === NETWORK OVERRIDES ===
        def blocked_requests(*args, **kwargs):
            if restrictions.get('allow_network', False):
                import requests
                # Could add URL filtering here
                allowed_domains = restrictions.get('allowed_domains', [])
                if allowed_domains:
                    # TODO: Extract domain from request and validate
                    pass
                return requests.request(*args, **kwargs)
            else:
                raise PermissionError("Network requests not permitted in sandbox")
        
        def blocked_urlopen(*args, **kwargs):
            if restrictions.get('allow_network', False):
                import urllib.request
                return urllib.request.urlopen(*args, **kwargs)
            else:
                raise PermissionError("URL access not permitted in sandbox")
        
        def blocked_socket(*args, **kwargs):
            if restrictions.get('allow_network', False):
                import socket
                return socket.socket(*args, **kwargs)
            else:
                raise PermissionError("Socket creation not permitted in sandbox")
        
        # === SYSTEM CONTROL OVERRIDES ===
        def trapped_exit(code=0):
            """Trap sys.exit() calls and convert to exception"""
            raise SandboxError(f"Process attempted sys.exit({code}) - converted to exception")
        
        def trapped_quit(code=0):
            """Trap quit() calls"""
            raise SandboxError(f"Process attempted quit({code}) - converted to exception")
        
        def blocked_exec(*args, **kwargs):
            raise PermissionError("exec() not permitted in sandbox")
        
        def blocked_eval(source, *args, **kwargs):
            if restrictions.get('allow_eval', False):
                # Could use RestrictedPython's safe_eval here
                return eval(source, *args, **kwargs)
            else:
                raise PermissionError("eval() not permitted in sandbox")
        
        def blocked_compile(*args, **kwargs):
            if restrictions.get('allow_compile', False):
                return compile(*args, **kwargs)
            else:
                raise PermissionError("compile() not permitted in sandbox")
        
        def blocked_input(prompt=""):
            raise PermissionError("input() not permitted in sandbox")
        
        # === PROCESS/OS OVERRIDES ===
        def blocked_system(command):
            raise PermissionError("os.system() not permitted in sandbox")
        
        def blocked_popen(*args, **kwargs):
            raise PermissionError("subprocess operations not permitted in sandbox")
        
        def blocked_spawn(*args, **kwargs):
            raise PermissionError("os.spawn* operations not permitted in sandbox")
        
        def blocked_remove(path):
            if restrictions.get('allow_file_delete', False):
                import os
                # Could add path validation here
                return os.remove(path)
            else:
                raise PermissionError("File deletion not permitted in sandbox")
        
        # === IMPORT OVERRIDES ===
        def controlled_import(name, globals=None, locals=None, fromlist=(), level=0):
            """Control which modules can be imported"""
            blocked_modules = restrictions.get('blocked_modules', [
                'os', 'subprocess', 'multiprocessing', 'threading',
                'ctypes', 'importlib', 'pkgutil', 'runpy'
            ])
            
            allowed_modules = restrictions.get('allowed_modules', [
                'json', 'base64', 'math', 'datetime', 're', 'string',
                'collections', 'itertools', 'functools', 'operator'
            ])
            
            if name in blocked_modules:
                raise ImportError(f"Module '{name}' is not permitted in sandbox")
            
            if allowed_modules and name not in allowed_modules:
                raise ImportError(f"Module '{name}' is not in allowed modules list")
            
            return __import__(name, globals, locals, fromlist, level)
        
        # Add all overrides to the return dictionary
        if not restrictions.get('allow_network', False):
            overrides.update({
                'requests': type('BlockedRequests', (), {
                    'get': blocked_requests, 'post': blocked_requests,
                    'put': blocked_requests, 'delete': blocked_requests,
                    'request': blocked_requests
                })(),
                'urllib': type('BlockedUrllib', (), {
                    'request': type('BlockedRequest', (), {'urlopen': blocked_urlopen})()
                })(),
                'socket': blocked_socket,
            })
        
        overrides.update({
            'exit': trapped_exit,
            'quit': trapped_quit,
            'exec': blocked_exec,
            'eval': blocked_eval,
            'compile': blocked_compile,
            'input': blocked_input,
            '__import__': controlled_import,
        })
        
        # Add os module overrides
        if 'os' in config.get('allowed_modules', {}):
            import os
            safe_os = type('SafeOS', (), {
                'path': os.path,  # Usually safe
                'environ': dict(os.environ),  # Read-only copy
                'getcwd': os.getcwd,
                'listdir': lambda p: os.listdir(p) if restrictions.get('allow_file_access') else None,
                'system': blocked_system,
                'remove': blocked_remove,
                'unlink': blocked_remove,
            })()
            overrides['os'] = safe_os
        
        return overrides
    
    def _build_safe_globals(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Build the safe globals dictionary for RestrictedPython"""
        allowed_modules = config.get('allowed_modules', {
            'json': __import__('json'),
            'base64': __import__('base64'),
            'sys': sys,
            'math': __import__('math'),
            'datetime': __import__('datetime'),
            're': __import__('re'),
        })
        
        safe_globals = dict(self.safe_builtins)
        
        # Get capability-aware security overrides
        security_overrides = self._create_capability_aware_overrides(config)
        
        # Create a restricted sys module
        restricted_sys = type('RestrictedSys', (), {
            'version': sys.version,
            'version_info': sys.version_info,
            'platform': sys.platform,
            'path': sys.path.copy(),  # Read-only copy
            'modules': {},  # Empty modules dict
            'exit': security_overrides['exit'],  # Trapped exit
            'stdout': sys.stdout,  # Allow stdout for results
            'stderr': sys.stderr,  # Allow stderr for logging
            'stdin': None,  # Block stdin
        })()
        
        safe_globals.update({
            # RestrictedPython guards
            '_getattr_': self.default_guarded_getattr,
            '_getitem_': self.default_guarded_getitem,
            '_getiter_': self.default_guarded_getiter,
            '_unpack_sequence_': self.guarded_unpack_sequence,
            '_iter_unpack_sequence_': self.guarded_iter_unpack_sequence,
            '_apply_': self.guarded_apply,
            
            # Core types
            'list': list, 'dict': dict, 'str': str, 'int': int,
            'float': float, 'bool': bool, 'bytes': bytes,
            
            # Restricted sys module
            'sys': restricted_sys,
            
            # Security overrides
            **security_overrides,
            
            # Builtins with controlled import
            '__builtins__': {
                **self.safe_builtins, 
                '__import__': security_overrides['__import__'],
                'open': security_overrides['open'],
                'exec': security_overrides['exec'],
                'eval': security_overrides['eval'],
                'compile': security_overrides['compile'],
                'input': security_overrides['input'],
                'exit': security_overrides['exit'],
                'quit': security_overrides['quit'],
            }
        })
        
        # Add allowed modules (which may be overridden versions)
        safe_globals.update(allowed_modules)
        return safe_globals
    
    def execute(self, function_name: str, function_def: str, 
                params: Dict[str, Any], config: Dict[str, Any]) -> Tuple[Any, Dict]:
        """Execute code using RestrictedPython"""
        if not self.is_available():
            raise SandboxError("RestrictedPython backend not available")
        
        # Build the execution script
        jail_script = f'''
# Future-proof type hints evaluation
from __future__ import annotations
{function_def}
params = {repr(params)}
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
        
        logging.debug("Compiling restricted code:\n%s", jail_script)
        
        try:
            # Compile the restricted code
            compiled_code = self.compile_restricted(jail_script, '<string>', 'exec')
            if compiled_code is None:
                raise SandboxError("Failed to compile restricted code")
            
            # Build safe execution environment
            restricted_globals = self._build_safe_globals(config)
            
            # Execute the code
            exec(compiled_code, restricted_globals)
            jail_output = restricted_globals.get('jail_output')
            
            if not jail_output:
                raise SandboxError("No output from jailed execution")
            
            # Process the results
            if jail_output['success']:
                if jail_output['result_type'] == 'bytes':
                    result = base64.b64decode(jail_output['result_data'])
                else:
                    result = json.loads(jail_output['result_data'])
                status = jail_output['status']
            else:
                result, status = b'', jail_output['status']
            
            return result, status
            
        except Exception as e:
            error_msg = f"Sandbox execution failed: {str(e)}"
            logging.error(error_msg)
            return b'', {"status": {"value": 1, "reason": error_msg}}

class AgentCapabilityManager:
    """Manages fine-grained capabilities for specific agents"""
    
    def __init__(self):
        self.agent_profiles = {}
        self.capability_registry = {
            'file_read': {'description': 'Read files from disk', 'risk': 'medium'},
            'file_write': {'description': 'Write files to disk', 'risk': 'medium'},
            'file_delete': {'description': 'Delete files from disk', 'risk': 'high'},
            'network_http': {'description': 'Make HTTP requests', 'risk': 'high'},
            'network_socket': {'description': 'Raw socket access', 'risk': 'critical'},
            'process_exec': {'description': 'Execute system commands', 'risk': 'critical'},
            'code_eval': {'description': 'Dynamic code evaluation', 'risk': 'critical'},
            'module_import': {'description': 'Import Python modules', 'risk': 'medium'},
            'system_exit': {'description': 'Exit/quit system calls', 'risk': 'low'},
        }
        self.activity_log = []
        
    def register_agent_profile(self, agent_name: str, capabilities: Dict[str, Any]):
        """Register capabilities for a specific agent"""
        self.agent_profiles[agent_name] = {
            'capabilities': capabilities,
            'created': time.time(),
            'violations': 0,
            'last_activity': None
        }
        logging.info(f"Registered agent profile for '{agent_name}' with capabilities: {list(capabilities.keys())}")
    
    def check_capability(self, agent_name: str, capability: str, context: Dict[str, Any] = None) -> bool:
        """Check if an agent has a specific capability"""
        if agent_name not in self.agent_profiles:
            logging.warning(f"Unknown agent '{agent_name}' attempted {capability}")
            self._log_activity(agent_name, capability, 'DENIED', 'Unknown agent', context)
            return False
        
        profile = self.agent_profiles[agent_name]
        capabilities = profile['capabilities']
        
        # Check basic capability
        if capability not in capabilities:
            logging.warning(f"Agent '{agent_name}' denied {capability} - not in profile")
            self._log_violation(agent_name, capability, 'Capability not granted', context)
            return False
        
        capability_config = capabilities[capability]
        
        # Check domain restrictions for network capabilities
        if capability.startswith('network_') and context:
            allowed_domains = capability_config.get('allowed_domains', [])
            if allowed_domains:
                request_domain = self._extract_domain(context)
                if request_domain and request_domain not in allowed_domains:
                    logging.warning(f"Agent '{agent_name}' denied network access to '{request_domain}' - not in whitelist {allowed_domains}")
                    self._log_violation(agent_name, capability, f'Domain {request_domain} not allowed', context)
                    return False
        
        # Check file path restrictions
        if capability.startswith('file_') and context:
            allowed_paths = capability_config.get('allowed_paths', [])
            if allowed_paths:
                file_path = context.get('path', '')
                if not any(file_path.startswith(allowed_path) for allowed_path in allowed_paths):
                    logging.warning(f"Agent '{agent_name}' denied file access to '{file_path}' - not in allowed paths")
                    self._log_violation(agent_name, capability, f'Path {file_path} not allowed', context)
                    return False
        
        # Log successful capability use
        self._log_activity(agent_name, capability, 'GRANTED', 'Success', context)
        profile['last_activity'] = time.time()
        return True
    
    def _extract_domain(self, context: Dict[str, Any]) -> str:
        """Extract domain from network request context"""
        url = context.get('url', '')
        if url:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        return context.get('domain', '')
    
    def _log_activity(self, agent_name: str, capability: str, status: str, reason: str, context: Dict[str, Any] = None):
        """Log agent activity"""
        log_entry = {
            'timestamp': time.time(),
            'agent': agent_name,
            'capability': capability,
            'status': status,
            'reason': reason,
            'context': context or {}
        }
        self.activity_log.append(log_entry)
        
        # Keep log size manageable
        if len(self.activity_log) > 10000:
            self.activity_log = self.activity_log[-5000:]
    
    def _log_violation(self, agent_name: str, capability: str, reason: str, context: Dict[str, Any] = None):
        """Log security violation and increment counter"""
        self.agent_profiles[agent_name]['violations'] += 1
        self._log_activity(agent_name, capability, 'VIOLATION', reason, context)
        
        # Could trigger alerts here for repeated violations
        violations = self.agent_profiles[agent_name]['violations']
        if violations > 5:
            logging.error(f"Agent '{agent_name}' has {violations} security violations - consider review")
    
    def get_agent_activity(self, agent_name: str, hours: int = 24) -> list:
        """Get recent activity for an agent"""
        cutoff = time.time() - (hours * 3600)
        return [log for log in self.activity_log 
                if log['agent'] == agent_name and log['timestamp'] > cutoff]
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate security activity report"""
        recent_violations = [log for log in self.activity_log[-1000:] 
                           if log['status'] == 'VIOLATION']
        
        agent_stats = {}
        for agent_name, profile in self.agent_profiles.items():
            agent_activity = self.get_agent_activity(agent_name, 24)
            agent_stats[agent_name] = {
                'total_violations': profile['violations'],
                'recent_activity': len(agent_activity),
                'last_seen': profile['last_activity'],
                'capabilities': list(profile['capabilities'].keys())
            }
        
        return {
            'total_agents': len(self.agent_profiles),
            'recent_violations': len(recent_violations),
            'agent_stats': agent_stats,
            'violation_details': recent_violations[-10:]  # Last 10 violations
        }

# Global capability manager
capability_manager = AgentCapabilityManager()

def register_standard_agent_profiles():
    """Register standard agent capability profiles"""
    
    # File processing agents - only file access
    capability_manager.register_agent_profile('csv_processor', {
        'file_read': {'allowed_paths': ['/tmp/uploads/', '/data/csv/']},
        'file_write': {'allowed_paths': ['/tmp/output/']},
        'module_import': {'allowed_modules': ['csv', 'pandas', 'json']}
    })
    
    capability_manager.register_agent_profile('image_processor', {
        'file_read': {'allowed_paths': ['/tmp/uploads/', '/data/images/']},
        'file_write': {'allowed_paths': ['/tmp/output/']},
        'module_import': {'allowed_modules': ['PIL', 'cv2', 'numpy']}
    })
    
    # Network agents - domain-restricted
    capability_manager.register_agent_profile('usps_tracker', {
        'network_http': {'allowed_domains': ['tools.usps.com', 'www.usps.com']},
        'module_import': {'allowed_modules': ['requests', 'json', 'urllib']}
    })
    
    capability_manager.register_agent_profile('weather_agent', {
        'network_http': {'allowed_domains': ['api.openweathermap.org', 'api.weather.gov']},
        'module_import': {'allowed_modules': ['requests', 'json']}
    })
    
    capability_manager.register_agent_profile('openai_agent', {
        'network_http': {'allowed_domains': ['api.openai.com']},
        'module_import': {'allowed_modules': ['openai', 'requests', 'json']}
    })
    
    # Admin agents - elevated permissions  
    capability_manager.register_agent_profile('system_admin', {
        'file_read': {'allowed_paths': ['/']},  # Full system access
        'file_write': {'allowed_paths': ['/tmp/', '/var/log/agents/']},
        'file_delete': {'allowed_paths': ['/tmp/']},
        'network_http': {'allowed_domains': []},  # All domains
        'module_import': {'allowed_modules': []},  # All modules
    })
    
    # Template agents - no special capabilities
    capability_manager.register_agent_profile('template_agent', {
        'module_import': {'allowed_modules': ['json', 'string', 're']}
    })
    
    logging.info("Registered standard agent capability profiles") 
                         jail_user: str = 'agent_worker',
                         backend_type: str = 'restricted_python',
                         allowed_modules: Optional[Dict[str, Any]] = None,
                         security_profile: str = 'strict') -> Optional[Dict[str, Any]]:
    """Create a sandbox configuration dictionary with security profiles"""
    
    if jail_dir is None:
        return None
    
    # Test if the backend is available
    manager = SandboxManager()
    if backend_type not in manager.list_backends():
        logging.error(f"Sandbox backend '{backend_type}' not available")
        return None
    
    # Define security profiles
    security_profiles = {
        'strict': {
            'allow_imports': False,
            'allow_file_access': False,
            'allow_network': False,
            'allow_eval': False,
            'allow_compile': False,
            'allow_write': False,
            'allow_file_delete': False,
            'blocked_modules': [
                'os', 'subprocess', 'multiprocessing', 'threading',
                'ctypes', 'importlib', 'pkgutil', 'runpy', 'sys',
                'socket', 'urllib', 'requests', 'http', 'ftplib',
                'smtplib', 'poplib', 'imaplib', 'telnetlib'
            ],
            'allowed_modules': ['json', 'base64', 'math', 'datetime', 're', 'string']
        },
        'moderate': {
            'allow_imports': True,
            'allow_file_access': True,  # Only in jail
            'allow_network': False,
            'allow_eval': False,
            'allow_compile': False,
            'allow_write': True,  # Only in jail
            'allow_file_delete': False,
            'blocked_modules': [
                'subprocess', 'multiprocessing', 'ctypes', 'importlib',
                'socket', 'urllib', 'requests', 'http'
            ],
            'allowed_modules': [
                'json', 'base64', 'math', 'datetime', 're', 'string',
                'collections', 'itertools', 'functools', 'operator',
                'pathlib', 'os'  # Restricted version
            ]
        },
        'permissive': {
            'allow_imports': True,
            'allow_file_access': True,
            'allow_network': True,
            'allow_eval': False,  # Still dangerous
            'allow_compile': False,  # Still dangerous
            'allow_write': True,
            'allow_file_delete': True,
            'blocked_modules': ['subprocess', 'multiprocessing', 'ctypes'],
            'allowed_domains': ['api.openai.com', 'localhost'],  # Example whitelist
        }
    }
    
    restrictions = security_profiles.get(security_profile, security_profiles['strict'])
    
    # Run a simple test
    try:
        test_backend = manager.get_backend(backend_type)
        if test_backend and hasattr(test_backend, 'compile_restricted'):
            test_code = "result = 1 + 1"
            compiled = test_backend.compile_restricted(test_code, '<string>', 'exec')
            if compiled is None:
                raise SandboxError("Test compilation failed")
    except Exception as e:
        logging.error(f"Sandbox test failed: {e}")
        return None
    
    config = {
        "enabled": True,
        "type": backend_type,
        "path": jail_dir,
        "user": jail_user,
        "agent_name": agent_name,  # Key addition for capability checking
        "security_profile": security_profile,
        "allowed_modules": allowed_modules or {
            'json': __import__('json'),
            'base64': __import__('base64'), 
            'sys': None,  # Will be replaced with restricted version
            'math': __import__('math'),
            'datetime': __import__('datetime'),
            're': __import__('re'),
        },
        "restrictions": restrictions
    }
    
    logging.info(f"Sandbox config created: {backend_type}/{security_profile} in {jail_dir}")
    return config

# Global sandbox manager instance
sandbox_manager = SandboxManager()

def execute_sandboxed_function(function_name: str, function_def: str,
                              params: Dict[str, Any], 
                              sandbox_config: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict]:
    """
    Main entry point for sandboxed function execution.
    This is the function that your core workflow system would call.
    """
    if sandbox_config is None:
        raise SandboxError("No sandbox configuration provided")
    
    return sandbox_manager.execute_sandboxed(
        function_name, function_def, params, sandbox_config
    )

# Convenience function for testing with capabilities
def test_sandbox_with_capabilities():
    """Test the capability-aware sandbox functionality"""
    
    # Register some test profiles
    register_standard_agent_profiles()
    
    # Test a file agent
    test_function = '''
def test_file_access(filename):
    try:
        with open(filename, 'r') as f:
            content = f.read()
        return f"Read {len(content)} chars", {"status": {"value": 0, "reason": "Success"}}
    except Exception as e:
        return str(e), {"status": {"value": 1, "reason": str(e)}}
'''
    
    # Test legitimate access
    config = create_sandbox_config("/tmp", agent_name="csv_processor")
    if config:
        try:
            result, status = execute_sandboxed_function(
                "test_file_access", test_function, 
                {"filename": "/tmp/uploads/test.txt"}, config
            )
            print(f"Legitimate access result: {result}")
        except Exception as e:
            print(f"Legitimate access failed: {e}")
    
    # Test illegitimate access  
    config = create_sandbox_config("/tmp", agent_name="usps_tracker")  # Network agent trying file access
    if config:
        try:
            result, status = execute_sandboxed_function(
                "test_file_access", test_function,
                {"filename": "/etc/passwd"}, config
            )
            print(f"Illegitimate access result: {result}")
        except Exception as e:
            print(f"Illegitimate access blocked: {e}")
    
    # Print security report
    report = capability_manager.get_security_report()
    print(f"Security report: {json.dumps(report, indent=2)}")
    
    return True

if __name__ == "__main__":
    # Run capability-aware tests
    success = test_sandbox_with_capabilities()
    sys.exit(0 if success else 1)
