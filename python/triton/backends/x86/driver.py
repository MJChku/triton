import ctypes
import os
import tempfile
import hashlib
from pathlib import Path
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget
from triton.runtime.build import _build
from triton.runtime.cache import get_cache_manager
from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver
import functools
import subprocess

dirname = os.path.dirname(os.path.realpath(__file__))
include_dir = [os.path.join(dirname, "include")]
libdevice_dir = os.path.join(dirname, "lib")
libraries = ['cuda']

@functools.lru_cache()
def libcuda_dirs():
    env_libcuda_path = os.getenv("TRITON_LIBCUDA_PATH")
    if env_libcuda_path:
        return [env_libcuda_path]

    libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode()
    # each line looks like the following:
    # libcuda.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcuda.so.1
    locs = [line.split()[-1] for line in libs.splitlines() if "libcuda.so.1" in line]
    dirs = [os.path.dirname(loc) for loc in locs]
    env_ld_library_path = os.getenv("LD_LIBRARY_PATH")
    if env_ld_library_path and not dirs:
        dirs = [dir for dir in env_ld_library_path.split(":") if os.path.exists(os.path.join(dir, "libcuda.so.1"))]
    msg = 'libcuda.so cannot found!\n'
    if locs:
        msg += 'Possible files are located at %s.' % str(locs)
        msg += 'Please create a symlink of libcuda.so to any of the files.'
    else:
        msg += 'Please make sure GPU is set up and then run "/sbin/ldconfig"'
        msg += ' (requires sudo) to refresh the linker cache.'
    assert any(os.path.exists(os.path.join(path, 'libcuda.so.1')) for path in dirs), msg
    return dirs


@functools.lru_cache()
def library_dirs():
    return [libdevice_dir, *libcuda_dirs()]

class X86Target(GPUTarget):
    def __init__(self, arch=80):
        super().__init__(backend="x86", arch=arch, warp_size=1)
    
    def __str__(self):
        return f"x86:{self.arch}"


class X86Driver(DriverBase):
    """Driver for x86 CPU backend"""
    
    def __init__(self):
        self.utils = X86Utils()
        self.launcher_cls = X86Launcher
        super().__init__()

    @staticmethod
    def is_active():
        """Check if the x86 driver is active"""
        return True
    
    def get_current_device(self):
        return 0  # Single CPU device
    
    def get_current_stream(self, device=None):
        return 0  # No streams on CPU

    def get_current_target(self):
        return X86Target()


class _X86Utils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(_X86Utils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        mod = compile_module_from_src_with_cuda(Path(os.path.join(dirname, "driver.c")).read_text(), "cuda_utils")
        self.get_device_properties = mod.get_device_properties
        self.cuOccupancyMaxActiveClusters = mod.cuOccupancyMaxActiveClusters
        self.set_printf_fifo_size = mod.set_printf_fifo_size
        self.fill_1d_tma_descriptor = mod.fill_1d_tma_descriptor
        self.fill_2d_tma_descriptor = mod.fill_2d_tma_descriptor
    
    @staticmethod
    def load_binary(name, binary_data, shared_mem_bytes, device):
        """Load compiled binary as a callable function (x86 version)"""
        # Write binary to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.so') as f:
            f.write(binary_data)
            so_path = f.name
        
        try:
            # Load the shared library
            lib = ctypes.CDLL(so_path)
            
            # Get the function
            func = getattr(lib, name)

            print(f"Loaded x86 kernel '{name}' from {so_path}")

            func2 = getattr(lib, "x86Launcher")

            print(f"Loaded x86LaunchKernel  {func2} from {so_path}")
            
            # Return: module, function, n_regs, n_spills
            return lib, {"kernel":func, "launcher": func2}, 0, 0
            
        except Exception as e:
            if os.path.exists(so_path):
                os.unlink(so_path)
            raise e




class X86Utils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(X86Utils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        pass

    @staticmethod
    def load_binary(name, binary_data, shared_mem_bytes, device):
        """Load compiled binary as a callable function (x86 version)"""
        # Write binary to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.so') as f:
            f.write(binary_data)
            so_path = f.name
        
        try:
            # Load the shared library
            lib = ctypes.CDLL(so_path)
            
            # Get the function
            func = getattr(lib, name)

            print(f"Loaded x86 kernel '{name}' from {so_path}")

            func2 = getattr(lib, "x86Launcher")

            print(f"Loaded x86LaunchKernel  {func2} from {so_path}")
            
            # Return: module, function, n_regs, n_spills
            return lib, {"kernel":func, "launcher": func2}, 0, 0
            
        except Exception as e:
            if os.path.exists(so_path):
                os.unlink(so_path)
            raise e

    @staticmethod
    def get_device_properties(device=0):
        return {
            'multiprocessor_count': 108,  # Fake multiprocessor count for x86
            'warpSize': 32,
            'max_num_regs': 65536,  # Fake register count for x86
            'max_shared_mem': 164*1024,  # 1MB fake shared memory
            'regs_per_multiprocessor': 65536,  # Fake register count
            'warp_size': 32,
            'mem_pitch': 2147483647,
            'max_threads_per_block': 1024,  # Fake thread limit
            'max_block_dim_x': 1024,
            'max_block_dim_y': 1024, 
            'max_block_dim_z': 64,
            'max_grid_dim_x': 2147483647,
            'max_grid_dim_y': 65535,
            'max_grid_dim_z': 65535,
        }

    def cuOccupancyMaxActiveClusters(self, *args, **kwargs):
        """Fake occupancy calculation for x86"""
        # Return 1 cluster since CPU doesn't have GPU-style clusters
        return 1

    def set_printf_fifo_size(self, size):
        """Fake printf FIFO size setting (no-op for x86)"""
        # CPU printf doesn't need FIFO management
        pass

    def fill_1d_tma_descriptor(self, *args, **kwargs):
        """Fake TMA descriptor filling (no-op for x86)"""
        # x86 doesn't use Tensor Memory Accelerator
        pass

    def fill_2d_tma_descriptor(self, *args, **kwargs):
        """Fake TMA descriptor filling (no-op for x86)"""
        # x86 doesn't use Tensor Memory Accelerator  
        pass

def ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"  # Use void* for pointers on CPU
    return {
        "i1": "int32_t",
        "i8": "int8_t", 
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t", 
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]

def make_launcher(constants, signature, ids):
    arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())

    def _extracted_type(ty):
        if ty[0] == '*':
            return "PyObject*"
        return ty_to_cpp(ty)

    def format_of(ty):
        return {
            "PyObject*": "O",
            "float": "f", 
            "double": "d",
            "long": "l",
            "int8_t": "b",
            "int16_t": "h", 
            "int32_t": "i",
            "int64_t": "l",
            "uint8_t": "B",
            "uint16_t": "H",
            "uint32_t": "I", 
            "uint64_t": "K",
        }[ty]

    args_format = ''.join([format_of(_extracted_type(ty)) for ty in signature.values()])
    format = "iiiKKKOOOO" + args_format
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''
    # generate glue code for x86
    # params = [i for i in signature.keys() if i not in constants]
    params = [i for i in signature.keys()]
    src = f"""
#include <stdio.h>
#include <stdlib.h>
#include <Python.h>
#include <dlfcn.h>

// Function pointer type for the compiled kernel
typedef void (*kernel_func_t)({arg_decls});
typedef void (*launcher_func_t)(char*, int, int, int,  int, int, int, size_t, void*, int);
static  launcher_func_t indicate_launch = NULL;

// Declare the x86LaunchKernel function from stubs.c

static void _launch(int gridX, int gridY, int gridZ, int num_warps, int num_ctas, 
                   int clusterDimX, int clusterDimY, int clusterDimZ, int shared_memory, 
                   void* stream, char* func_name, void* function{', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
    // Cast function pointer and call it
    kernel_func_t kernel = (kernel_func_t)function;
    if (!indicate_launch) {{
        void* lib = dlopen("/tmp/libtriton_x86_stubs.so", RTLD_LAZY);
        indicate_launch = dlsym(lib, "x86Launcher");
    }}

    if (kernel && gridX*gridY*gridZ > 0) {{
        // For CPU, we ignore grid dimensions and just call once
        void* args_array[] = {{ {', '.join(f"&arg{i}" for i in params)} }};
        //kernel(args_array);
        
        indicate_launch(func_name, gridX, gridY, gridZ, 32*num_warps, 1, 1, shared_memory, stream, 0);
        // print all arguments
        printf("Launching kernel %s with args: \\n", func_name);
        for (int i = 0; i < {len(params)}; i++) {{
            if (i > 0) printf(", ");
            if (sizeof(args_array[i]) == sizeof(void*)) {{
                printf("arg%d=%p\\n", i, *(void**)&args_array[i]);
            }} else if (sizeof(args_array[i]) == sizeof(int)) {{
                printf("arg%d=%d\\n", i, *(int*)&args_array[i]);
            }} else if (sizeof(args_array[i]) == sizeof(float)) {{
                printf("arg%d=%.2f\\n", i, *(float*)&args_array[i]);
            }} else if (sizeof(args_array[i]) == sizeof(double)) {{
                printf("arg%d=%.2f\\n", i, *(double*)&args_array[i]);
            }} else {{
                printf("arg%d=unknown_type", i);
            }}
        }}
        printf("\\n");
        kernel({', '.join(f"arg{i}" for i in params)});

        indicate_launch(func_name, gridX, gridY, gridZ, 32*num_warps, 1, 1, shared_memory, stream, 1);
    }}
}}

static inline void* getPointer(PyObject *obj, int idx) {{
    if (PyLong_Check(obj)) {{
        return (void*)PyLong_AsUnsignedLongLong(obj);
    }}
    if (obj == Py_None) {{
        return NULL;
    }}
    PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
    if(ptr) {{
        PyObject *empty_tuple = PyTuple_New(0);
        PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
        Py_DECREF(empty_tuple);
        Py_DECREF(ptr);
        if (!PyLong_Check(ret)) {{
            PyErr_SetString(PyExc_TypeError, "data_ptr method must return 64-bit int");
            return NULL;
        }}
        void* result = (void*)PyLong_AsUnsignedLongLong(ret);
        Py_DECREF(ret);
        return result;
    }}
    PyErr_SetString(PyExc_TypeError, "Argument must be either uint64 or have data_ptr method");
    return NULL;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
    int gridX, gridY, gridZ;
    uint64_t _stream;
    uint64_t _function;
    uint64_t _func_name;
    PyObject *launch_enter_hook = NULL;
    PyObject *launch_exit_hook = NULL;
    PyObject *kernel_metadata = NULL;
    PyObject *launch_metadata = NULL;
    {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
    
    printf("Parsing arguments for x86 kernel launch\\n");
    if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &_stream, &_function, &_func_name,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
        return NULL;
    }}
    
    printf("Parsing arguments for x86 kernel launch done\\n");

    int num_warps, num_ctas, shared_memory, clusterDimX, clusterDimY, clusterDimZ;
    if (!PyArg_ParseTuple(kernel_metadata, \"iiiiii\", &num_warps, &num_ctas, &shared_memory, 
                         &clusterDimX, &clusterDimY, &clusterDimZ)) {{
        PyErr_SetString(PyExc_TypeError, "kernel_metadata must be a tuple");
        return NULL;
    }}

    // extract launch metadata
    if (launch_enter_hook != Py_None){{
        PyObject* args = Py_BuildValue("(O)", launch_metadata);
        PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
        Py_DECREF(args);
        if (!ret)
        return NULL;
    }}

    // Convert pointer arguments
    {"; ".join([f"void* ptr{i} = getPointer(_arg{i}, {i}); if (PyErr_Occurred()) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])}
    Py_BEGIN_ALLOW_THREADS;
    // Launch the kernel
    _launch(gridX, gridY, gridZ, num_warps, num_ctas, clusterDimX, clusterDimY, clusterDimZ, 
            shared_memory, (void*)_stream, (char*)_func_name, (void*)_function{', ' + ', '.join(f"ptr{i}" if ty[0]=="*" else f"_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''});
    Py_END_ALLOW_THREADS;
    if (PyErr_Occurred()) {{
        return NULL;
    }}

   
    if(launch_exit_hook != Py_None){{
        PyObject* args = Py_BuildValue("(O)", launch_metadata);
        PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
        Py_DECREF(args);
        if (!ret)
        return NULL;
    }}

    Py_INCREF(Py_None);
    return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
    {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
    {{NULL, NULL, 0, NULL}}
}};

static struct PyModuleDef ModuleDef = {{
    PyModuleDef_HEAD_INIT,
    \"__triton_launcher\",
    NULL,
    -1,
    ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_launcher(void) {{
    PyObject *m = PyModule_Create(&ModuleDef);
    if(m == NULL) {{
        return NULL;
    }}
    PyModule_AddFunctions(m, ModuleMethods);
    return m;
}}
"""
    return src

def compile_module_from_src_with_cuda(src, name):
    key = hashlib.sha256(src.encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    cache_path = cache.get_file(f"{name}.so")
    if True or cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "main.c")
            with open(src_path, "w") as f:
                f.write(src)
            so = _build(name, src_path, tmpdir, library_dirs(), include_dir, libraries)
            with open(so, "rb") as f:
                cache_path = cache.put(f.read(), f"{name}.so", binary=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, cache_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def compile_module_from_src(src, name):
    key = hashlib.sha256(src.encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    cache_path = cache.get_file(f"{name}.so")
    print(f"Compiling x86 module '{name}' with cache key {key} at {cache_path}")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "main.c")
            with open(src_path, "w") as f:
                f.write(src)
            so = _build(name, src_path, tmpdir, [], [], [])
            with open(so, "rb") as f:
                cache_path = cache.put(f.read(), f"{name}.so", binary=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, cache_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class X86Launcher(object):
    def __init__(self, src, metadata):
        self.kernel_name = metadata.name
        ids = {"ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()}
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        print(f"Creating x86 launcher with constants: {constants}, signature: {signature}, ids: {ids}")
        src_code = make_launcher(constants, signature, ids)
        mod = compile_module_from_src(src_code, "__triton_launcher")
        self.launch = mod.launch

    def __call__(self, *args, **kwargs):
        # Convert args to list so we can modify them
        args_list = list(args)
        
        # Check if we have enough args (9 fixed args + kernel args)
        if len(args_list) < 9:
            raise ValueError(f"Expected at least 9 arguments, got {len(args_list)}")
        
        func_dict = args_list[4]
        
        def translate_func(function_arg):
            if function_arg is None:
                return 0
            elif isinstance(function_arg, ctypes._CFuncPtr):
                # This is the key fix: convert ctypes function pointer to integer
                return ctypes.cast(function_arg, ctypes.c_void_p).value
            elif hasattr(function_arg, 'value'):
                # Some ctypes objects have a .value attribute
                return function_arg.value
            elif not isinstance(function_arg, int):
                # Try to convert to integer
                try:
                    return int(function_arg)
                except (ValueError, TypeError):
                    return 0

        kernel_args_start = 9

        args_list[4] = translate_func(func_dict["kernel"])
        list_copy = args_list[5:]

        # insert kernel metadata
        kernel_name_bytes = self.kernel_name.encode('utf-8')
        print("!!!!!! kernle name : ", self.kernel_name)
        kernel_name_ptr = ctypes.c_char_p(kernel_name_bytes)
        args_list[5] = ctypes.cast(kernel_name_ptr, ctypes.c_void_p).value  # Get raw pointer value

        kernel_args_start += 1
        args_list[6:] = list_copy

        # The first 9 arguments are fixed: gridX, gridY, gridZ, stream, function, 
        # kernel_metadata, launch_metadata, launch_enter_hook, launch_exit_hook
        # The rest are kernel arguments that might need GPU->CPU conversion
        
        for i in range(kernel_args_start, len(args_list)):
            arg = args_list[i]

            if hasattr(arg, 'device'):
                needs_cpu_conversion = False
                if hasattr(arg.device, 'type') and arg.device.type != 'cpu':
                    needs_cpu_conversion = True
                elif not hasattr(arg.device, 'type'):
                    assert(0)
                
                if needs_cpu_conversion:
                    print(f"Info: Automatically copying GPU tensor (argument {i-kernel_args_start}) to CPU for x86 backend")
                    if hasattr(arg, 'cpu'):
                        cpu_tensor = arg.cpu()
                        args_list[i] = cpu_tensor
                    else:
                        print(f"Warning: Don't know how to convert argument {i-kernel_args_start} to CPU")
                        assert(0)
            
            # Handle objects that might be GPU pointers but don't have device attribute
            elif hasattr(arg, 'data_ptr'):
                # This could be a raw tensor pointer - check if we can convert it
                if hasattr(arg, 'cpu'):
                    cpu_version = arg.cpu()
                    args_list[i] = cpu_version
                    print(f"Info: Converted data_ptr object (argument {i-kernel_args_start}) to CPU")
                else:
                    print(f"Warning: Don't know how to convert argument {i-kernel_args_start} to CPU")
                    assert(0)
         
        # Call the actual launcher with converted arguments
        self.launch(*args_list, **kwargs)
