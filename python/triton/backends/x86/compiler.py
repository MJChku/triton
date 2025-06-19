from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, llvm, nvidia

from dataclasses import dataclass
import functools
from typing import Any, Tuple, Optional
import hashlib
import re
import tempfile
import signal
import os
import subprocess
from pathlib import Path


@functools.lru_cache()
def _path_to_binary(binary: str):
    """Find system binaries like gcc, clang"""
    paths = [
        os.environ.get(f"TRITON_{binary.upper()}_PATH", ""),
        binary,  # Try system PATH
    ]

    for bin_path in paths:
        if not bin_path:
            continue
        try:
            result = subprocess.check_output([bin_path, "--version"], stderr=subprocess.STDOUT)
            if result is not None:
                return bin_path, "system"
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    raise RuntimeError(f"Cannot find {binary}")


@functools.lru_cache()
def get_gcc_version():
    """Get GCC version for x86 compilation"""
    try:
        gcc_path, _ = _path_to_binary("gcc")
        version = subprocess.check_output([gcc_path, "--version"]).decode("utf-8")
        return version.split('\n')[0]
    except:
        return "gcc-unknown"


@functools.lru_cache(None)
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@dataclass(frozen=True)
class X86Options:
    # x86
    optimization_level: str = "O2"  # Optimization level for GCC
    extern_libs: dict = None

    # fake cuda options
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 3
    # maxnreg corresponds to the ptx parameter .maxnreg, which controls the
    # maximum number of 32-bit registers used by one thread.
    maxnreg: Optional[int] = None
    cluster_dims: tuple = (1, 1, 1)
    ptx_version: int = None
    enable_fp_fusion: bool = True
    allow_fp8e4nv: bool = False
    allow_fp8e4b15: bool = False
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str] = ("tf32", "tf32x3", "ieee")
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None
    debug: bool = False
    backend_name: str = 'x86'
    waves_per_eu: int = 0

    def __post_init__(self):
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))

    def hash(self):
        hash_dict = dict(self.__dict__)
        hash_dict["extern_libs"] = tuple((k, file_hash(v)) for k, v in sorted(hash_dict["extern_libs"]))
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class X86Backend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        print(f"Checking if target {target}:{target.backend} is supported by X86Backend")
        return target.backend == 'x86'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.capability = target.arch
        assert isinstance(self.capability, int)
        self.arch = target.arch
        self.binary_ext = "so"

    def parse_options(self, opts) -> Any:
        args = {k: opts[k] for k in X86Options.__dataclass_fields__.keys() if k in opts}
        args["allow_fp8e4nv"] = self.capability >= 89
        args["allow_fp8e4b15"] = self.capability < 90
        args["max_num_imprecise_acc_default"] = 2**30 if self.capability == 90 else 0
        return X86Options(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
        )


    def get_codegen_implementation(self):
        # No custom codegen needed for x86
        return {}

    def load_dialects(self, ctx):
        # Load NVIDIA dialects since we're converting from GPU IR
        nvidia.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, opt, capability):
        cluster_info = nvidia.ClusterInfo()
        if opt.cluster_dims is not None:
            cluster_info.clusterDimX = opt.cluster_dims[0]
            cluster_info.clusterDimY = opt.cluster_dims[1]
            cluster_info.clusterDimZ = opt.cluster_dims[2]
        # TTIR -> TTGIR
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(pm, f"cuda:{capability}", opt.num_warps, 32, opt.num_ctas)
        # optimize TTGIR
        passes.ttgpuir.add_coalesce(pm)
        if capability // 10 >= 8:
            passes.ttgpuir.add_f32_dot_tc(pm)
        # TODO(Qingyi): Move PlanCTAPass to the front of CoalescePass
        nvidia.passes.ttnvgpuir.add_plan_cta(pm, cluster_info)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        passes.ttgpuir.add_accelerate_matmul(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
        passes.common.add_cse(pm)
        if capability // 10 >= 8:
            passes.ttgpuir.add_combine_tensor_select_and_if(pm)
            passes.ttgpuir.add_pipeline(pm, opt.num_stages)
        passes.ttgpuir.add_prefetch(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_reduce_data_duplication(pm)
        passes.ttgpuir.add_reorder_instructions(pm)
        passes.ttgpuir.add_emit_thread_iters(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        if capability // 10 >= 9:
            nvidia.passes.ttnvgpuir.add_fence_insertion(pm)
            nvidia.passes.ttnvgpuir.add_tma_lowering(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod)
        metadata["cluster_dims"] = (cluster_info.clusterDimX, cluster_info.clusterDimY, cluster_info.clusterDimZ)
        return mod

    @staticmethod
    def make_llir(src, metadata, options, capability):
        # warp-specialization mutates num_warps
        num_warp_groups = src.get_int_attr("triton_gpu.num-warp-groups-per-cta")
        if num_warp_groups is not None:
            metadata["num_warps"] *= num_warp_groups
        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        nvidia.passes.ttgpuir.add_decompose_unsupported_conversions(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_index_to_llvmir(pm)
        passes.ttgpuir.add_allocate_shared_memory(pm)

        nvidia.passes.ttgpuir.add_to_llvmir(pm, capability)
        # nvidia.passes.ttgpuir.add_x86_taint_analysis(pm)
    
        nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm(pm)
        passes.convert.add_arith_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        # if os.environ.get("TRITON_DISABLE_LINE_INFO", "0") == "0":
        #     passes.llvmir.add_di_scope(pm)
        pm.run(mod)
        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)

        if options.extern_libs:
            paths = [path for (name, path) in options.extern_libs]
            llvm.link_extern_libs(llvm_mod, paths)

        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)

        # Get some metadata
        metadata["shared"] = src.get_int_attr("triton_x86.shared")
        print(f"Shared memory size: {metadata['shared']} bytes")
        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret

    @staticmethod
    def make_x86_asm(src, metadata, opt, arch):
        """Convert LLVM IR to x86 assembly"""
        triple = 'x86_64-pc-linux-gnu'
        proc = 'x86-64'
        features = ''  # No special features needed
        
        ret = llvm.translate_to_asm(src, triple, proc, features, [], opt.enable_fp_fusion, False)
        
        # Find function names
        names = re.findall(r'\.globl\s+([a-zA-Z_][a-zA-Z0-9_]*)', ret)
        if names:
            metadata["name"] = names[0]
        else:
            # Fallback - look for function definitions
            names = re.findall(r'^([a-zA-Z_][a-zA-Z0-9_]*):', ret, re.MULTILINE)
            if names:
                metadata["name"] = names[0]
        
        if os.environ.get("X86_ENABLE_DUMP", "0") == "1":
            print("// -----// x86 Assembly Dump //----- //")
            print(ret)
        
        return ret

    @staticmethod
    @functools.lru_cache()
    def get_cached_stub_so(opt_hash):
        """Get cached stub shared library or compile if not exists"""
        gcc_path, _ = _path_to_binary("gcc")
        
        # Get stubs.c path and hash
        current_dir = os.path.dirname(__file__)
        stub_file_path = os.path.join(current_dir, 'stubs.c')
        stub_hash = file_hash(stub_file_path)
        
        cached_stub_path = Path("/tmp/libtriton_x86_stubs.so") 
        
        # if cached_stub_path.exists():
        #     return str(cached_stub_path)
        
        # Compile stubs.c to cached location
        with tempfile.NamedTemporaryFile(delete=False, mode='r', suffix='.log') as flog:
            optimization = '-O2'  # Use default optimization for stubs
            debug_flag = '-g'
            cmd = f'{gcc_path} -shared -fPIC {optimization} {debug_flag} {stub_file_path} -lm -ldl -o {cached_stub_path} 2> {flog.name}'
            
            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                with open(flog.name) as log_file:
                    log = log_file.read()
                raise RuntimeError(f'GCC compilation of stubs failed with error code {e.returncode}: \n{log}')
            finally:
                if os.path.exists(flog.name):
                    os.remove(flog.name)
        
        return str(cached_stub_path)

    @staticmethod
    def make_so(src, metadata, opt, arch):
        """Compile x86 assembly and link with cached stubs shared library"""
        gcc_path, _ = _path_to_binary("gcc")
        
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.s') as fsrc, \
            tempfile.NamedTemporaryFile(delete=False, mode='r', suffix='.log') as flog:
            
            # Write assembly
            fsrc.write(src)
            fsrc.flush()
            
            # Get cached stub shared library
            stub_so_path = X86Backend.get_cached_stub_so(opt.hash())
            
            fbin = fsrc.name.replace('.s', '.so')

            # Compile assembly and link with stub shared library
            optimization = f'-{opt.optimization_level}' if opt.optimization_level else '-O2'
            debug_flag = '-g' if opt.debug else ''
            
            cmd = f'{gcc_path} -shared -fPIC {optimization} {debug_flag} {fsrc.name} {stub_so_path} -lm -o {fbin} 2> {flog.name}'

            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                with open(flog.name) as log_file:
                    log = log_file.read()
                raise RuntimeError(f'GCC compilation failed with error code {e.returncode}: \n{log}')
            finally:
                # Cleanup intermediate files
                for f in [fsrc.name, flog.name]:
                    if os.path.exists(f):
                        os.remove(f)

            # Read compiled binary
            with open(fbin, 'rb') as f:
                binary = f.read()
            
            # Cleanup final binary file
            if os.path.exists(fbin):
                os.remove(fbin)
            
            return binary
            
    @staticmethod
    def get_stub_code():
        """Return the stub C code with NVIDIA function implementations"""
        try:
            current_dir = os.path.dirname(__file__)
            stub_file_path = os.path.join(current_dir, 'stubs.c')
            with open(stub_file_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            raise RuntimeError("Stub file 'stubs.c' not found in x86 backend directory.")

    def add_stages(self, stages, options):
        """Define the x86 compilation pipeline stages"""
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, self.arch)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, self.arch)
        stages["x86_asm"] = lambda src, metadata: self.make_x86_asm(src, metadata, options, self.arch)
        stages["so"] = lambda src, metadata: self.make_so(src, metadata, options, self.arch)

    @functools.lru_cache()
    def hash(self):
        version = get_gcc_version()
        return f'{version}-{self.arch}'
