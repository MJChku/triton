#define _GNU_SOURCE
#define __USE_GNU
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <execinfo.h>
#include <dlfcn.h>
#include <signal.h>
#include <inttypes.h>
#include <link.h>

static const int MAX_FRAMES = 100;
static void
crash_handler(int sig, siginfo_t *si, void *unused)
{
    void *frames[MAX_FRAMES];
    int  frame_count;
    int  log_fd;

    /* Open fresh crash log */
    log_fd = open("crash_trace.log",
                  O_CREAT|O_WRONLY|O_TRUNC, 0644);
    if (log_fd < 0) log_fd = STDERR_FILENO;

    /* Signal info */
    dprintf(log_fd,
            "ERROR: signal %d (%s) at address %p\n",
            sig, strsignal(sig), si->si_addr);

    /* Backtrace */
    frame_count = backtrace(frames, MAX_FRAMES);
    backtrace_symbols_fd(frames, frame_count, log_fd);

    unsetenv("LD_PRELOAD");
    /* For each frame, resolve via addr2line (full path) and write only that */
   for (int i = 0; i < frame_count; ++i) {
        Dl_info info;
        if (dladdr(frames[i], &info) != 0) {  // Check if dladdr succeeded
            const char *obj = info.dli_fname ?: "/proc/self/exe";
            uintptr_t base = (uintptr_t)info.dli_fbase;
            uintptr_t addr = (uintptr_t)frames[i];
            uintptr_t offset = addr - base;

            char cmd[512];
            snprintf(cmd, sizeof(cmd),
                "/usr/bin/addr2line -f -p -e %s 0x%" PRIxPTR,
                obj, offset);

            FILE *fp = popen(cmd, "r");
            if (fp) {
                char buf[512];
                while (fgets(buf, sizeof(buf), fp)) {
                    dprintf(log_fd, "    src: %s", buf);
                }
                pclose(fp);
            }
        } else {
            dprintf(log_fd, "    dladdr failed for frame %d\n", i);
        }
    }
    if (log_fd != STDERR_FILENO) close(log_fd);
    _exit(1);
}
static int installed_handler = 0;

static void
install_crash_handler(void)
{
    struct sigaction sa;
    sa.sa_sigaction = crash_handler;
    sa.sa_flags     = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);

    /* Catch abort (SIGABRT), illegal instruction, segfault, etc. */
    sigaction(SIGABRT, &sa, NULL);
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGILL,  &sa, NULL);
    sigaction(SIGFPE,  &sa, NULL);
}

void memory_assertion_failure(const char* message) {
    fprintf(stderr, "MEMORY ASSERTION FAILURE: %s\n", message);
    fprintf(stderr, "This indicates a memory operation using addresses computed from dummy/tainted data\n");
    fprintf(stderr, "This could lead to segmentation faults or memory corruption\n");
    
    // Print stack trace
    // print_stack_trace();
    
    // Optionally continue with a warning instead of aborting
    // abort();  // Uncomment to abort on memory assertion failure
}

void branch_assertion_failure(char* message){
    printf("BRANCH ASSERTION FAILURE: %s\n", message);
    fflush(stdout);
    assert(0);
}

void debug_load_address(uint64_t address){
    // Print the address being loaded for debugging purposes
    // printf("DEBUG LOAD ADDRESS: 0x%" PRIx64 "\n", address);
    // fflush(stdout);
    // void* ptr = (void*)address;
    // volatile uint64_t value = *(volatile uint64_t*)ptr;  // Force load
    // printf("DEBUG LOAD VALUE: 0x%" PRIx64 "\n", value);
    // fflush(stdout);
}

// NVIDIA math functions - map to standard C library equivalents
float __nv_exp2f(float x) {
    return exp2f(x);  // 2^x
}

float __nv_log2f(float x) {
    return log2f(x);  // log base 2 of x
}

float __nv_expf(float x) {
    return expf(x);   // e^x
}

float __nv_logf(float x) {
    return logf(x);   // natural log
}

float __nv_sinf(float x) {
    return sinf(x);
}

float __nv_cosf(float x) {
    return cosf(x);
}

float __nv_tanf(float x) {
    return tanf(x);
}

float __nv_sqrtf(float x) {
    return sqrtf(x);
}

float __nv_powf(float x, float y) {
    return powf(x, y);
}

float __nv_fabsf(float x) {
    return fabsf(x);
}

// Double precision versions if needed
double __nv_exp2(double x) {
    return exp2(x);
}

double __nv_log2(double x) {
    return log2(x);
}

// Linked list node to track allocations
typedef struct allocation_node {
    void* ptr;
    size_t size;
    char* name;
    struct allocation_node* next;
} allocation_node_t;

// Global head of the allocation list
static allocation_node_t* allocation_head = NULL;

void* metrics_alloca(size_t size, const char* name) {
    printf("Allocating %zu bytes for Triton metrics: %s\n", size, name);
    
    void* ptr = malloc(size);
    memset(ptr, 0, size); 
    if (!ptr) {
        fprintf(stderr, "Failed to allocate %zu bytes for metrics: %s\n", size, name);
        exit(1);
    }
    
    // Create new node for linked list
    allocation_node_t* node = malloc(sizeof(allocation_node_t));
    if (!node) {
        fprintf(stderr, "Failed to allocate tracking node for: %s\n", name);
        free(ptr);
        exit(1);
    }
    
    // Fill node data
    node->ptr = ptr;
    node->size = size;
    node->name = strdup(name);  // Copy the name string
    node->next = allocation_head;
    allocation_head = node;
    
    return ptr;
}

// Function to print all tracked allocations
void print_all_allocations(void) {
    
    printf("\n=== Triton Metrics Allocations ===\n");
    allocation_node_t* current = allocation_head;
    int count = 0;
    
    while (current) {
        printf("Allocation %d:\n", ++count);
        printf("  Name: %s\n", current->name);
        printf("  Address: %p\n", current->ptr);
        printf("  Size: %zu bytes\n", current->size);
        
        // Print first few values if it's an array of int32
        if (current->size >= 4 && current->size % 4 == 0) {
            int* int_array = (int*)current->ptr;
            int num_ints = current->size / 4;
            printf("  Values (only print first # of %d ints): ", num_ints > 8 ? 8 : num_ints);
            for (int i = 0; i < num_ints && i < 8; i++) {
                printf("%d ", int_array[i]);
            }
            printf("\n");
        }
        printf("\n");
        current = current->next;
    }
    
    if (count == 0) {
        printf("No allocations tracked.\n");
    }
    printf("=== End of Allocations ===\n\n");
}

// Function to get allocation by name
void* get_allocation_by_name(const char* name) {
    allocation_node_t* current = allocation_head;
    while (current) {
        if (strcmp(current->name, name) == 0) {
            return current->ptr;
        }
        current = current->next;
    }
    return NULL;
}

// Cleanup function
void metrics_free_all(void) {
    allocation_node_t* current = allocation_head;
    while (current) {
        allocation_node_t* next = current->next;
        printf("Freeing Triton metrics allocation: %s\n", current->name);
        free(current->ptr);
        free(current->name);
        free(current);
        current = next;
    }
    allocation_head = NULL;
}

// Rest of your existing stub functions...
void nvvm_barrier0(void) {
    // No-op for CPU
}

void nvvm_star(void) {
    // No-op for CPU
}

int nvvm_ctaid(int dim) {
    return 0;  // Single block on CPU
}

int nvvm_nctaid(int dim) {
    return 1;  // Single block on CPU
}

int nvgpu_cluster_id(void) {
    return 0;
}

int nvvm_tid(int dim) {
    return 0;
}

// In gpu_stubs.c, enhance the metrics_dummy function
static uint64_t dummy_counter = 0;

uint64_t metrics_dummy(void) {
    
    dummy_counter++;
    // int tid = syscall(SYS_gettid);  // Get thread ID for debugging
    // printf("metrics_dummy(%d) #%lu: returning %d\n", tid, dummy_counter, 0);
    return 0;

}

// Global shared memory placeholder
#define GSIZE 1024*1024+4096 // 164KB, aligned 16B

char global_smem[GSIZE];  // Adjust size as needed


// Add this function to your stubs.c file
void aggregateAndDumpAllocationStats(const char* funcName,
                                    int gridDimX, int gridDimY, int gridDimZ,
                                    int blockDimX, int blockDimY, int blockDimZ) {
    
    // Simple hash map using arrays (since we don't have C++ containers)
    // Assuming max 100 unique (name, position) combinations
    #define MAX_UNIQUE_COUNTERS 100
    
    typedef struct {
        char name[256];
        int position;
        int64_t value;
        int used;
    } unique_counter_t;
    
    static unique_counter_t uniqueCounters[MAX_UNIQUE_COUNTERS];
    memset(uniqueCounters, 0, sizeof(uniqueCounters));
    int numUniqueCounters = 0;
    
    // Iterate through all allocation nodes
    allocation_node_t* current = allocation_head;
    while (current) {
        // Process each allocation node's int_array
        if (current->size >= 4 && current->size % 4 == 0) {
            int* int_array = (int*)current->ptr;
            int num_ints = current->size / 4;
            
            // For each position i in int_array[i]
            for (int i = 0; i < num_ints; i++) {
                // Find or create unique counter entry
                int found = -1;
                for (int j = 0; j < numUniqueCounters; j++) {
                    if (uniqueCounters[j].used && 
                        strcmp(uniqueCounters[j].name, current->name) == 0 &&
                        uniqueCounters[j].position == i) {
                        found = j;
                        break;
                    }
                }
                
                if (found >= 0) {
                    // Sum with existing counter
                    uniqueCounters[found].value += int_array[i];
                } else if (numUniqueCounters < MAX_UNIQUE_COUNTERS) {
                    // Create new unique counter
                    strncpy(uniqueCounters[numUniqueCounters].name, current->name, 255);
                    uniqueCounters[numUniqueCounters].name[255] = '\0';
                    uniqueCounters[numUniqueCounters].position = i;
                    uniqueCounters[numUniqueCounters].value = int_array[i];
                    uniqueCounters[numUniqueCounters].used = 1;
                    numUniqueCounters++;
                } else {
                    fprintf(stderr, "Warning: Max unique counters exceeded!\n");
                }
            }
        }
        current = current->next;
    }
    
    char filename[2048];
    int tid = syscall(SYS_gettid);
    snprintf(filename, sizeof(filename), "/tmp/%d_stats.txt", tid);
    
    // Open file for writing
    FILE* outFile = fopen(filename, "w");
    if (!outFile) {
        fprintf(stderr, "Error: Could not open file %s for writing\n", filename);
        return;
    }
    
    // Write function metadata
    fprintf(outFile, "func_name=%s\n", funcName);
    fprintf(outFile, "gridDimX=%d\n", gridDimX);
    fprintf(outFile, "gridDimY=%d\n", gridDimY);
    fprintf(outFile, "gridDimZ=%d\n", gridDimZ);
    fprintf(outFile, "blockDimX=%d\n", blockDimX);
    fprintf(outFile, "blockDimY=%d\n", blockDimY);
    fprintf(outFile, "blockDimZ=%d\n", blockDimZ);
    
    // Write all unique counter values in one line, separated by commas
    fprintf(outFile, "counters=");
    for (int i = 0; i < numUniqueCounters; i++) {
        if (i > 0) {
            fprintf(outFile, ",");
        }
        fprintf(outFile, "%ld", uniqueCounters[i].value);
    }
    fprintf(outFile, "\n");
    
    // Write detailed breakdown for reference
    fprintf(outFile, "\n# Detailed counter breakdown:\n");
    for (int i = 0; i < numUniqueCounters; i++) {
        fprintf(outFile, "# %s[%d] = %ld\n", 
                uniqueCounters[i].name, 
                uniqueCounters[i].position, 
                uniqueCounters[i].value);
    }
    
    fclose(outFile);
    
    printf("Allocation stats dumped to: %s\n", filename);
    printf("Found %d unique counters\n", numUniqueCounters);
}

// x86 kernel launch function - mimics cudaLaunchKernel
void x86Launcher(char* func_name, 
                     int gridDimX, int gridDimY, int gridDimZ,
                     int blockDimX, int blockDimY, int blockDimZ,
                     size_t sharedMem, 
                     void* stream,
                     int is_end) {
    if (!installed_handler) {
        install_crash_handler();
        installed_handler = 1;
    }
    if (!is_end) {
        memset(global_smem, 0, GSIZE);  // Clear global shared memory
        printf("Starting kernel: %s with grid (%d, %d, %d) and block (%d, %d, %d)\n",
               func_name, gridDimX, gridDimY, gridDimZ,
               blockDimX, blockDimY, blockDimZ);
    } else {
        print_all_allocations();
        printf("Ending kernel: %s\n", func_name);
        metrics_free_all();
    }
}