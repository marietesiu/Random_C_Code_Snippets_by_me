// /sys/kernel/mm/page_idle/bitmap is useful for page tracking
access or return the Referenced and Dirty metadata flags for a specific page from userspace, you must read the /proc/[pid]/pagemap:
Bit 62 -> dirty bit
Bit 63 -> if page is in RAM

trackPage(){
if (kernel_page){ return 1;}
if (self){ return 1;}
if (external_DMA) {return 1;}
if (too_dynamic) {return 1;} // clear dirty flag and check if set back to 1

else {return 0;}

}


possible code to use / review & learn from:

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>

// Returns 1 if the page at 'addr' is present and dirty, 0 otherwise
int check_page_metadata(pid_t pid, uintptr_t addr) {
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/pagemap", pid);
    
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;

    // Calculate index: virtual address divided by system page size (usually 4096)
    size_t page_size = sysconf(_SC_PAGESIZE);
    uint64_t page_index = addr / page_size;
    off_t offset = page_index * sizeof(uint64_t);

    uint64_t pagemap_entry = 0;
    if (lseek(fd, offset, SEEK_SET) == -1) {
        close(fd);
        return -1;
    }

    if (read(fd, &pagemap_entry, sizeof(pagemap_entry)) != sizeof(pagemap_entry)) {
        close(fd);
        return -1;
    }
    close(fd);

    // Bit 63 indicates if the page is present in RAM
    int is_present = (pagemap_entry >> 63) & 1;
    // Bit 62 tracks the soft-dirty state
    int is_dirty = (pagemap_entry >> 62) & 1;

    printf("Page Present: %d | Soft-Dirty: %d\n", is_present, is_dirty);
    return is_dirty;
}
