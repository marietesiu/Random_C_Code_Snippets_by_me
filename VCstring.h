#include <stddef.h>
#include <stdlib.h>
#include <string.h>
/**
 * @brief Copies memory where the destination, source, or both are volatile.
 *        Ensures the compiler does not optimize away any read or write.
 */

/**
 * @brief Allocates new memory, copies data into it, and returns it as read-only.
 *        This effectively acts as a safe "copy to const" operation.
 */


void volatile_memcpy(volatile void *dest, const volatile void *src, size_t n) {
    volatile char *d = (volatile char *)dest;
    const volatile char *s = (const volatile char *)src;
    
    for (size_t i = 0; i < n; i++) {
        d[i] = s[i]; // Every read and write is strictly executed
    }
}


const void* const_memcpy_alloc(const void *src, size_t n) {
    // 1. Allocate normal, modifiable memory
    void *dest = malloc(n);
    if (dest == NULL) {
        return NULL; 
    }
    
    // 2. Safely copy the data into the temporary modifiable buffer
    memcpy(dest, src, n);
    
    // 3. Return it casted to a const pointer, locking it from future edits
    return (const void*)dest;
}



/**
 * @brief Fills volatile memory with a specific byte value.
 *        Guarantees that no write operations are optimized away by the compiler.
 */
volatile void* volatile_memset(volatile void *s, int c, size_t n) {
    // Cast to byte pointer to allow exact byte-by-byte memory access
    volatile unsigned char *p = (volatile unsigned char *)s;
    unsigned char value = (unsigned char)c;

    for (size_t i = 0; i < n; i++) {
        p[i] = value; // Volatile forces a physical write on every iteration
    }

    return s;
}


volatile void* volatile_memset(volatile void *s, int c, size_t n) {
    // Cast the generic volatile void pointer to a volatile unsigned char pointer.
    // This allows us to perform precise, byte-by-byte memory modification.
    volatile unsigned char *p = (volatile unsigned char *)s;
    
    // Cast the target value to an unsigned char to match the byte size.
    unsigned char value = (unsigned char)c;

    // Loop through every single byte specified by the size 'n'.
    for (size_t i = 0; i < n; i++) {
        // Because 'p' is a pointer to 'volatile', the compiler is forced
        // to execute this assignment explicitly on every loop iteration.
        p[i] = value;
    }
    return s;
}
