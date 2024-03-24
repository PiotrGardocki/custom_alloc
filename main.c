#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <threads.h>
#include <omp.h>
#include <assert.h>

#define MEMORY_SIZE (1024 * 1024)
unsigned char _Alignas(_Alignof(max_align_t)) buffer[MEMORY_SIZE];
mtx_t memory_lock;

struct block_header
{
    struct block_header *next;
    void *data;
    size_t size;
};

#define ALIGN_ADDRESS(address, alignment) \
    (((address + alignment - 1) / alignment) * alignment)
#define BLOCK_END_FROM_HEADER(header) \
    ((uintptr_t)header->data + header->size)
#define NEXT_HEADER_ADDRESS(header) \
    (ALIGN_ADDRESS(BLOCK_END_FROM_HEADER(header), _Alignof(*header)))
#define NEXT_ALIGNED_BLOCK_ADDRESS(header, alignment) \
    (ALIGN_ADDRESS(NEXT_HEADER_ADDRESS(header) + sizeof(*header), alignment))

void* my_aligned_alloc(size_t alignment, size_t size)
{
    uintptr_t available_memory, next_usable_address;
    struct block_header *prev, *current, *new;

    assert(alignment > 0);

    mtx_lock(&memory_lock);

    // The first header does not point to any data, it is just a list head
    prev = (struct block_header*)&buffer[0];
    if (prev->data == NULL)
        // Keep fake pointer for convenience (used in macros)
        prev->data = prev + 1;

    current = prev->next;

    // Look for free memory between allocated blocks
    while (current != NULL) {
        next_usable_address = NEXT_ALIGNED_BLOCK_ADDRESS(prev, alignment);
        available_memory = (uintptr_t)current - next_usable_address;

        if ((uintptr_t)current > next_usable_address && available_memory >= size) {
            new = (void*)NEXT_HEADER_ADDRESS(prev);
            new->data = (void*)next_usable_address;
            new->size = size;
            new->next = current;
            prev->next = new;

            mtx_unlock(&memory_lock);
            return new->data;
        }

        prev = current;
        current = current->next;
    }

    // Check for memory after the allocated blocks
    next_usable_address = NEXT_ALIGNED_BLOCK_ADDRESS(prev, alignment);
    available_memory = (uintptr_t)&buffer[MEMORY_SIZE] - next_usable_address;

    if ((uintptr_t)&buffer[MEMORY_SIZE] > next_usable_address && available_memory >= size) {
        new = (void*)NEXT_HEADER_ADDRESS(prev);
        new->data = (void*)next_usable_address;
        new->size = size;
        new->next = NULL;
        prev->next = new;

        mtx_unlock(&memory_lock);
        return new->data;
    }

    mtx_unlock(&memory_lock);
    return NULL;
}

void* my_malloc(size_t size)
{
    return my_aligned_alloc(_Alignof(max_align_t), size);
}

void my_free(void *ptr)
{
    struct block_header *prev, *current;

    mtx_lock(&memory_lock);

    prev = (struct block_header*)&buffer[0];
    current = prev->next;

    while (current != NULL) {
        if (ptr == current->data) {
            prev->next = current->next;
            mtx_unlock(&memory_lock);
            return;
        }

        prev = current;
        current = current->next;
    }

    mtx_unlock(&memory_lock);
}

int main()
{
    mtx_init(&memory_lock, mtx_plain);

    // printf("Buffer start:   %p, max_align_t: %u\n\n", buffer, _Alignof(max_align_t));

    // void *ptr1 = my_malloc(8);
    // printf("Allocated ptr1: %llu\n", (uintptr_t)ptr1 - (uintptr_t)buffer);
    // void *ptr2 = my_malloc(10);
    // printf("Allocated ptr2: %llu\n", (uintptr_t)ptr2 - (uintptr_t)buffer);
    // void *ptr3 = my_malloc(256);
    // printf("Allocated ptr3: %llu\n", (uintptr_t)ptr3 - (uintptr_t)buffer);
    // void *ptr4 = my_malloc(2);
    // printf("Allocated ptr4: %llu\n", (uintptr_t)ptr4 - (uintptr_t)buffer);

    // my_free(ptr2);
    // printf("Freed ptr2: %llu\n", (uintptr_t)ptr2 - (uintptr_t)buffer);

    // void *ptr5 = my_malloc(17);
    // printf("Allocated ptr5: %llu\n", (uintptr_t)ptr5 - (uintptr_t)buffer);
    // void *ptr6 = my_malloc(15);
    // printf("Allocated ptr6: %llu\n", (uintptr_t)ptr6 - (uintptr_t)buffer);

    //omp_set_num_threads(8);
    printf("Total threads: %u\n", omp_get_max_threads());

    #pragma omp parallel
    {
        #define N 1000
        unsigned *arr[N];
        unsigned tid = omp_get_thread_num();

        for (int i = 0; i < N; i++) {
            arr[i] = my_aligned_alloc(_Alignof(unsigned), sizeof(unsigned));
            arr[i][0] = tid;
        }

        #pragma omp barrier

        for (int i = 0; i < N; i++)
            assert(arr[i][0] == tid);

        #pragma omp barrier

        for (int i = 0; i < N; i++) {
            my_free(arr[i]);
            arr[i] = my_aligned_alloc(_Alignof(unsigned), sizeof(unsigned));
            my_free(arr[i]);
            arr[i] = my_aligned_alloc(_Alignof(unsigned), sizeof(unsigned));
            arr[i][0] = tid;
        }

        #pragma omp barrier

        for (int i = 0; i < N; i++)
            assert(arr[i][0] == tid);
    }

    mtx_destroy(&memory_lock);

    return 0;
}
