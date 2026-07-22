#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static int data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16 };

struct LArr { 
    int *data;
    struct LArr *ptr;
};

int main(void)
{
    struct LArr *head = malloc(sizeof(struct LArr));

    size_t len = sizeof(data) / sizeof(data[0]);

    head->data = data;
    // do not malloc if not going to use memcpy, dis results in allocated memory leaks,
    //but doing this means changing either or changes both as they share the array address
    
    
    head->ptr = NULL;

    struct LArr *Node1 = malloc(sizeof(struct LArr));

    Node1->data = malloc(sizeof(int));
   

    Node1->data = data; 
    Node1->ptr = NULL;

    head->ptr = Node1;
    
    
    
    for (size_t i = 0; i < len; i++) 
    {
        printf("%d ", head->data[i]);
    }
    printf("\n");

    /* Print the second node */
    for (size_t i = 0; i < len; i++) 
    {
    printf("Node1 value %d: %d\n", i, head->ptr->data[i]);
    }
    /* Print addresses */
    printf("head  = %p\n", (void *)head);
    printf("Node1 = %p\n", (void *)head->ptr);

    return 0;
}
