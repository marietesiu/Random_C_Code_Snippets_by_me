#include <stdio.h>

 struct pArr {
    void *ptr;
    void *ptr1;
    void *ptr2;
};




int main()
{
    
    int x = 24;                             // Pointer to pointer void typecasting
    struct pArr pointer;
    pointer.ptr = &x;
    pointer.ptr = (int*) pointer.ptr;

    printf("%d\n", *(int*)pointer.ptr);




    return 0;
}

