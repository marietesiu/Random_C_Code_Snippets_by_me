#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main()
{
    int s[] = {1, 2, 3, 4, 5, 6, 7};
    int x[] = {1, 2, 3, 4, 5, 6, 8};
    
    int r = memcmp(s, x, (int)sizeof(int) * 7);
    
    printf("%d\n", r); 
//memcmp takes in two variables, and the amount of bytes long to compare said variables, returning 0 for equal and -1 for not equal
   
    return r;
}
