#include <stdio.h>
#include <stdlib.h>

int main() {
   // Initially allocate memory for 2 integers
   int *ptr = malloc(2 * sizeof(int));
   ptr[0] = 5;
   ptr[1] = 10;

    // increase the size of the integer
   int *ptr_new = realloc(ptr,  3* sizeof(int));
 
    
   ptr_new[2] = 15;
   
    // Display the array
   for (int i = 0; i < 3; i++) {
      printf("%d ", ptr_new[i]);
   }

    ptr_new[1] = 3;
    //without next statement would cause to output 3 twice, 
    ptr[1] = 2;    
    // causes output 2 twice
    
    printf("\n%d\n", ptr_new[1]);
    printf("%d\n", ptr[1]);        // here it is observed how both share the same memory space, change on one acts on both
   
   free(ptr_new);
   return 0;
}
