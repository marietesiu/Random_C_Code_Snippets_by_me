#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main() {
    char *str1 = (char *)malloc(6); // declare dynamic memory
    strcpy(str1, "Geeks"); //assign value to said memory


    char str2[6] = ""; // secondary destination

    // Copies contents of str1 to str2
    memcpy(str2, str1, 6); 

    printf("str2 after memcpy: ");
    printf("%s",str2);
// print str2

    free((char *)str1);
    str1 = NULL;
    //free and unallocate str1, to be used again
    return 0;
    
    
    
}
