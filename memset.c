// C program to demonstrate working of memset()
#include <stdio.h>
#include <string.h>

int main()
{
	char str[50] = "GeeksForGeeks is for programming geeks.";
	printf("\nBefore memset(): %s\n", str);

	// Fill 8 characters starting from str[13] with '.'
	memset(str + 1, '3', 8); 
	/*no matter what you do, the very first value can never be replaced in this case, the G.
this is because if you do + 0, it simply starts before the first character, and if you do + 1 
it starts at the second character
*/
	printf("After memset():  %s", str);
	return 0;
}
