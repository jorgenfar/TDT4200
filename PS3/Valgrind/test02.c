#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
	char *mem = malloc(128);
	int i;
	for(i = 0; i < 32; i++)
		mem[i] = i;
	for(i = 0; i < 32; i++)
		mem[i+64] = mem[i];
	return 0;
}

//mem[i+64] = i; BEDRE
//mallocer for mye
//


	char *mem = malloc(128);
	int i;
	for(i = 0; i < 32; i++)
		mem[i] = i;
		mem[i+64] = i;
	for(i = 0; i < 32; i++)
		mem[i+64] = mem[i];
	return 0;