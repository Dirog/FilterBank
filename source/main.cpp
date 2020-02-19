#include <stdio.h>
#include "filterbank.h"

using namespace std;
int main(){
    
    float a[5];
    
   
    filterbank fb(1,2,3,4,5,a);
    
    unsigned *dims = fb.getOutDim();
    
    printf("N = %d\n", dims[0]);
    printf("C = %d\n", dims[1]);
    printf("F = %d\n", dims[2]);
    return 1;
}