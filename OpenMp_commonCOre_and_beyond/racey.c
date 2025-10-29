// program using tasks that will “randomly” generate one of two 
// strings:
// – I think race cars are fun
// – I think car races are fun
// • Hint: use tasks to print the indeterminate part of the output 
// (i.e. the “race” or “car” parts).    
// • This is called a “Race Condition”.  
// It occurs when the result of a program depends on how the OS 
// schedules the threads.
// •NOTE: A “data race” is when threads “race to update a shared variable”.  
// They produce race conditions.  Programs containing data races are 
// undefined (in OpenMP but also ANSI standards C++’11 and beyond).

#include <stdio.h>
#include <omp.h>

int main()
{ printf("I think");
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
                printf(" car");
            #pragma omp task
                printf(" race");
        }
    }
    printf("s");
    printf(" are fun!\n");
}