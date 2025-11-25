! Parallel sum with reduction-classic parallel pattern
PROGRAM parallel_reduction
    USE omp_lib
    IMPLICIT NONE
    INTEGER, PARAMETER :: n = 1000000
    REAL(8), DIMENSION(n) :: array
    REAL(8) :: total_sum
    INTEGER :: i
    
    ! Initialize array
    DO i = 1, n
        array(i) = DBLE(i)
    END DO
    
    total_sum = 0.0d0
    
!$OMP PARALLEL DO REDUCTION(+:total_sum) SCHEDULE(STATIC)
    DO i = 1, n
        total_sum = total_sum + array(i)**2
    END DO
!$OMP END PARALLEL DO
    
    PRINT *, 'Total sum:', total_sum
    
END PROGRAM parallel_reduction

!use case : Matrix operations, vector computations, accumulation operations where race conditions must be avoided.
