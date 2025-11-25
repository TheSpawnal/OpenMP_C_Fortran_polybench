! Parallel nested loops - Common in scientific computing
PROGRAM nested_parallel_loops
    USE omp_lib
    IMPLICIT NONE
    INTEGER, PARAMETER :: nx = 1000, ny = 1000, nz = 100
    REAL(8), DIMENSION(nx, ny, nz) :: grid, result
    INTEGER :: i, j, k
    
    ! Initialize grid
    grid = 1.0d0
    result = 0.0d0
    
    ! COLLAPSE merges nested loops into single iteration space
!$OMP PARALLEL DO COLLAPSE(3) PRIVATE(i, j, k) SCHEDULE(DYNAMIC, 10)
    DO k = 2, nz-1
        DO j = 2, ny-1
            DO i = 2, nx-1
                ! 3D Laplacian stencil computation
                result(i, j, k) = (grid(i-1, j, k) + grid(i+1, j, k) + &
                                   grid(i, j-1, k) + grid(i, j+1, k) + &
                                   grid(i, j, k-1) + grid(i, j, k+1) - &
                                   6.0d0 * grid(i, j, k))
            END DO
        END DO
    END DO
!$OMP END PARALLEL DO
    
    PRINT *, 'Computation completed with', omp_get_max_threads(), 'threads'
    
END PROGRAM nested_parallel_loops

!use case : CFD simulations, finite difference methods, 3D grid computations, stencil operations.

