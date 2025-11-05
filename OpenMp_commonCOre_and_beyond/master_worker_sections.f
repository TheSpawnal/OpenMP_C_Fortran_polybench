!usage:heterogeneous workloads, pipeline stages, independent tasks that can run concurrently (I/O + computation).

!different independent tasks executed by different threads
PROGRAM master_worker_sections
    USE omp_lib
    IMPLICIT NONE
    INTEGER :: matrix_a(100,100), matrix_b(100,100), matrix_c(100,100)
    REAL(8) :: stats_mean, stats_variance
    
!$OMP PARALLEL SECTIONS
    
!$OMP SECTION
    ! Task 1: Matrix initialization
    CALL initialize_matrix(matrix_a, 100, 100)
    PRINT *, 'Thread', omp_get_thread_num(), 'initialized matrix A'
    
!$OMP SECTION
    ! Task 2: Different matrix initialization
    CALL initialize_matrix(matrix_b, 100, 100)
    PRINT *, 'Thread', omp_get_thread_num(), 'initialized matrix B'
    
!$OMP SECTION
    ! Task 3: Compute statistics
    CALL compute_statistics(matrix_a, stats_mean, stats_variance)
    PRINT *, 'Thread', omp_get_thread_num(), 'computed statistics'
    
!$OMP SECTION
    ! Task 4: Independent I/O operation
    CALL write_checkpoint()
    PRINT *, 'Thread', omp_get_thread_num(), 'wrote checkpoint'
    
!$OMP END PARALLEL SECTIONS

END PROGRAM master_worker_sections
