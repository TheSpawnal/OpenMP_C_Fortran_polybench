! Parallel loop in shuffle.f of Gafort.
! The code implements a parallel Fisher-Yates shuffle with ordered locking to prevent deadlocks when swapping array elements.


!$OMP PARALLEL PRIVATE(rand, iother, itemp, temp, my_cpu_id)
    my_cpu_id = 1
!$  my_cpu_id = omp_get_thread_num() + 1
    
!$OMP DO
    DO j = 1, npopsiz-1
        CALL ran3(1, rand, my_cpu_id, 0)
        iother = j + 1 + DINT(DBLE(npopsiz-j)*rand)
        
        ! Exclusive access to array elements.
        ! Ordered locking prevents deadlock.
        IF (j < iother) THEN
            CALL omp_set_lock(lck(j))
            CALL omp_set_lock(lck(iother))
        ELSE
            CALL omp_set_lock(lck(iother))
            CALL omp_set_lock(lck(j))
        END IF
        
        ! Swap parent chromosomes
        itemp(1:nchrome) = iparent(1:nchrome, iother)
        iparent(1:nchrome, iother) = iparent(1:nchrome, j)
        iparent(1:nchrome, j) = itemp(1:nchrome)
        
        ! Swap fitness values
        temp = fitness(iother)
        fitness(iother) = fitness(j)
        fitness(j) = temp
        
        ! Release locks in reverse order
        IF (j < iother) THEN
            CALL omp_unset_lock(lck(iother))
            CALL omp_unset_lock(lck(j))
        ELSE
            CALL omp_unset_lock(lck(j))
            CALL omp_unset_lock(lck(iother))
        END IF
    END DO
!$OMP END DO
!$OMP END PARALLEL
