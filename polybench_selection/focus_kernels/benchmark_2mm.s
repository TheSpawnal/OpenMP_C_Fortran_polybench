	.file	"benchmark_2mm.c"
# GNU C17 (Ubuntu 11.4.0-1ubuntu1~22.04.2) version 11.4.0 (x86_64-linux-gnu)
#	compiled by GNU C version 11.4.0, GMP version 6.2.1, MPFR version 4.1.0, MPC version 1.2.1, isl version isl-0.24-GMP

# GGC heuristics: --param ggc-min-expand=100 --param ggc-min-heapsize=131072
# options passed: -march=skylake -mmmx -mpopcnt -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mavx -mavx2 -mno-sse4a -mno-fma4 -mno-xop -mfma -mno-avx512f -mbmi -mbmi2 -maes -mpclmul -mno-avx512vl -mno-avx512bw -mno-avx512dq -mno-avx512cd -mno-avx512er -mno-avx512pf -mno-avx512vbmi -mno-avx512ifma -mno-avx5124vnniw -mno-avx5124fmaps -mno-avx512vpopcntdq -mno-avx512vbmi2 -mno-gfni -mno-vpclmulqdq -mno-avx512vnni -mno-avx512bitalg -mno-avx512bf16 -mno-avx512vp2intersect -mno-3dnow -madx -mabm -mno-cldemote -mclflushopt -mno-clwb -mno-clzero -mcx16 -mno-enqcmd -mf16c -mfsgsbase -mfxsr -mno-hle -msahf -mno-lwp -mlzcnt -mmovbe -mno-movdir64b -mno-movdiri -mno-mwaitx -mno-pconfig -mno-pku -mno-prefetchwt1 -mprfchw -mno-ptwrite -mno-rdpid -mrdrnd -mrdseed -mno-rtm -mno-serialize -mno-sgx -mno-sha -mno-shstk -mno-tbm -mno-tsxldtrk -mno-vaes -mno-waitpkg -mno-wbnoinvd -mxsave -mxsavec -mxsaveopt -mxsaves -mno-amx-tile -mno-amx-int8 -mno-amx-bf16 -mno-uintr -mno-hreset -mno-kl -mno-widekl -mno-avxvnni --param=l1-cache-size=32 --param=l1-cache-line-size=64 --param=l2-cache-size=6144 -mtune=skylake -O3 -fopenmp -fasynchronous-unwind-tables -fstack-protector-strong -fstack-clash-protection -fcf-protection
	.text
	.p2align 4
	.type	kernel_2mm_basic_parallel._omp_fn.1, @function
kernel_2mm_basic_parallel._omp_fn.1:
.LFB5553:
	.cfi_startproc
	endbr64	
	pushq	%r15	#
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14	#
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13	#
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12	#
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp	#
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx	#
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movq	%rdi, %rbx	# tmp150, .omp_data_i
	subq	$24, %rsp	#,
	.cfi_def_cfa_offset 80
	call	omp_get_num_threads@PLT	#
	movl	%eax, %ebp	# tmp151, _19
	call	omp_get_thread_num@PLT	#
	movl	%eax, %ecx	# tmp152, _20
	movl	32(%rbx), %eax	# *.omp_data_i_11(D).ni, *.omp_data_i_11(D).ni
	cltd
	idivl	%ebp	# _19
	cmpl	%edx, %ecx	# tt.7_2, _20
	jl	.L2	#,
.L9:
	imull	%eax, %ecx	# q.6_1, tmp136
	leal	(%rcx,%rdx), %ebp	#, i
	leal	(%rax,%rbp), %r12d	#, _26
	cmpl	%r12d, %ebp	# _26, i
	jge	.L17	#,
# benchmark_2mm.c:149:     #pragma omp parallel for
	movq	8(%rbx), %rax	# *.omp_data_i_11(D).C, C
	movq	24(%rbx), %r14	# *.omp_data_i_11(D).tmp, tmp
	movq	%rax, (%rsp)	# C, %sfp
	movl	40(%rbx), %eax	# *.omp_data_i_11(D).nl, nl
	movq	16(%rbx), %rdx	# *.omp_data_i_11(D).D, D
	vmovsd	(%rbx), %xmm1	# *.omp_data_i_11(D).beta, beta
	movl	36(%rbx), %r10d	# *.omp_data_i_11(D).nj, nj
	testl	%eax, %eax	# nl
	jle	.L17	#,
	movslq	%eax, %rbx	# nl, _82
	imull	%ebp, %eax	# i, tmp138
	movl	%ebp, %r13d	# i, ivtmp.146
	imull	%r10d, %r13d	# nj, ivtmp.146
	cltq
	leaq	(%rdx,%rax,8), %r9	#, ivtmp.144
	leal	-1(%r10), %eax	#, tmp143
	movq	%rax, 8(%rsp)	# tmp143, %sfp
	leaq	0(,%rbx,8), %rcx	#, _77
	leaq	8(%r14), %r15	#, tmp149
	.p2align 4,,10
	.p2align 3
.L5:
	movslq	%r13d, %rax	# ivtmp.146, _120
	leaq	(%r14,%rax,8), %r11	#, ivtmp.129
	movq	(%rsp), %r8	# %sfp, ivtmp.138
	addq	8(%rsp), %rax	# %sfp, tmp144
	leaq	(%r15,%rax,8), %rsi	#, _97
	xorl	%edi, %edi	# ivtmp.134
	.p2align 4,,10
	.p2align 3
.L4:
# benchmark_2mm.c:152:             double sum = beta * D[i*nl + j];
	vmulsd	(%r9,%rdi,8), %xmm1, %xmm0	# MEM[(double *)_89 + ivtmp.134_96 * 8], beta, sum
# benchmark_2mm.c:153:             for (int k = 0; k < nj; k++) {
	movq	%r8, %rdx	# ivtmp.138, ivtmp.130
	movq	%r11, %rax	# ivtmp.129, ivtmp.129
	testl	%r10d, %r10d	# nj
	jle	.L8	#,
	.p2align 4,,10
	.p2align 3
.L6:
# benchmark_2mm.c:154:                 sum += tmp[i*nj + k] * C[k*nl + j];
	vmovsd	(%rax), %xmm2	# MEM[(double *)_108], tmp198
# benchmark_2mm.c:153:             for (int k = 0; k < nj; k++) {
	addq	$8, %rax	#, ivtmp.129
# benchmark_2mm.c:154:                 sum += tmp[i*nj + k] * C[k*nl + j];
	vfmadd231sd	(%rdx), %xmm2, %xmm0	# MEM[(double *)_107], tmp198, sum
# benchmark_2mm.c:153:             for (int k = 0; k < nj; k++) {
	addq	%rcx, %rdx	# _77, ivtmp.130
	cmpq	%rax, %rsi	# ivtmp.129, _97
	jne	.L6	#,
.L8:
# benchmark_2mm.c:156:             D[i*nl + j] = sum;
	vmovsd	%xmm0, (%r9,%rdi,8)	# sum, MEM[(double *)_89 + ivtmp.134_96 * 8]
# benchmark_2mm.c:151:         for (int j = 0; j < nl; j++) {
	incq	%rdi	# ivtmp.134
	addq	$8, %r8	#, ivtmp.138
	cmpq	%rdi, %rbx	# ivtmp.134, _82
	jne	.L4	#,
	incl	%ebp	# i
	addq	%rcx, %r9	# _77, ivtmp.144
	addl	%r10d, %r13d	# nj, ivtmp.146
	cmpl	%ebp, %r12d	# i, _26
	jne	.L5	#,
.L17:
# benchmark_2mm.c:149:     #pragma omp parallel for
	addq	$24, %rsp	#,
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx	#
	.cfi_def_cfa_offset 48
	popq	%rbp	#
	.cfi_def_cfa_offset 40
	popq	%r12	#
	.cfi_def_cfa_offset 32
	popq	%r13	#
	.cfi_def_cfa_offset 24
	popq	%r14	#
	.cfi_def_cfa_offset 16
	popq	%r15	#
	.cfi_def_cfa_offset 8
	ret	
.L2:
	.cfi_restore_state
	incl	%eax	# q.6_1
# benchmark_2mm.c:149:     #pragma omp parallel for
	xorl	%edx, %edx	# tt.7_2
	jmp	.L9	#
	.cfi_endproc
.LFE5553:
	.size	kernel_2mm_basic_parallel._omp_fn.1, .-kernel_2mm_basic_parallel._omp_fn.1
	.p2align 4
	.type	kernel_2mm_basic_parallel._omp_fn.0, @function
kernel_2mm_basic_parallel._omp_fn.0:
.LFB5552:
	.cfi_startproc
	endbr64	
	pushq	%r15	#
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14	#
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13	#
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12	#
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp	#
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx	#
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movq	%rdi, %rbx	# tmp191, .omp_data_i
	subq	$56, %rsp	#,
	.cfi_def_cfa_offset 112
	call	omp_get_num_threads@PLT	#
	movl	%eax, %ebp	# tmp192, _20
	call	omp_get_thread_num@PLT	#
	movl	%eax, %ecx	# tmp193, _21
	movl	32(%rbx), %eax	# *.omp_data_i_11(D).ni, *.omp_data_i_11(D).ni
	cltd
	idivl	%ebp	# _20
	cmpl	%edx, %ecx	# tt.9_2, _21
	jl	.L21	#,
.L32:
	imull	%eax, %ecx	# q.8_1, tmp160
	movl	%eax, %edi	# q.8_1, q.8_1
	leal	(%rcx,%rdx), %esi	#, i
	addl	%esi, %edi	# i, q.8_1
	movl	%esi, 20(%rsp)	# i, %sfp
	movl	%edi, 28(%rsp)	# _27, %sfp
	movl	%esi, %eax	# i, i
	cmpl	%edi, %esi	# _27, i
	jge	.L37	#,
# benchmark_2mm.c:138:     #pragma omp parallel for
	movq	24(%rbx), %rsi	# *.omp_data_i_11(D).tmp, tmp
	movl	36(%rbx), %edi	# *.omp_data_i_11(D).nj, nj
	movq	%rsi, 32(%rsp)	# tmp, %sfp
	movq	16(%rbx), %r8	# *.omp_data_i_11(D).B, B
	movq	8(%rbx), %r9	# *.omp_data_i_11(D).A, A
	vmovsd	(%rbx), %xmm3	# *.omp_data_i_11(D).alpha, alpha
	movl	40(%rbx), %r13d	# *.omp_data_i_11(D).nk, nk
	testl	%edi, %edi	# nj
	jle	.L37	#,
	movl	%eax, %ebx	# i, ivtmp.187
	imull	%r13d, %eax	# nk, ivtmp.188
	imull	%edi, %ebx	# nj, ivtmp.187
	movl	%r13d, %r15d	# nk, niters_vector_mult_vf.157
	movl	%eax, %esi	# ivtmp.188, ivtmp.188
	movslq	%edi, %rax	# nj, _45
	movq	%rax, %rbp	# _45, _80
	movl	%ebx, 24(%rsp)	# ivtmp.187, %sfp
	movq	%rax, (%rsp)	# _45, %sfp
	leaq	0(,%rax,8), %rbx	#, _76
	movl	%r13d, %eax	# nk, bnd.156
	shrl	%eax	# bnd.156
	decl	%eax	# _57
	incq	%rax	# tmp184
	salq	$4, %rax	#, tmp184
	movq	%rax, 40(%rsp)	# tmp184, %sfp
	salq	$4, %rbp	#, _80
	andl	$-2, %r15d	#, niters_vector_mult_vf.157
	vxorpd	%xmm5, %xmm5, %xmm5	# sum
	vmovddup	%xmm3, %xmm4	# alpha, vect_cst__86
	.p2align 4,,10
	.p2align 3
.L27:
	movq	40(%rsp), %r11	# %sfp, _123
	movslq	%esi, %rax	# ivtmp.188, _36
	leaq	(%r9,%rax,8), %rax	#, ivtmp.174
	movq	%rax, 8(%rsp)	# ivtmp.174, %sfp
	movq	32(%rsp), %rcx	# %sfp, tmp
	addq	%rax, %r11	# ivtmp.174, _123
	movslq	24(%rsp), %rax	# %sfp, ivtmp.187
	movq	%r8, %r12	# B, ivtmp.181
	leaq	(%rcx,%rax,8), %r14	#, _48
	xorl	%r10d, %r10d	# ivtmp.177
	.p2align 4,,10
	.p2align 3
.L26:
	movl	%r10d, %ecx	# ivtmp.177, j
# benchmark_2mm.c:142:             for (int k = 0; k < nk; k++) {
	testl	%r13d, %r13d	# nk
	jle	.L39	#,
	cmpl	$1, %r13d	#, nk
	je	.L33	#,
	movq	8(%rsp), %rdx	# %sfp, ivtmp.174
	movq	%r12, %rax	# ivtmp.181, ivtmp.171
# benchmark_2mm.c:141:             double sum = 0.0;
	vmovsd	%xmm5, %xmm5, %xmm1	# sum, sum
	.p2align 4,,10
	.p2align 3
.L29:
# benchmark_2mm.c:143:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	vmulpd	(%rdx), %xmm4, %xmm2	# MEM <vector(2) double> [(double *)_4], vect_cst__86, vect__42.162
# benchmark_2mm.c:143:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	vmovsd	(%rax), %xmm0	# MEM[(double *)_19], MEM[(double *)_19]
	addq	$16, %rdx	#, ivtmp.174
	vmovhpd	(%rax,%rbx), %xmm0, %xmm0	# MEM[(double *)_19 + _76 * 1], MEM[(double *)_19], tmp170
	addq	%rbp, %rax	# _80, ivtmp.171
# benchmark_2mm.c:143:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	vmulpd	%xmm2, %xmm0, %xmm0	# vect__42.162, tmp170, vect__49.163
	vaddsd	%xmm0, %xmm1, %xmm1	# stmp_sum_50.164, sum, stmp_sum_50.164
# benchmark_2mm.c:143:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	vunpckhpd	%xmm0, %xmm0, %xmm0	# vect__49.163, stmp_sum_50.164
	vaddsd	%xmm1, %xmm0, %xmm1	# stmp_sum_50.164, stmp_sum_50.164, sum
	cmpq	%rdx, %r11	# ivtmp.174, _123
	jne	.L29	#,
# benchmark_2mm.c:142:             for (int k = 0; k < nk; k++) {
	movl	%r15d, %eax	# niters_vector_mult_vf.157, k
	cmpl	%r13d, %r15d	# nk, niters_vector_mult_vf.157
	je	.L25	#,
.L28:
# benchmark_2mm.c:143:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	movl	%edi, %edx	# nj, tmp176
	imull	%eax, %edx	# k, tmp176
# benchmark_2mm.c:143:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	addl	%esi, %eax	# ivtmp.188, tmp179
	cltq
# benchmark_2mm.c:143:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	vmulsd	(%r9,%rax,8), %xmm3, %xmm0	# *_111, alpha, tmp181
# benchmark_2mm.c:143:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	addl	%ecx, %edx	# j, tmp177
	movslq	%edx, %rdx	# tmp177, tmp178
# benchmark_2mm.c:143:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	vfmadd231sd	(%r8,%rdx,8), %xmm0, %xmm1	# *_104, tmp181, sum
.L25:
# benchmark_2mm.c:145:             tmp[i*nj + j] = sum;
	vmovsd	%xmm1, (%r14,%r10,8)	# sum, MEM[(double *)_48 + ivtmp.177_122 * 8]
# benchmark_2mm.c:140:         for (int j = 0; j < nj; j++) {
	incq	%r10	# ivtmp.177
	addq	$8, %r12	#, ivtmp.181
	cmpq	%r10, (%rsp)	# ivtmp.177, %sfp
	jne	.L26	#,
	incl	20(%rsp)	# %sfp
	addl	%edi, 24(%rsp)	# nj, %sfp
	addl	%r13d, %esi	# nk, ivtmp.188
	movl	20(%rsp), %eax	# %sfp, i
	cmpl	%eax, 28(%rsp)	# i, %sfp
	jne	.L27	#,
.L37:
# benchmark_2mm.c:138:     #pragma omp parallel for
	addq	$56, %rsp	#,
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx	#
	.cfi_def_cfa_offset 48
	popq	%rbp	#
	.cfi_def_cfa_offset 40
	popq	%r12	#
	.cfi_def_cfa_offset 32
	popq	%r13	#
	.cfi_def_cfa_offset 24
	popq	%r14	#
	.cfi_def_cfa_offset 16
	popq	%r15	#
	.cfi_def_cfa_offset 8
	ret	
	.p2align 4,,10
	.p2align 3
.L39:
	.cfi_restore_state
# benchmark_2mm.c:141:             double sum = 0.0;
	vmovsd	%xmm5, %xmm5, %xmm1	# sum, sum
	jmp	.L25	#
.L33:
# benchmark_2mm.c:142:             for (int k = 0; k < nk; k++) {
	xorl	%eax, %eax	# k
# benchmark_2mm.c:141:             double sum = 0.0;
	vmovsd	%xmm5, %xmm5, %xmm1	# sum, sum
	jmp	.L28	#
.L21:
	incl	%eax	# q.8_1
# benchmark_2mm.c:138:     #pragma omp parallel for
	xorl	%edx, %edx	# tt.9_2
	jmp	.L32	#
	.cfi_endproc
.LFE5552:
	.size	kernel_2mm_basic_parallel._omp_fn.0, .-kernel_2mm_basic_parallel._omp_fn.0
	.p2align 4
	.globl	kernel_2mm_basic_parallel
	.type	kernel_2mm_basic_parallel, @function
kernel_2mm_basic_parallel:
.LFB5546:
	.cfi_startproc
	endbr64	
	pushq	%r15	#
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	vmovd	%edi, %xmm3	# tmp109, tmp109
# benchmark_2mm.c:138:     #pragma omp parallel for
	vmovq	%r8, %xmm4	# tmp115, tmp115
# benchmark_2mm.c:137:                                double *tmp) {
	pushq	%r14	#
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	vmovsd	%xmm0, %xmm0, %xmm2	# alpha, tmp113
	leaq	kernel_2mm_basic_parallel._omp_fn.0(%rip), %rdi	#, tmp104
	pushq	%r13	#
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	vpinsrd	$1, %esi, %xmm3, %xmm0	# tmp110, tmp109, tmp101
	pushq	%r12	#
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp	#
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	vmovq	%xmm1, %rbp	# tmp114, beta
# benchmark_2mm.c:138:     #pragma omp parallel for
	vpinsrq	$1, %r9, %xmm4, %xmm1	# tmp116, tmp115, tmp102
# benchmark_2mm.c:137:                                double *tmp) {
	pushq	%rbx	#
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movl	%ecx, %ebx	# tmp112, nl
	xorl	%ecx, %ecx	#
	subq	$88, %rsp	#,
	.cfi_def_cfa_offset 144
# benchmark_2mm.c:137:                                double *tmp) {
	movq	160(%rsp), %r15	# tmp, tmp
	movq	144(%rsp), %r12	# C, C
	leaq	16(%rsp), %r14	#, tmp103
	movq	%fs:40, %rax	# MEM[(<address-space-1> long unsigned int *)40B], tmp117
	movq	%rax, 72(%rsp)	# tmp117, D.37918
	xorl	%eax, %eax	# tmp117
	movq	152(%rsp), %r13	# D, D
# benchmark_2mm.c:138:     #pragma omp parallel for
	movl	%edx, 56(%rsp)	# tmp111, MEM[(struct .omp_data_s.0 *)_31].nk
	movq	%r14, %rsi	# tmp103,
	xorl	%edx, %edx	#
	vmovdqu	%xmm1, 24(%rsp)	# tmp102, MEM <vector(2) long unsigned int> [(double * *)_31]
	vmovq	%xmm0, 48(%rsp)	# tmp101, MEM <vector(2) int> [(int *)_31]
	vmovq	%xmm0, 8(%rsp)	# tmp101, %sfp
	movq	%r15, 40(%rsp)	# tmp, MEM[(struct .omp_data_s.0 *)_31].tmp
	vmovsd	%xmm2, 16(%rsp)	# tmp113, MEM[(struct .omp_data_s.0 *)_31].alpha
	call	GOMP_parallel@PLT	#
# benchmark_2mm.c:149:     #pragma omp parallel for
	vmovq	8(%rsp), %xmm0	# %sfp, tmp101
	vmovq	%r12, %xmm5	# C, C
	vpinsrq	$1, %r13, %xmm5, %xmm1	# D, C, tmp105
	xorl	%ecx, %ecx	#
	xorl	%edx, %edx	#
	movq	%r14, %rsi	# tmp103,
	leaq	kernel_2mm_basic_parallel._omp_fn.1(%rip), %rdi	#, tmp107
	movq	%r15, 40(%rsp)	# tmp, MEM[(struct .omp_data_s.1 *)_31].tmp
	movq	%rbp, 16(%rsp)	# beta, MEM[(struct .omp_data_s.1 *)_31].beta
	movl	%ebx, 56(%rsp)	# nl, MEM[(struct .omp_data_s.1 *)_31].nl
	vmovdqu	%xmm1, 24(%rsp)	# tmp105, MEM <vector(2) long unsigned int> [(double * *)_31]
	vmovq	%xmm0, 48(%rsp)	# tmp101, MEM <vector(2) int> [(int *)_31]
	call	GOMP_parallel@PLT	#
# benchmark_2mm.c:159: }
	movq	72(%rsp), %rax	# D.37918, tmp118
	subq	%fs:40, %rax	# MEM[(<address-space-1> long unsigned int *)40B], tmp118
	jne	.L44	#,
	addq	$88, %rsp	#,
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx	#
	.cfi_def_cfa_offset 48
	popq	%rbp	#
	.cfi_def_cfa_offset 40
	popq	%r12	#
	.cfi_def_cfa_offset 32
	popq	%r13	#
	.cfi_def_cfa_offset 24
	popq	%r14	#
	.cfi_def_cfa_offset 16
	popq	%r15	#
	.cfi_def_cfa_offset 8
	ret	
.L44:
	.cfi_restore_state
	call	__stack_chk_fail@PLT	#
	.cfi_endproc
.LFE5546:
	.size	kernel_2mm_basic_parallel, .-kernel_2mm_basic_parallel
	.p2align 4
	.globl	kernel_2mm_collapsed
	.type	kernel_2mm_collapsed, @function
kernel_2mm_collapsed:
.LFB5547:
	.cfi_startproc
	endbr64	
	pushq	%r15	#
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	vmovd	%edi, %xmm3	# tmp109, tmp109
# benchmark_2mm.c:166:     #pragma omp parallel for collapse(2)
	vmovq	%r8, %xmm4	# tmp115, tmp115
# benchmark_2mm.c:165:                          double *tmp) {
	pushq	%r14	#
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	vmovsd	%xmm0, %xmm0, %xmm2	# alpha, tmp113
	leaq	kernel_2mm_collapsed._omp_fn.0(%rip), %rdi	#, tmp104
	pushq	%r13	#
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	vpinsrd	$1, %esi, %xmm3, %xmm0	# tmp110, tmp109, tmp101
	pushq	%r12	#
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp	#
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	vmovq	%xmm1, %rbp	# tmp114, beta
# benchmark_2mm.c:166:     #pragma omp parallel for collapse(2)
	vpinsrq	$1, %r9, %xmm4, %xmm1	# tmp116, tmp115, tmp102
# benchmark_2mm.c:165:                          double *tmp) {
	pushq	%rbx	#
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movl	%ecx, %ebx	# tmp112, nl
	xorl	%ecx, %ecx	#
	subq	$88, %rsp	#,
	.cfi_def_cfa_offset 144
# benchmark_2mm.c:165:                          double *tmp) {
	movq	160(%rsp), %r15	# tmp, tmp
	movq	144(%rsp), %r12	# C, C
	leaq	16(%rsp), %r14	#, tmp103
	movq	%fs:40, %rax	# MEM[(<address-space-1> long unsigned int *)40B], tmp117
	movq	%rax, 72(%rsp)	# tmp117, D.37932
	xorl	%eax, %eax	# tmp117
	movq	152(%rsp), %r13	# D, D
# benchmark_2mm.c:166:     #pragma omp parallel for collapse(2)
	movl	%edx, 56(%rsp)	# tmp111, MEM[(struct .omp_data_s.10 *)_31].nk
	movq	%r14, %rsi	# tmp103,
	xorl	%edx, %edx	#
	vmovdqu	%xmm1, 24(%rsp)	# tmp102, MEM <vector(2) long unsigned int> [(double * *)_31]
	vmovq	%xmm0, 48(%rsp)	# tmp101, MEM <vector(2) int> [(int *)_31]
	vmovq	%xmm0, 8(%rsp)	# tmp101, %sfp
	movq	%r15, 40(%rsp)	# tmp, MEM[(struct .omp_data_s.10 *)_31].tmp
	vmovsd	%xmm2, 16(%rsp)	# tmp113, MEM[(struct .omp_data_s.10 *)_31].alpha
	call	GOMP_parallel@PLT	#
# benchmark_2mm.c:177:     #pragma omp parallel for collapse(2)
	vmovq	8(%rsp), %xmm0	# %sfp, tmp101
	vmovq	%r12, %xmm5	# C, C
	vpinsrq	$1, %r13, %xmm5, %xmm1	# D, C, tmp105
	xorl	%ecx, %ecx	#
	xorl	%edx, %edx	#
	movq	%r14, %rsi	# tmp103,
	leaq	kernel_2mm_collapsed._omp_fn.1(%rip), %rdi	#, tmp107
	movq	%r15, 40(%rsp)	# tmp, MEM[(struct .omp_data_s.11 *)_31].tmp
	movq	%rbp, 16(%rsp)	# beta, MEM[(struct .omp_data_s.11 *)_31].beta
	movl	%ebx, 56(%rsp)	# nl, MEM[(struct .omp_data_s.11 *)_31].nl
	vmovdqu	%xmm1, 24(%rsp)	# tmp105, MEM <vector(2) long unsigned int> [(double * *)_31]
	vmovq	%xmm0, 48(%rsp)	# tmp101, MEM <vector(2) int> [(int *)_31]
	call	GOMP_parallel@PLT	#
# benchmark_2mm.c:187: }
	movq	72(%rsp), %rax	# D.37932, tmp118
	subq	%fs:40, %rax	# MEM[(<address-space-1> long unsigned int *)40B], tmp118
	jne	.L49	#,
	addq	$88, %rsp	#,
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx	#
	.cfi_def_cfa_offset 48
	popq	%rbp	#
	.cfi_def_cfa_offset 40
	popq	%r12	#
	.cfi_def_cfa_offset 32
	popq	%r13	#
	.cfi_def_cfa_offset 24
	popq	%r14	#
	.cfi_def_cfa_offset 16
	popq	%r15	#
	.cfi_def_cfa_offset 8
	ret	
.L49:
	.cfi_restore_state
	call	__stack_chk_fail@PLT	#
	.cfi_endproc
.LFE5547:
	.size	kernel_2mm_collapsed, .-kernel_2mm_collapsed
	.p2align 4
	.globl	kernel_2mm_tiled
	.type	kernel_2mm_tiled, @function
kernel_2mm_tiled:
.LFB5548:
	.cfi_startproc
	endbr64	
	pushq	%r15	#
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	vmovd	%edi, %xmm3	# tmp111, tmp111
	vpinsrd	$1, %esi, %xmm3, %xmm2	# tmp112, tmp111, tmp101
	pushq	%r14	#
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	leaq	kernel_2mm_tiled._omp_fn.0(%rip), %rdi	#, tmp103
	movq	%r9, %r14	# tmp118, B
	pushq	%r13	#
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	movl	%edx, %r13d	# tmp113, nk
	xorl	%edx, %edx	#
	pushq	%r12	#
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp	#
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	movl	%ecx, %ebp	# tmp114, nl
	xorl	%ecx, %ecx	#
	pushq	%rbx	#
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$120, %rsp	#,
	.cfi_def_cfa_offset 176
# benchmark_2mm.c:193:                      double *tmp) {
	movq	176(%rsp), %rax	# C, C
	movq	%r8, 24(%rsp)	# tmp117, %sfp
	movq	%rax, 32(%rsp)	# C, %sfp
	movq	192(%rsp), %r12	# tmp, tmp
	leaq	48(%rsp), %rbx	#, tmp102
	vmovsd	%xmm0, 8(%rsp)	# tmp115, %sfp
	vmovsd	%xmm1, 16(%rsp)	# tmp116, %sfp
	movq	%rbx, %rsi	# tmp102,
	movq	%fs:40, %rax	# MEM[(<address-space-1> long unsigned int *)40B], tmp119
	movq	%rax, 104(%rsp)	# tmp119, D.37947
	xorl	%eax, %eax	# tmp119
	movq	184(%rsp), %r15	# D, D
# benchmark_2mm.c:197:     #pragma omp parallel for collapse(2)
	vmovq	%xmm2, 40(%rsp)	# tmp101, %sfp
	vmovq	%xmm2, 56(%rsp)	# tmp101, MEM <vector(2) int> [(int *)_38]
	movq	%r12, 48(%rsp)	# tmp, MEM[(struct .omp_data_s.32 *)_38].tmp
	call	GOMP_parallel@PLT	#
# benchmark_2mm.c:203:     #pragma omp parallel for collapse(2) schedule(static)
	vmovq	24(%rsp), %xmm4	# %sfp, A
	vmovsd	8(%rsp), %xmm5	# %sfp, alpha
	vpinsrq	$1, %r14, %xmm4, %xmm0	# B, A, tmp104
	movq	40(%rsp), %r14	# %sfp, tmp101
	xorl	%ecx, %ecx	#
	xorl	%edx, %edx	#
	movq	%rbx, %rsi	# tmp102,
	leaq	kernel_2mm_tiled._omp_fn.1(%rip), %rdi	#, tmp106
	vmovdqu	%xmm0, 56(%rsp)	# tmp104, MEM <vector(2) long unsigned int> [(double * *)_38]
	movq	%r12, 72(%rsp)	# tmp, MEM[(struct .omp_data_s.33 *)_38].tmp
	movl	%r13d, 88(%rsp)	# nk, MEM[(struct .omp_data_s.33 *)_38].nk
	movq	%r14, 80(%rsp)	# tmp101, MEM <vector(2) int> [(int *)_38]
	vmovsd	%xmm5, 48(%rsp)	# alpha, MEM[(struct .omp_data_s.33 *)_38].alpha
	call	GOMP_parallel@PLT	#
# benchmark_2mm.c:225:     #pragma omp parallel for collapse(2) schedule(static)
	vmovq	32(%rsp), %xmm6	# %sfp, C
	vmovsd	16(%rsp), %xmm7	# %sfp, beta
	vpinsrq	$1, %r15, %xmm6, %xmm0	# D, C, tmp107
	xorl	%ecx, %ecx	#
	xorl	%edx, %edx	#
	movq	%rbx, %rsi	# tmp102,
	leaq	kernel_2mm_tiled._omp_fn.2(%rip), %rdi	#, tmp109
	movq	%r12, 72(%rsp)	# tmp, MEM[(struct .omp_data_s.34 *)_38].tmp
	movl	%ebp, 88(%rsp)	# nl, MEM[(struct .omp_data_s.34 *)_38].nl
	movq	%r14, 80(%rsp)	# tmp101, MEM <vector(2) int> [(int *)_38]
	vmovdqu	%xmm0, 56(%rsp)	# tmp107, MEM <vector(2) long unsigned int> [(double * *)_38]
	vmovsd	%xmm7, 48(%rsp)	# beta, MEM[(struct .omp_data_s.34 *)_38].beta
	call	GOMP_parallel@PLT	#
# benchmark_2mm.c:243: }
	movq	104(%rsp), %rax	# D.37947, tmp120
	subq	%fs:40, %rax	# MEM[(<address-space-1> long unsigned int *)40B], tmp120
	jne	.L54	#,
	addq	$120, %rsp	#,
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx	#
	.cfi_def_cfa_offset 48
	popq	%rbp	#
	.cfi_def_cfa_offset 40
	popq	%r12	#
	.cfi_def_cfa_offset 32
	popq	%r13	#
	.cfi_def_cfa_offset 24
	popq	%r14	#
	.cfi_def_cfa_offset 16
	popq	%r15	#
	.cfi_def_cfa_offset 8
	ret	
.L54:
	.cfi_restore_state
	call	__stack_chk_fail@PLT	#
	.cfi_endproc
.LFE5548:
	.size	kernel_2mm_tiled, .-kernel_2mm_tiled
	.p2align 4
	.globl	kernel_2mm_tasks
	.type	kernel_2mm_tasks, @function
kernel_2mm_tasks:
.LFB5550:
	.cfi_startproc
	endbr64	
	pushq	%rbp	#
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
# benchmark_2mm.c:298:     #pragma omp parallel
	vunpcklpd	%xmm1, %xmm0, %xmm1	# tmp116, tmp115, tmp101
	vmovq	%r8, %xmm3	# tmp117, tmp117
# benchmark_2mm.c:295:                      double *tmp) {
	movq	%rsp, %rbp	#,
	.cfi_def_cfa_register 6
	andq	$-32, %rsp	#,
	subq	$96, %rsp	#,
# benchmark_2mm.c:298:     #pragma omp parallel
	vmovq	16(%rbp), %xmm2	# C, tmp122
# benchmark_2mm.c:295:                      double *tmp) {
	movq	%fs:40, %rax	# MEM[(<address-space-1> long unsigned int *)40B], tmp119
	movq	%rax, 88(%rsp)	# tmp119, D.37957
	xorl	%eax, %eax	# tmp119
# benchmark_2mm.c:298:     #pragma omp parallel
	vmovapd	%xmm1, (%rsp)	# tmp101, MEM <vector(2) double> [(double *)&.omp_data_o.91]
	vpinsrq	$1, 24(%rbp), %xmm2, %xmm1	# D, tmp122, tmp103
	vpinsrq	$1, %r9, %xmm3, %xmm0	# tmp118, tmp117, tmp104
	vinserti128	$0x1, %xmm1, %ymm0, %ymm0	# tmp103, tmp104, tmp102
	vmovd	%edx, %xmm4	# tmp113, tmp113
	vmovd	%edi, %xmm5	# tmp111, tmp111
	vpinsrd	$1, %ecx, %xmm4, %xmm1	# tmp114, tmp113, tmp106
	movq	32(%rbp), %rax	# tmp, tmp
	vmovdqu	%ymm0, 16(%rsp)	# tmp102, MEM <vector(4) long unsigned int> [(double * *)&.omp_data_o.91 + 16B]
	vpinsrd	$1, %esi, %xmm5, %xmm0	# tmp112, tmp111, tmp107
	vpunpcklqdq	%xmm1, %xmm0, %xmm0	# tmp106, tmp107, tmp105
	movq	%rsp, %rsi	#, tmp108
	xorl	%ecx, %ecx	#
	xorl	%edx, %edx	#
	leaq	kernel_2mm_tasks._omp_fn.0(%rip), %rdi	#, tmp109
	movq	%rax, 48(%rsp)	# tmp, .omp_data_o.91.tmp
	vmovdqu	%xmm0, 56(%rsp)	# tmp105, MEM <vector(4) int> [(int *)&.omp_data_o.91 + 56B]
	vzeroupper
	call	GOMP_parallel@PLT	#
# benchmark_2mm.c:346: }
	movq	88(%rsp), %rax	# D.37957, tmp120
	subq	%fs:40, %rax	# MEM[(<address-space-1> long unsigned int *)40B], tmp120
	jne	.L59	#,
	leave	
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret	
.L59:
	.cfi_restore_state
	call	__stack_chk_fail@PLT	#
	.cfi_endproc
.LFE5550:
	.size	kernel_2mm_tasks, .-kernel_2mm_tasks
	.p2align 4
	.type	kernel_2mm_collapsed._omp_fn.0, @function
kernel_2mm_collapsed._omp_fn.0:
.LFB5554:
	.cfi_startproc
	endbr64	
	pushq	%rbp	#
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp	#,
	.cfi_def_cfa_register 6
	pushq	%r15	#
	pushq	%r14	#
	pushq	%r13	#
	pushq	%r12	#
	pushq	%rbx	#
	andq	$-32, %rsp	#,
	subq	$32, %rsp	#,
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
# benchmark_2mm.c:166:     #pragma omp parallel for collapse(2)
	movl	32(%rdi), %r13d	# *.omp_data_i_11(D).ni, ni
	movl	36(%rdi), %r12d	# *.omp_data_i_11(D).nj, nj
	testl	%r13d, %r13d	# ni
	jle	.L94	#,
	testl	%r12d, %r12d	# nj
	jle	.L94	#,
	imull	%r12d, %r13d	# nj, tmp236
	movq	%rdi, %rbx	# tmp311, .omp_data_i
	call	omp_get_num_threads@PLT	#
	movl	%eax, %r14d	# tmp312, _24
	call	omp_get_thread_num@PLT	#
	movl	%eax, %ecx	# tmp313, _27
	xorl	%edx, %edx	# tt.30_2
	movl	%r13d, %eax	# tmp236, tmp236
	divl	%r14d	# _24
	movl	%eax, %r13d	# tmp236, q.29_1
	cmpl	%edx, %ecx	# tt.30_2, _27
	jb	.L63	#,
.L83:
	imull	%r13d, %ecx	# q.29_1, tmp239
	leal	(%rcx,%rdx), %eax	#, _33
	leal	0(%r13,%rax), %edx	#, tmp240
	cmpl	%edx, %eax	# tmp240, _33
	jnb	.L94	#,
	movq	24(%rbx), %rdi	# *.omp_data_i_11(D).tmp, tmp
	xorl	%edx, %edx	# j
	movq	%rdi, 24(%rsp)	# tmp, %sfp
	movq	8(%rbx), %r8	# *.omp_data_i_11(D).A, A
	movq	16(%rbx), %rdi	# *.omp_data_i_11(D).B, B
	vmovsd	(%rbx), %xmm4	# *.omp_data_i_11(D).alpha, alpha
	divl	%r12d	# nj
	movl	40(%rbx), %ebx	# *.omp_data_i_11(D).nk, nk
	cmpl	$1, %r12d	#, nj
	jne	.L96	#,
	movl	%ebx, %ecx	# nk, bnd.239
	shrl	$2, %ecx	#, bnd.239
	leal	-1(%rcx), %r10d	#, tmp245
	leal	-1(%rbx), %esi	#, _76
	incq	%r10	# tmp246
	movl	%ebx, %r15d	# nk, niters_vector_mult_vf.240
	movl	%esi, 20(%rsp)	# _76, %sfp
	decl	%r13d	# _160
	salq	$5, %r10	#, _98
	andl	$-4, %r15d	#, niters_vector_mult_vf.240
	xorl	%r14d, %r14d	# ivtmp.284
	vxorpd	%xmm5, %xmm5, %xmm5	# sum
# benchmark_2mm.c:171:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	vmovddup	%xmm4, %xmm7	# alpha, tmp302
	vbroadcastsd	%xmm4, %ymm6	# alpha, vect_cst__141
# benchmark_2mm.c:170:             for (int k = 0; k < nk; k++) {
	testl	%ebx, %ebx	# nk
	jle	.L84	#,
	.p2align 4,,10
	.p2align 3
.L97:
# benchmark_2mm.c:171:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	movl	%ebx, %esi	# nk, _132
	imull	%eax, %esi	# i, _132
	cmpl	$2, 20(%rsp)	#, %sfp
	jbe	.L85	#,
	movslq	%esi, %rcx	# _132, _132
	leaq	(%r8,%rcx,8), %r11	#, vectp.243
	movslq	%edx, %rcx	# j, j
	leaq	(%rdi,%rcx,8), %r9	#, vectp.247
# benchmark_2mm.c:169:             double sum = 0.0;
	vmovsd	%xmm5, %xmm5, %xmm0	# sum, sum
# benchmark_2mm.c:171:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	xorl	%ecx, %ecx	# ivtmp.279
	.p2align 4,,10
	.p2align 3
.L69:
# benchmark_2mm.c:171:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	vmulpd	(%r11,%rcx), %ymm6, %ymm1	# MEM <vector(4) double> [(double *)vectp.243_30 + ivtmp.279_123 * 1], vect_cst__141, vect__124.245
# benchmark_2mm.c:171:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	vmulpd	(%r9,%rcx), %ymm1, %ymm2	# MEM <vector(4) double> [(double *)vectp.247_143 + ivtmp.279_123 * 1], vect__124.245, vect__117.249
	addq	$32, %rcx	#, ivtmp.279
	vaddsd	%xmm2, %xmm0, %xmm0	# stmp_sum_116.250, sum, stmp_sum_116.250
	vunpckhpd	%xmm2, %xmm2, %xmm3	# tmp255, stmp_sum_116.250
	vextractf128	$0x1, %ymm2, %xmm1	# vect__117.249, tmp257
	vaddsd	%xmm3, %xmm0, %xmm0	# stmp_sum_116.250, stmp_sum_116.250, stmp_sum_116.250
# benchmark_2mm.c:171:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	vaddsd	%xmm1, %xmm0, %xmm0	# stmp_sum_116.250, stmp_sum_116.250, stmp_sum_116.250
	vunpckhpd	%xmm1, %xmm1, %xmm1	# tmp257, stmp_sum_116.250
	vaddsd	%xmm1, %xmm0, %xmm0	# stmp_sum_116.250, stmp_sum_116.250, sum
	cmpq	%rcx, %r10	# ivtmp.279, _98
	jne	.L69	#,
	cmpl	%ebx, %r15d	# nk, niters_vector_mult_vf.240
	je	.L67	#,
	movl	%r15d, %r9d	# niters_vector_mult_vf.240,
# benchmark_2mm.c:170:             for (int k = 0; k < nk; k++) {
	movl	%r15d, %ecx	# niters_vector_mult_vf.240, tmp.254
.L68:
	movl	%ebx, %r12d	# nk, niters.251
	subl	%r9d, %r12d	# _67, niters.251
	cmpl	$1, %r12d	#, niters.251
	je	.L71	#,
# benchmark_2mm.c:171:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	movslq	%esi, %r11	# _132, _132
	addq	%r9, %r11	# _191, tmp262
# benchmark_2mm.c:171:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	vmulpd	(%r8,%r11,8), %xmm7, %xmm1	# MEM <vector(2) double> [(double *)vectp.256_189], tmp302, vect__87.258
# benchmark_2mm.c:171:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	movslq	%edx, %r11	# j, j
	addq	%r9, %r11	# _191, tmp267
	movl	%r12d, %r9d	# niters.251, niters_vector_mult_vf.253
	andl	$-2, %r9d	#, niters_vector_mult_vf.253
# benchmark_2mm.c:171:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	vmulpd	(%rdi,%r11,8), %xmm1, %xmm1	# MEM <vector(2) double> [(double *)vectp.260_199], vect__87.258, vect__81.262
	addl	%r9d, %ecx	# niters_vector_mult_vf.253, tmp.254
	vaddsd	%xmm1, %xmm0, %xmm0	# stmp_sum_80.263, sum, stmp_sum_80.263
# benchmark_2mm.c:171:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	vunpckhpd	%xmm1, %xmm1, %xmm1	# vect__81.262, stmp_sum_80.263
	vaddsd	%xmm0, %xmm1, %xmm0	# stmp_sum_80.263, stmp_sum_80.263, sum
	cmpl	%r9d, %r12d	# niters_vector_mult_vf.253, niters.251
	je	.L67	#,
.L71:
# benchmark_2mm.c:171:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	addl	%ecx, %esi	# tmp.254, tmp270
	movslq	%esi, %rsi	# tmp270, tmp271
# benchmark_2mm.c:171:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	vmulsd	(%r8,%rsi,8), %xmm4, %xmm1	# *_168, alpha, tmp272
# benchmark_2mm.c:171:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	addl	%edx, %ecx	# j, tmp273
	movslq	%ecx, %rcx	# tmp273, tmp274
# benchmark_2mm.c:171:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	vfmadd231sd	(%rdi,%rcx,8), %xmm1, %xmm0	# *_174, tmp272, sum
.L67:
# benchmark_2mm.c:173:             tmp[i*nj + j] = sum;
	movq	24(%rsp), %rsi	# %sfp, tmp
# benchmark_2mm.c:173:             tmp[i*nj + j] = sum;
	addl	%eax, %edx	# i, tmp275
	movslq	%edx, %rdx	# tmp275, tmp276
# benchmark_2mm.c:173:             tmp[i*nj + j] = sum;
	vmovsd	%xmm0, (%rsi,%rdx,8)	# sum, *_108
	cmpl	%r13d, %r14d	# _160, ivtmp.284
	je	.L93	#,
	incl	%eax	# i
	incl	%r14d	# ivtmp.284
	xorl	%edx, %edx	# j
# benchmark_2mm.c:170:             for (int k = 0; k < nk; k++) {
	testl	%ebx, %ebx	# nk
	jg	.L97	#,
.L84:
# benchmark_2mm.c:169:             double sum = 0.0;
	vmovsd	%xmm5, %xmm5, %xmm0	# sum, sum
	jmp	.L67	#
.L93:
	vzeroupper
.L94:
# benchmark_2mm.c:166:     #pragma omp parallel for collapse(2)
	leaq	-40(%rbp), %rsp	#,
	popq	%rbx	#
	popq	%r12	#
	popq	%r13	#
	popq	%r14	#
	popq	%r15	#
	popq	%rbp	#
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret	
.L85:
	.cfi_restore_state
# benchmark_2mm.c:171:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	xorl	%r9d, %r9d	#
# benchmark_2mm.c:170:             for (int k = 0; k < nk; k++) {
	xorl	%ecx, %ecx	# tmp.254
# benchmark_2mm.c:169:             double sum = 0.0;
	vmovsd	%xmm5, %xmm5, %xmm0	# sum, sum
	jmp	.L68	#
.L63:
	incl	%r13d	# q.29_1
# benchmark_2mm.c:166:     #pragma omp parallel for collapse(2)
	xorl	%edx, %edx	# tt.30_2
	jmp	.L83	#
.L96:
	movl	%ebx, %ecx	# nk, bnd.265
	shrl	%ecx	# bnd.265
	decl	%ecx	# _89
	incq	%rcx	# tmp308
	leal	-1(%r13), %esi	#, _119
	salq	$4, %rcx	#, tmp308
	movslq	%r12d, %r11	# nj, _258
	movl	%esi, 20(%rsp)	# _119, %sfp
	movq	%rcx, 8(%rsp)	# tmp308, %sfp
	movq	%r11, %r13	# _258, _259
	movl	%ebx, %r15d	# nk, niters_vector_mult_vf.266
	salq	$4, %r13	#, _259
	salq	$3, %r11	#, _263
	andl	$-2, %r15d	#, niters_vector_mult_vf.266
# benchmark_2mm.c:166:     #pragma omp parallel for collapse(2)
	xorl	%r14d, %r14d	# ivtmp.298
	vxorpd	%xmm5, %xmm5, %xmm5	# sum
	vmovddup	%xmm4, %xmm3	# alpha, vect_cst__253
	.p2align 4,,10
	.p2align 3
.L66:
# benchmark_2mm.c:170:             for (int k = 0; k < nk; k++) {
	testl	%ebx, %ebx	# nk
	jle	.L98	#,
# benchmark_2mm.c:171:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	movl	%eax, %ecx	# i, _49
	imull	%ebx, %ecx	# nk, _49
	cmpl	$1, %ebx	#, nk
	je	.L86	#,
	movq	8(%rsp), %r10	# %sfp, tmp309
	movslq	%ecx, %rsi	# _49, _49
	leaq	(%r8,%rsi,8), %rsi	#, ivtmp.290
	movslq	%edx, %r9	# j, j
	leaq	(%rdi,%r9,8), %r9	#, ivtmp.293
	addq	%rsi, %r10	# ivtmp.290, _93
# benchmark_2mm.c:169:             double sum = 0.0;
	vmovsd	%xmm5, %xmm5, %xmm0	# sum, sum
	.p2align 4,,10
	.p2align 3
.L79:
# benchmark_2mm.c:171:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	vmulpd	(%rsi), %xmm3, %xmm2	# MEM <vector(2) double> [(double *)_85], vect_cst__253, vect__55.271
# benchmark_2mm.c:171:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	vmovsd	(%r9), %xmm1	# MEM[(double *)_86], MEM[(double *)_86]
	addq	$16, %rsi	#, ivtmp.290
	vmovhpd	(%r9,%r11), %xmm1, %xmm1	# MEM[(double *)_86 + _263 * 1], MEM[(double *)_86], tmp291
	addq	%r13, %r9	# _259, ivtmp.293
# benchmark_2mm.c:171:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	vmulpd	%xmm1, %xmm2, %xmm2	# tmp291, vect__55.271, vect__62.272
	vaddsd	%xmm2, %xmm0, %xmm0	# stmp_sum_63.273, sum, stmp_sum_63.273
# benchmark_2mm.c:171:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	vunpckhpd	%xmm2, %xmm2, %xmm2	# vect__62.272, stmp_sum_63.273
	vaddsd	%xmm0, %xmm2, %xmm0	# stmp_sum_63.273, stmp_sum_63.273, sum
	cmpq	%r10, %rsi	# _93, ivtmp.290
	jne	.L79	#,
# benchmark_2mm.c:170:             for (int k = 0; k < nk; k++) {
	movl	%r15d, %esi	# niters_vector_mult_vf.266, k
	cmpl	%r15d, %ebx	# niters_vector_mult_vf.266, nk
	je	.L81	#,
.L78:
# benchmark_2mm.c:171:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	addl	%esi, %ecx	# k, tmp295
	movslq	%ecx, %rcx	# tmp295, tmp296
# benchmark_2mm.c:171:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	vmulsd	(%r8,%rcx,8), %xmm4, %xmm1	# *_228, alpha, tmp297
# benchmark_2mm.c:171:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	imull	%r12d, %esi	# nj, tmp298
# benchmark_2mm.c:171:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	addl	%edx, %esi	# j, tmp299
	movslq	%esi, %rsi	# tmp299, tmp300
# benchmark_2mm.c:171:                 sum += alpha * A[i*nk + k] * B[k*nj + j];
	vfmadd231sd	(%rdi,%rsi,8), %xmm1, %xmm0	# *_235, tmp297, sum
.L81:
# benchmark_2mm.c:173:             tmp[i*nj + j] = sum;
	movl	%eax, %ecx	# i, tmp279
	imull	%r12d, %ecx	# nj, tmp279
# benchmark_2mm.c:173:             tmp[i*nj + j] = sum;
	movq	24(%rsp), %rsi	# %sfp, tmp
# benchmark_2mm.c:173:             tmp[i*nj + j] = sum;
	addl	%edx, %ecx	# j, tmp280
	movslq	%ecx, %rcx	# tmp280, tmp281
# benchmark_2mm.c:173:             tmp[i*nj + j] = sum;
	vmovsd	%xmm0, (%rsi,%rcx,8)	# sum, *_44
	cmpl	20(%rsp), %r14d	# %sfp, ivtmp.298
	je	.L94	#,
	incl	%edx	# j
	cmpl	%edx, %r12d	# j, nj
	jle	.L99	#,
.L77:
	incl	%r14d	# ivtmp.298
	jmp	.L66	#
.L98:
# benchmark_2mm.c:169:             double sum = 0.0;
	vmovsd	%xmm5, %xmm5, %xmm0	# sum, sum
	jmp	.L81	#
.L99:
	incl	%eax	# i
# benchmark_2mm.c:173:             tmp[i*nj + j] = sum;
	xorl	%edx, %edx	# j
	jmp	.L77	#
.L86:
# benchmark_2mm.c:170:             for (int k = 0; k < nk; k++) {
	xorl	%esi, %esi	# k
# benchmark_2mm.c:169:             double sum = 0.0;
	vmovsd	%xmm5, %xmm5, %xmm0	# sum, sum
	jmp	.L78	#
	.cfi_endproc
.LFE5554:
	.size	kernel_2mm_collapsed._omp_fn.0, .-kernel_2mm_collapsed._omp_fn.0
	.p2align 4
	.type	kernel_2mm_collapsed._omp_fn.1, @function
kernel_2mm_collapsed._omp_fn.1:
.LFB5555:
	.cfi_startproc
	endbr64	
	pushq	%rbp	#
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp	#,
	.cfi_def_cfa_register 6
	pushq	%r15	#
	pushq	%r14	#
	pushq	%r13	#
	pushq	%r12	#
	pushq	%rbx	#
	andq	$-32, %rsp	#,
	subq	$32, %rsp	#,
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
# benchmark_2mm.c:177:     #pragma omp parallel for collapse(2)
	movl	32(%rdi), %r14d	# *.omp_data_i_11(D).ni, ni
	movl	40(%rdi), %ebx	# *.omp_data_i_11(D).nl, nl
	testl	%r14d, %r14d	# ni
	jle	.L129	#,
	testl	%ebx, %ebx	# nl
	jle	.L129	#,
	movq	%rdi, %r12	# tmp265, .omp_data_i
	call	omp_get_num_threads@PLT	#
	movl	%eax, %r13d	# tmp266, _22
	call	omp_get_thread_num@PLT	#
	movl	%eax, %ecx	# tmp267, _25
	movl	%r14d, %eax	# ni, ni
	imull	%ebx, %eax	# nl, ni
	xorl	%edx, %edx	# tt.25_2
	divl	%r13d	# _22
	movl	%eax, %esi	# tmp209, q.24_1
	cmpl	%edx, %ecx	# tt.25_2, _25
	jb	.L103	#,
.L120:
	imull	%esi, %ecx	# q.24_1, tmp212
	leal	(%rcx,%rdx), %eax	#, _30
	leal	(%rsi,%rax), %edx	#, tmp213
	cmpl	%edx, %eax	# tmp213, _30
	jnb	.L129	#,
	movq	16(%r12), %rdx	# *.omp_data_i_11(D).D, D
	movq	24(%r12), %rdi	# *.omp_data_i_11(D).tmp, tmp
	movq	%rdx, 24(%rsp)	# D, %sfp
	xorl	%edx, %edx	# j
	movq	8(%r12), %r8	# *.omp_data_i_11(D).C, C
	vmovsd	(%r12), %xmm4	# *.omp_data_i_11(D).beta, beta
	divl	%ebx	# nl
	movl	36(%r12), %r12d	# *.omp_data_i_11(D).nj, nj
	cmpl	$1, %ebx	#, nl
	jne	.L131	#,
	leal	-1(%rsi), %ebx	#, _208
	movl	%ebx, 20(%rsp)	# _208, %sfp
	leal	-1(%r12), %ebx	#, _84
	movl	%ebx, 16(%rsp)	# _84, %sfp
	movl	%r12d, %ecx	# nj, bnd.307
	movl	%r12d, %ebx	# nj, niters_vector_mult_vf.308
	shrl	$2, %ecx	#, bnd.307
	andl	$-4, %ebx	#, niters_vector_mult_vf.308
	leal	-1(%rcx), %r10d	#, tmp218
	movl	%ebx, 12(%rsp)	# niters_vector_mult_vf.308, %sfp
	incq	%r10	# tmp219
	salq	$5, %r10	#, _197
	xorl	%r15d, %r15d	# ivtmp.339
	.p2align 4,,10
	.p2align 3
.L114:
# benchmark_2mm.c:180:             double sum = beta * D[i*nl + j];
	movq	24(%rsp), %rbx	# %sfp, D
# benchmark_2mm.c:180:             double sum = beta * D[i*nl + j];
	leal	(%rax,%rdx), %ecx	#, tmp220
	movslq	%ecx, %rcx	# tmp220, tmp221
# benchmark_2mm.c:180:             double sum = beta * D[i*nl + j];
	leaq	(%rbx,%rcx,8), %r14	#, _136
# benchmark_2mm.c:180:             double sum = beta * D[i*nl + j];
	vmulsd	(%r14), %xmm4, %xmm0	# *_136, beta, sum
# benchmark_2mm.c:181:             for (int k = 0; k < nj; k++) {
	testl	%r12d, %r12d	# nj
	jle	.L107	#,
# benchmark_2mm.c:182:                 sum += tmp[i*nj + k] * C[k*nl + j];
	movl	%r12d, %ecx	# nj, _133
	imull	%eax, %ecx	# i, _133
	cmpl	$2, 16(%rsp)	#, %sfp
	jbe	.L121	#,
	movslq	%ecx, %rsi	# _133, _133
	leaq	(%rdi,%rsi,8), %r11	#, vectp.311
	movslq	%edx, %rsi	# j, j
	leaq	(%r8,%rsi,8), %r9	#, vectp.314
	xorl	%esi, %esi	# ivtmp.334
	.p2align 4,,10
	.p2align 3
.L109:
# benchmark_2mm.c:182:                 sum += tmp[i*nj + k] * C[k*nl + j];
	vmovupd	(%r11,%rsi), %ymm5	# MEM <vector(4) double> [(double *)vectp.311_75 + ivtmp.334_85 * 1], tmp328
	vmulpd	(%r9,%rsi), %ymm5, %ymm2	# MEM <vector(4) double> [(double *)vectp.314_46 + ivtmp.334_85 * 1], tmp328, vect__119.316
	addq	$32, %rsi	#, ivtmp.334
	vaddsd	%xmm2, %xmm0, %xmm0	# stmp_sum_118.317, sum, stmp_sum_118.317
	vunpckhpd	%xmm2, %xmm2, %xmm3	# tmp229, stmp_sum_118.317
	vextractf128	$0x1, %ymm2, %xmm1	# vect__119.316, tmp231
	vaddsd	%xmm3, %xmm0, %xmm0	# stmp_sum_118.317, stmp_sum_118.317, stmp_sum_118.317
# benchmark_2mm.c:182:                 sum += tmp[i*nj + k] * C[k*nl + j];
	vaddsd	%xmm1, %xmm0, %xmm0	# stmp_sum_118.317, stmp_sum_118.317, stmp_sum_118.317
	vunpckhpd	%xmm1, %xmm1, %xmm1	# tmp231, stmp_sum_118.317
	vaddsd	%xmm1, %xmm0, %xmm0	# stmp_sum_118.317, stmp_sum_118.317, sum
	cmpq	%r10, %rsi	# _197, ivtmp.334
	jne	.L109	#,
	movl	12(%rsp), %ebx	# %sfp, niters_vector_mult_vf.308
	cmpl	%r12d, %ebx	# nj, niters_vector_mult_vf.308
	je	.L107	#,
	movl	%ebx, %r9d	# niters_vector_mult_vf.308,
# benchmark_2mm.c:181:             for (int k = 0; k < nj; k++) {
	movl	%ebx, %esi	# niters_vector_mult_vf.308, tmp.321
.L108:
	movl	%r12d, %r13d	# nj, niters.318
	subl	%r9d, %r13d	# _77, niters.318
	cmpl	$1, %r13d	#, niters.318
	je	.L111	#,
# benchmark_2mm.c:182:                 sum += tmp[i*nj + k] * C[k*nl + j];
	movslq	%edx, %rbx	# j, j
	addq	%r9, %rbx	# _189, tmp239
	vmovupd	(%r8,%rbx,8), %xmm1	# MEM <vector(2) double> [(double *)vectp.326_195], vect__90.327
# benchmark_2mm.c:182:                 sum += tmp[i*nj + k] * C[k*nl + j];
	movslq	%ecx, %r11	# _133, _133
	addq	%r9, %r11	# _189, tmp236
# benchmark_2mm.c:182:                 sum += tmp[i*nj + k] * C[k*nl + j];
	vmulpd	(%rdi,%r11,8), %xmm1, %xmm1	# MEM <vector(2) double> [(double *)vectp.323_187], vect__90.327, vect__89.328
	movl	%r13d, %r9d	# niters.318, niters_vector_mult_vf.320
	andl	$-2, %r9d	#, niters_vector_mult_vf.320
	addl	%r9d, %esi	# niters_vector_mult_vf.320, tmp.321
	vaddsd	%xmm1, %xmm0, %xmm0	# stmp_sum_88.329, sum, stmp_sum_88.329
# benchmark_2mm.c:182:                 sum += tmp[i*nj + k] * C[k*nl + j];
	vunpckhpd	%xmm1, %xmm1, %xmm1	# vect__89.328, stmp_sum_88.329
	vaddsd	%xmm0, %xmm1, %xmm0	# stmp_sum_88.329, stmp_sum_88.329, sum
	cmpl	%r9d, %r13d	# niters_vector_mult_vf.320, niters.318
	je	.L107	#,
.L111:
# benchmark_2mm.c:182:                 sum += tmp[i*nj + k] * C[k*nl + j];
	addl	%esi, %ecx	# tmp.321, tmp242
	movslq	%ecx, %rcx	# tmp242, tmp243
# benchmark_2mm.c:182:                 sum += tmp[i*nj + k] * C[k*nl + j];
	vmovsd	(%rdi,%rcx,8), %xmm6	# *_167, tmp332
# benchmark_2mm.c:182:                 sum += tmp[i*nj + k] * C[k*nl + j];
	addl	%esi, %edx	# tmp.321, tmp244
	movslq	%edx, %rdx	# tmp244, tmp245
# benchmark_2mm.c:182:                 sum += tmp[i*nj + k] * C[k*nl + j];
	vfmadd231sd	(%r8,%rdx,8), %xmm6, %xmm0	# *_172, tmp332, sum
.L107:
# benchmark_2mm.c:184:             D[i*nl + j] = sum;
	vmovsd	%xmm0, (%r14)	# sum, *_136
	cmpl	20(%rsp), %r15d	# %sfp, ivtmp.339
	je	.L128	#,
	incl	%eax	# i
	incl	%r15d	# ivtmp.339
	xorl	%edx, %edx	# j
	jmp	.L114	#
.L128:
	vzeroupper
.L129:
# benchmark_2mm.c:177:     #pragma omp parallel for collapse(2)
	leaq	-40(%rbp), %rsp	#,
	popq	%rbx	#
	popq	%r12	#
	popq	%r13	#
	popq	%r14	#
	popq	%r15	#
	popq	%rbp	#
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret	
.L121:
	.cfi_restore_state
# benchmark_2mm.c:182:                 sum += tmp[i*nj + k] * C[k*nl + j];
	xorl	%r9d, %r9d	#
# benchmark_2mm.c:181:             for (int k = 0; k < nj; k++) {
	xorl	%esi, %esi	# tmp.321
	jmp	.L108	#
.L103:
	incl	%esi	# q.24_1
# benchmark_2mm.c:177:     #pragma omp parallel for collapse(2)
	xorl	%edx, %edx	# tt.25_2
	jmp	.L120	#
.L131:
	decl	%esi	# _89
	movl	%esi, 20(%rsp)	# _89, %sfp
	movslq	%ebx, %r10	# nl, nl
	salq	$3, %r10	#, _126
# benchmark_2mm.c:177:     #pragma omp parallel for collapse(2)
	xorl	%r13d, %r13d	# ivtmp.349
	leal	-1(%r12), %r14d	#, tmp263
	leaq	8(%rdi), %r15	#, tmp264
	.p2align 4,,10
	.p2align 3
.L106:
# benchmark_2mm.c:180:             double sum = beta * D[i*nl + j];
	movl	%eax, %ecx	# i, tmp247
	imull	%ebx, %ecx	# nl, tmp247
# benchmark_2mm.c:180:             double sum = beta * D[i*nl + j];
	movq	24(%rsp), %rsi	# %sfp, D
# benchmark_2mm.c:180:             double sum = beta * D[i*nl + j];
	addl	%edx, %ecx	# j, tmp248
	movslq	%ecx, %rcx	# tmp248, tmp249
# benchmark_2mm.c:180:             double sum = beta * D[i*nl + j];
	leaq	(%rsi,%rcx,8), %r11	#, _40
# benchmark_2mm.c:180:             double sum = beta * D[i*nl + j];
	vmulsd	(%r11), %xmm4, %xmm0	# *_40, beta, sum
# benchmark_2mm.c:181:             for (int k = 0; k < nj; k++) {
	testl	%r12d, %r12d	# nj
	jle	.L119	#,
# benchmark_2mm.c:182:                 sum += tmp[i*nj + k] * C[k*nl + j];
	movl	%eax, %r9d	# i, tmp251
	imull	%r12d, %r9d	# nj, tmp251
	movslq	%edx, %rsi	# j, j
	leaq	(%r8,%rsi,8), %rsi	#, ivtmp.347
	movslq	%r9d, %r9	# tmp251, _163
	leaq	(%rdi,%r9,8), %rcx	#, ivtmp.346
	addq	%r14, %r9	# tmp263, tmp257
	leaq	(%r15,%r9,8), %r9	#, _93
	.p2align 4,,10
	.p2align 3
.L118:
# benchmark_2mm.c:182:                 sum += tmp[i*nj + k] * C[k*nl + j];
	vmovsd	(%rcx), %xmm7	# MEM[(double *)_120], tmp339
# benchmark_2mm.c:181:             for (int k = 0; k < nj; k++) {
	addq	$8, %rcx	#, ivtmp.346
# benchmark_2mm.c:182:                 sum += tmp[i*nj + k] * C[k*nl + j];
	vfmadd231sd	(%rsi), %xmm7, %xmm0	# MEM[(double *)_119], tmp339, sum
# benchmark_2mm.c:181:             for (int k = 0; k < nj; k++) {
	addq	%r10, %rsi	# _126, ivtmp.347
	cmpq	%rcx, %r9	# ivtmp.346, _93
	jne	.L118	#,
.L119:
# benchmark_2mm.c:184:             D[i*nl + j] = sum;
	vmovsd	%xmm0, (%r11)	# sum, *_40
	cmpl	%r13d, 20(%rsp)	# ivtmp.349, %sfp
	je	.L129	#,
	incl	%edx	# j
	cmpl	%edx, %ebx	# j, nl
	jle	.L132	#,
.L117:
	incl	%r13d	# ivtmp.349
	jmp	.L106	#
.L132:
	incl	%eax	# i
	xorl	%edx, %edx	# j
	jmp	.L117	#
	.cfi_endproc
.LFE5555:
	.size	kernel_2mm_collapsed._omp_fn.1, .-kernel_2mm_collapsed._omp_fn.1
	.p2align 4
	.type	kernel_2mm_tiled._omp_fn.0, @function
kernel_2mm_tiled._omp_fn.0:
.LFB5556:
	.cfi_startproc
	endbr64	
	pushq	%r13	#
	.cfi_def_cfa_offset 16
	.cfi_offset 13, -16
	pushq	%r12	#
	.cfi_def_cfa_offset 24
	.cfi_offset 12, -24
	pushq	%rbp	#
	.cfi_def_cfa_offset 32
	.cfi_offset 6, -32
	pushq	%rbx	#
	.cfi_def_cfa_offset 40
	.cfi_offset 3, -40
	subq	$8, %rsp	#,
	.cfi_def_cfa_offset 48
# benchmark_2mm.c:197:     #pragma omp parallel for collapse(2)
	movl	8(%rdi), %ebx	# *.omp_data_i_9(D).ni, ni
	movl	12(%rdi), %ebp	# *.omp_data_i_9(D).nj, nj
	testl	%ebx, %ebx	# ni
	jle	.L147	#,
	testl	%ebp, %ebp	# nj
	jle	.L147	#,
	movq	%rdi, %r13	# tmp141, .omp_data_i
	call	omp_get_num_threads@PLT	#
	movl	%eax, %r12d	# tmp142, _18
	call	omp_get_thread_num@PLT	#
	movl	%eax, %esi	# tmp143, _21
	movl	%ebx, %eax	# ni, ni
	imull	%ebp, %eax	# nj, ni
	xorl	%edx, %edx	# tt.69_2
	divl	%r12d	# _18
	movl	%eax, %ecx	# tmp124, q.68_1
	cmpl	%edx, %esi	# tt.69_2, _21
	jb	.L136	#,
.L143:
	imull	%ecx, %esi	# q.68_1, tmp127
	addl	%edx, %esi	# tt.69_2, _27
	leal	(%rcx,%rsi), %r9d	#, _28
	cmpl	%r9d, %esi	# _28, _27
	jnb	.L147	#,
	movl	%esi, %eax	# _27, tmp128
	xorl	%edx, %edx	# tmp129
	divl	%ebp	# nj
	movq	0(%r13), %rdi	# *.omp_data_i_9(D).tmp, tmp
# benchmark_2mm.c:200:             tmp[i*nj + j] = 0.0;
	movl	%ebp, %r8d	# nj, tmp130
	incl	%esi	# tmp134
	imull	%eax, %r8d	# i, tmp130
# benchmark_2mm.c:200:             tmp[i*nj + j] = 0.0;
	addl	%edx, %r8d	# j, tmp131
	movslq	%r8d, %r8	# tmp131, tmp132
# benchmark_2mm.c:200:             tmp[i*nj + j] = 0.0;
	movq	$0x000000000, (%rdi,%r8,8)	#, *_46
	cmpl	%esi, %r9d	# tmp134, _28
	jbe	.L147	#,
	cmpl	$1, %ebp	#, nj
	jne	.L149	#,
	decl	%ecx	# _72
	xorl	%edx, %edx	# ivtmp.359
	.p2align 4,,10
	.p2align 3
.L142:
	incl	%eax	# i
# benchmark_2mm.c:200:             tmp[i*nj + j] = 0.0;
	movslq	%eax, %rsi	# i, i
	incl	%edx	# ivtmp.359
# benchmark_2mm.c:200:             tmp[i*nj + j] = 0.0;
	movq	$0x000000000, (%rdi,%rsi,8)	#, *_3
	cmpl	%ecx, %edx	# _72, ivtmp.359
	jne	.L142	#,
.L147:
# benchmark_2mm.c:197:     #pragma omp parallel for collapse(2)
	addq	$8, %rsp	#,
	.cfi_remember_state
	.cfi_def_cfa_offset 40
	popq	%rbx	#
	.cfi_def_cfa_offset 32
	popq	%rbp	#
	.cfi_def_cfa_offset 24
	popq	%r12	#
	.cfi_def_cfa_offset 16
	popq	%r13	#
	.cfi_def_cfa_offset 8
	ret	
	.p2align 4,,10
	.p2align 3
.L136:
	.cfi_restore_state
	incl	%ecx	# q.68_1
# benchmark_2mm.c:197:     #pragma omp parallel for collapse(2)
	xorl	%edx, %edx	# tt.69_2
	jmp	.L143	#
	.p2align 4,,10
	.p2align 3
.L149:
	decl	%ecx	# _68
# benchmark_2mm.c:200:             tmp[i*nj + j] = 0.0;
	xorl	%r8d, %r8d	# ivtmp.364
	jmp	.L140	#
	.p2align 4,,10
	.p2align 3
.L139:
# benchmark_2mm.c:200:             tmp[i*nj + j] = 0.0;
	movl	%ebp, %esi	# nj, tmp135
	imull	%eax, %esi	# i, tmp135
	incl	%r8d	# ivtmp.364
# benchmark_2mm.c:200:             tmp[i*nj + j] = 0.0;
	addl	%edx, %esi	# j, tmp136
	movslq	%esi, %rsi	# tmp136, tmp137
# benchmark_2mm.c:200:             tmp[i*nj + j] = 0.0;
	movq	$0x000000000, (%rdi,%rsi,8)	#, *_92
	cmpl	%r8d, %ecx	# ivtmp.364, _68
	je	.L147	#,
.L140:
	incl	%edx	# j
	cmpl	%edx, %ebp	# j, nj
	jg	.L139	#,
	incl	%eax	# i
	xorl	%edx, %edx	# j
	jmp	.L139	#
	.cfi_endproc
.LFE5556:
	.size	kernel_2mm_tiled._omp_fn.0, .-kernel_2mm_tiled._omp_fn.0
	.p2align 4
	.type	kernel_2mm_tiled._omp_fn.1, @function
kernel_2mm_tiled._omp_fn.1:
.LFB5557:
	.cfi_startproc
	endbr64	
	pushq	%rbp	#
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp	#,
	.cfi_def_cfa_register 6
	pushq	%r15	#
	pushq	%r14	#
	pushq	%r13	#
	pushq	%r12	#
	pushq	%rbx	#
	andq	$-32, %rsp	#,
	subq	$160, %rsp	#,
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
# benchmark_2mm.c:203:     #pragma omp parallel for collapse(2) schedule(static)
	movl	36(%rdi), %eax	# *.omp_data_i_17(D).nj, nj
	movl	32(%rdi), %ebx	# *.omp_data_i_17(D).ni, ni
	movl	%eax, 148(%rsp)	# nj, %sfp
	movq	%rdi, 152(%rsp)	# .omp_data_i, %sfp
	movl	%ebx, 32(%rsp)	# ni, %sfp
	testl	%ebx, %ebx	# ni
	jle	.L182	#,
	addl	$31, %ebx	#, tmp218
	sarl	$5, %ebx	#, .count.60_27
	testl	%eax, %eax	# nj
	jle	.L182	#,
	leal	31(%rax), %r12d	#, tmp219
	call	omp_get_num_threads@PLT	#
	movl	%eax, %r13d	# tmp284, _32
	call	omp_get_thread_num@PLT	#
	sarl	$5, %r12d	#, .count.61_30
	movl	%eax, %ecx	# tmp285, _35
	movl	%ebx, %eax	# .count.60_27, .count.60_27
	imull	%r12d, %eax	# .count.61_30, .count.60_27
	xorl	%edx, %edx	# tt.63_2
	movq	152(%rsp), %rdi	# %sfp, .omp_data_i
	divl	%r13d	# _32
	cmpl	%edx, %ecx	# tt.63_2, _35
	movl	%eax, %esi	# tmp220, q.62_1
	jb	.L153	#,
.L170:
	movl	%ecx, %eax	# _35, _35
	imull	%esi, %eax	# q.62_1, _35
	addl	%edx, %eax	# tt.63_2, _40
	leal	(%rsi,%rax), %edx	#, tmp224
	cmpl	%edx, %eax	# tmp224, _40
	jnb	.L182	#,
	xorl	%edx, %edx	# tmp226
	divl	%r12d	# .count.61_30
	movq	24(%rdi), %r15	# *.omp_data_i_17(D).tmp, tmp
	movq	16(%rdi), %r14	# *.omp_data_i_17(D).B, B
	movq	8(%rdi), %rcx	# *.omp_data_i_17(D).A, A
	vmovsd	(%rdi), %xmm3	# *.omp_data_i_17(D).alpha, alpha
	movl	40(%rdi), %edi	# *.omp_data_i_17(D).nk, nk
	movl	%edi, 72(%rsp)	# nk, %sfp
	sall	$5, %edx	#, tmp226
	sall	$5, %eax	#, tmp225
	movl	%edx, 120(%rsp)	# tmp226, %sfp
	movl	%eax, 76(%rsp)	# ii, %sfp
	movl	%eax, %edx	# tmp225, ii
	testl	%edi, %edi	# nk
	jle	.L182	#,
	movl	148(%rsp), %ebx	# %sfp, nj
	leal	-1(%rsi), %eax	#, _262
	movl	%eax, 24(%rsp)	# _262, %sfp
	movl	%ebx, %eax	# nj, _112
	sall	$5, %eax	#, _112
	movl	%eax, 28(%rsp)	# _112, %sfp
	cltq
	movq	%rax, 16(%rsp)	# _239, %sfp
	movslq	%edi, %rax	# nk, nk
	salq	$3, %rax	#, _82
	movq	%rax, 96(%rsp)	# _82, %sfp
	leal	32(%rdx), %eax	#, _51
	movl	%eax, 56(%rsp)	# _51, %sfp
	leaq	8(%rcx), %rax	#, tmp275
	movq	%rax, 8(%rsp)	# tmp275, %sfp
	movslq	%ebx, %rax	# nj, _223
	movl	$0, 48(%rsp)	#, %sfp
	movq	%rax, 136(%rsp)	# _223, %sfp
.L154:
# benchmark_2mm.c:207:                 int i_end = (ii + TILE < ni) ? ii + TILE : ni;
	movl	32(%rsp), %ebx	# %sfp, ni
	movl	56(%rsp), %eax	# %sfp, _51
# benchmark_2mm.c:208:                 int j_end = (jj + TILE < nj) ? jj + TILE : nj;
	movl	148(%rsp), %edi	# %sfp, nj
# benchmark_2mm.c:207:                 int i_end = (ii + TILE < ni) ? ii + TILE : ni;
	cmpl	%eax, %ebx	# _51, ni
	cmovle	%ebx, %eax	# ni,, i_end
# benchmark_2mm.c:208:                 int j_end = (jj + TILE < nj) ? jj + TILE : nj;
	movl	120(%rsp), %ebx	# %sfp, jj
# benchmark_2mm.c:207:                 int i_end = (ii + TILE < ni) ? ii + TILE : ni;
	movl	%eax, 112(%rsp)	# i_end, %sfp
# benchmark_2mm.c:208:                 int j_end = (jj + TILE < nj) ? jj + TILE : nj;
	addl	$32, %ebx	#, _53
# benchmark_2mm.c:208:                 int j_end = (jj + TILE < nj) ? jj + TILE : nj;
	cmpl	%ebx, %edi	# _53, nj
# benchmark_2mm.c:208:                 int j_end = (jj + TILE < nj) ? jj + TILE : nj;
	movl	%ebx, 60(%rsp)	# _53, %sfp
# benchmark_2mm.c:208:                 int j_end = (jj + TILE < nj) ? jj + TILE : nj;
	cmovle	%edi, %ebx	# nj,, j_end
	movl	%ebx, 52(%rsp)	# j_end, %sfp
	cmpl	%eax, 76(%rsp)	# i_end, %sfp
	jge	.L159	#,
	movl	76(%rsp), %esi	# %sfp, ii
	movl	72(%rsp), %eax	# %sfp, tmp228
	movslq	120(%rsp), %rdi	# %sfp,
	imull	%esi, %eax	# ii, tmp228
	movl	%ebx, %r13d	# j_end, niters.376
	subl	%edi, %r13d	# jj, niters.376
	cltq
	movq	%rax, 40(%rsp)	# ivtmp.440, %sfp
	movl	148(%rsp), %eax	# %sfp, ivtmp.429
	movq	%rdi, 88(%rsp)	# ivtmp.438, %sfp
	imull	%esi, %eax	# ii, ivtmp.429
	movq	%rdi, 104(%rsp)	# ivtmp.438, %sfp
	movq	$0, 64(%rsp)	#, %sfp
	movl	%eax, 36(%rsp)	# ivtmp.429, %sfp
	leal	-1(%r13), %eax	#, _132
	movl	%eax, 144(%rsp)	# _132, %sfp
	movl	%r13d, %eax	# niters.376, bnd.377
	shrl	$2, %eax	#, bnd.377
	decl	%eax	# tmp231
	incq	%rax	# tmp232
	salq	$5, %rax	#, tmp232
	movq	%rax, %r12	# tmp232, _113
	movl	%r13d, %eax	# niters.376, niters_vector_mult_vf.378
	andl	$-4, %eax	#, niters_vector_mult_vf.378
	movl	%eax, 132(%rsp)	# niters_vector_mult_vf.378, %sfp
	addl	%edi, %eax	# jj, tmp.379
	movl	%eax, 128(%rsp)	# tmp.379, %sfp
	movl	$0, 116(%rsp)	#, %sfp
.L158:
	movq	64(%rsp), %rbx	# %sfp, ivtmp.439
# benchmark_2mm.c:209:                 int k_end = (kk + TILE < nk) ? kk + TILE : nk;
	movl	72(%rsp), %edi	# %sfp, nk
	movl	%ebx, %eax	# ivtmp.439, tmp385
	addl	$32, %eax	#, tmp233
	cmpl	%edi, %eax	# nk, tmp233
	cmovg	%edi, %eax	# tmp233,, nk, k_end
	cmpl	%ebx, %eax	# ivtmp.439, k_end
	jle	.L160	#,
	movl	52(%rsp), %edi	# %sfp, j_end
	cmpl	%edi, 120(%rsp)	# j_end, %sfp
	jge	.L160	#,
	subl	%ebx, %eax	# _255, tmp234
	movq	40(%rsp), %rbx	# %sfp, ivtmp.440
	decl	%eax	# _147
	leaq	(%rax,%rbx), %rdx	#, tmp236
	movq	8(%rsp), %rbx	# %sfp, tmp275
	notq	%rax	# tmp241
	leaq	(%rbx,%rdx,8), %rbx	#, ivtmp.427
	movq	%rbx, 152(%rsp)	# ivtmp.427, %sfp
	movl	76(%rsp), %ebx	# %sfp, ii
	salq	$3, %rax	#, tmp242
	movl	%ebx, 124(%rsp)	# ii, %sfp
	movq	%rax, 80(%rsp)	# tmp242, %sfp
	movl	36(%rsp), %edi	# %sfp, ivtmp.429
	.p2align 4,,10
	.p2align 3
.L162:
	movq	80(%rsp), %r11	# %sfp, ivtmp.417
	movslq	%edi, %rbx	# ivtmp.429, _66
	addq	88(%rsp), %rbx	# %sfp, _119
	movq	104(%rsp), %r8	# %sfp, ivtmp.422
	movl	116(%rsp), %esi	# %sfp, ivtmp.421
	addq	152(%rsp), %r11	# %sfp, ivtmp.417
	leaq	(%r15,%rbx,8), %rdx	#, vectp.381
	.p2align 4,,10
	.p2align 3
.L161:
	cmpl	$2, 144(%rsp)	#, %sfp
# benchmark_2mm.c:213:                         double aik = alpha * A[i*nk + k];
	vmulsd	(%r11), %xmm3, %xmm0	# MEM[(double *)_228], alpha, aik
	jbe	.L184	#,
	leaq	(%r14,%r8,8), %rcx	#, _75
	vbroadcastsd	%xmm0, %ymm2	# aik, vect_cst__106
# benchmark_2mm.c:216:                             tmp[i*nj + j] += aik * B[k*nj + j];
	xorl	%eax, %eax	# ivtmp.409
	.p2align 4,,10
	.p2align 3
.L166:
# benchmark_2mm.c:216:                             tmp[i*nj + j] += aik * B[k*nj + j];
	vmovupd	(%rcx,%rax), %ymm1	# MEM <vector(4) double> [(double *)_75 + ivtmp.409_172 * 1], vect__82.387
	vfmadd213pd	(%rdx,%rax), %ymm2, %ymm1	# MEM <vector(4) double> [(double *)vectp.381_122 + ivtmp.409_172 * 1], vect_cst__106, vect__82.387
	vmovupd	%ymm1, (%rdx,%rax)	# vect__82.387, MEM <vector(4) double> [(double *)vectp.381_122 + ivtmp.409_172 * 1]
	addq	$32, %rax	#, ivtmp.409
	cmpq	%rax, %r12	# ivtmp.409, _113
	jne	.L166	#,
	movl	132(%rsp), %eax	# %sfp, niters_vector_mult_vf.378
	cmpl	%r13d, %eax	# niters.376, niters_vector_mult_vf.378
	je	.L167	#,
# benchmark_2mm.c:216:                             tmp[i*nj + j] += aik * B[k*nj + j];
	movl	%eax, %ecx	# niters_vector_mult_vf.378,
	movl	128(%rsp), %eax	# %sfp, tmp.393
.L163:
	movl	%r13d, %r9d	# niters.376, niters.390
	subl	%ecx, %r9d	# _124, niters.390
	cmpl	$1, %r9d	#, niters.390
	je	.L168	#,
	leaq	(%rbx,%rcx), %r10	#, tmp249
	leaq	(%r15,%r10,8), %r10	#, vectp.395
	vmovupd	(%r10), %xmm5	# MEM <vector(2) double> [(double *)vectp.395_199], tmp414
# benchmark_2mm.c:216:                             tmp[i*nj + j] += aik * B[k*nj + j];
	addq	%r8, %rcx	# ivtmp.422, tmp251
# benchmark_2mm.c:216:                             tmp[i*nj + j] += aik * B[k*nj + j];
	vmovddup	%xmm0, %xmm1	# aik, tmp253
	vfmadd132pd	(%r14,%rcx,8), %xmm5, %xmm1	# MEM <vector(2) double> [(double *)vectp.398_209], tmp414, vect__139.401
	movl	%r9d, %ecx	# niters.390, niters_vector_mult_vf.392
	andl	$-2, %ecx	#, niters_vector_mult_vf.392
	addl	%ecx, %eax	# niters_vector_mult_vf.392, tmp.393
	vmovupd	%xmm1, (%r10)	# vect__139.401, MEM <vector(2) double> [(double *)vectp.395_199]
	cmpl	%ecx, %r9d	# niters_vector_mult_vf.392, niters.390
	je	.L167	#,
.L168:
	leal	(%rdi,%rax), %ecx	#, tmp255
	movslq	%ecx, %rcx	# tmp255, tmp256
	leaq	(%r15,%rcx,8), %rcx	#, _176
	vmovsd	(%rcx), %xmm4	# *_176, tmp416
# benchmark_2mm.c:216:                             tmp[i*nj + j] += aik * B[k*nj + j];
	addl	%esi, %eax	# ivtmp.421, tmp258
	cltq
# benchmark_2mm.c:216:                             tmp[i*nj + j] += aik * B[k*nj + j];
	vfmadd132sd	(%r14,%rax,8), %xmm4, %xmm0	# *_181, tmp416, _184
	vmovsd	%xmm0, (%rcx)	# _184, *_176
.L167:
# benchmark_2mm.c:212:                     for (int k = kk; k < k_end; k++) {
	addq	$8, %r11	#, ivtmp.417
	addl	148(%rsp), %esi	# %sfp, ivtmp.421
	addq	136(%rsp), %r8	# %sfp, ivtmp.422
	cmpq	%r11, 152(%rsp)	# ivtmp.417, %sfp
	jne	.L161	#,
# benchmark_2mm.c:211:                 for (int i = ii; i < i_end; i++) {
	incl	124(%rsp)	# %sfp
# benchmark_2mm.c:211:                 for (int i = ii; i < i_end; i++) {
	movq	96(%rsp), %rsi	# %sfp, _82
	addl	148(%rsp), %edi	# %sfp, ivtmp.429
	addq	%rsi, 152(%rsp)	# _82, %sfp
# benchmark_2mm.c:211:                 for (int i = ii; i < i_end; i++) {
	movl	124(%rsp), %eax	# %sfp, i
# benchmark_2mm.c:211:                 for (int i = ii; i < i_end; i++) {
	cmpl	%eax, 112(%rsp)	# i, %sfp
	jne	.L162	#,
.L160:
# benchmark_2mm.c:206:             for (int kk = 0; kk < nk; kk += TILE) {
	addq	$32, 64(%rsp)	#, %sfp
	movl	28(%rsp), %ebx	# %sfp, _112
	addq	$32, 40(%rsp)	#, %sfp
	addl	%ebx, 116(%rsp)	# _112, %sfp
	movq	64(%rsp), %rax	# %sfp, ivtmp.439
	movq	16(%rsp), %rbx	# %sfp, _239
	addq	%rbx, 104(%rsp)	# _239, %sfp
	cmpl	%eax, 72(%rsp)	# tmp395, %sfp
	jg	.L158	#,
.L159:
	movl	24(%rsp), %ebx	# %sfp, _262
	cmpl	%ebx, 48(%rsp)	# _262, %sfp
	je	.L181	#,
	movl	60(%rsp), %ebx	# %sfp, _53
	cmpl	%ebx, 148(%rsp)	# _53, %sfp
	jle	.L185	#,
.L156:
# benchmark_2mm.c:203:     #pragma omp parallel for collapse(2) schedule(static)
	movl	60(%rsp), %eax	# %sfp, _53
	incl	48(%rsp)	# %sfp
	movl	%eax, 120(%rsp)	# _53, %sfp
	jmp	.L154	#
.L181:
	vzeroupper
.L182:
	leaq	-40(%rbp), %rsp	#,
	popq	%rbx	#
	popq	%r12	#
	popq	%r13	#
	popq	%r14	#
	popq	%r15	#
	popq	%rbp	#
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret	
.L153:
	.cfi_restore_state
	incl	%esi	# q.62_1
# benchmark_2mm.c:203:     #pragma omp parallel for collapse(2) schedule(static)
	xorl	%edx, %edx	# tt.63_2
	jmp	.L170	#
	.p2align 4,,10
	.p2align 3
.L184:
# benchmark_2mm.c:216:                             tmp[i*nj + j] += aik * B[k*nj + j];
	movl	120(%rsp), %eax	# %sfp, tmp.393
	xorl	%ecx, %ecx	#
	jmp	.L163	#
.L185:
# benchmark_2mm.c:203:     #pragma omp parallel for collapse(2) schedule(static)
	movl	56(%rsp), %eax	# %sfp, _51
	movl	$0, 60(%rsp)	#, %sfp
	movl	%eax, 76(%rsp)	# _51, %sfp
	addl	$32, %eax	#, _51
	movl	%eax, 56(%rsp)	# _51, %sfp
	jmp	.L156	#
	.cfi_endproc
.LFE5557:
	.size	kernel_2mm_tiled._omp_fn.1, .-kernel_2mm_tiled._omp_fn.1
	.p2align 4
	.type	kernel_2mm_tiled._omp_fn.2, @function
kernel_2mm_tiled._omp_fn.2:
.LFB5558:
	.cfi_startproc
	endbr64	
	pushq	%rbp	#
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp	#,
	.cfi_def_cfa_register 6
	pushq	%r15	#
	pushq	%r14	#
	pushq	%r13	#
	pushq	%r12	#
	pushq	%rbx	#
	andq	$-32, %rsp	#,
	subq	$192, %rsp	#,
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
# benchmark_2mm.c:225:     #pragma omp parallel for collapse(2) schedule(static)
	movq	%fs:40, %rax	# MEM[(<address-space-1> long unsigned int *)40B], nl
	movq	%rax, 184(%rsp)	# nl, D.38262
	movl	40(%rdi), %eax	# *.omp_data_i_19(D).nl, nl
	movl	32(%rdi), %ecx	# *.omp_data_i_19(D).ni, ni
	movl	%ecx, 16(%rsp)	# ni, %sfp
	movl	%eax, 120(%rsp)	# nl, %sfp
	testl	%ecx, %ecx	# ni
	jle	.L186	#,
	leal	31(%rcx), %ebx	#, tmp222
	sarl	$5, %ebx	#, .count.54_29
	testl	%eax, %eax	# nl
	jle	.L186	#,
	leal	31(%rax), %r12d	#, tmp223
	movq	%rdi, %r14	# tmp293, .omp_data_i
	sarl	$5, %r12d	#, tmp223
	call	omp_get_num_threads@PLT	#
	movl	%r12d, %r13d	# tmp223, .count.55_32
	movl	%eax, %r12d	# tmp294, _34
	call	omp_get_thread_num@PLT	#
	movl	%eax, %esi	# tmp295, _37
	movl	%ebx, %eax	# .count.54_29, .count.54_29
	imull	%r13d, %eax	# .count.55_32, .count.54_29
	xorl	%edx, %edx	# tt.57_2
	divl	%r12d	# _34
	movl	%eax, %ecx	# tmp224, q.56_1
	cmpl	%edx, %esi	# tt.57_2, _37
	jb	.L189	#,
.L202:
	movl	%esi, %eax	# _37, _37
	imull	%ecx, %eax	# q.56_1, _37
	addl	%edx, %eax	# tt.57_2, _42
	leal	(%rcx,%rax), %edx	#, tmp228
	cmpl	%edx, %eax	# tmp228, _42
	jnb	.L186	#,
	xorl	%edx, %edx	# tmp230
	divl	%r13d	# .count.55_32
	movl	36(%r14), %r12d	# *.omp_data_i_19(D).nj, nj
	movq	24(%r14), %rbx	# *.omp_data_i_19(D).tmp, tmp
	movl	$0, 20(%rsp)	#, %sfp
	movq	%rbx, 72(%rsp)	# tmp, %sfp
	movq	16(%r14), %rbx	# *.omp_data_i_19(D).D, D
	vmovsd	(%r14), %xmm3	# *.omp_data_i_19(D).beta, beta
	movq	%rbx, 8(%rsp)	# D, %sfp
	movq	8(%r14), %rbx	# *.omp_data_i_19(D).C, C
	movq	%rbx, 112(%rsp)	# C, %sfp
	sall	$5, %eax	#, tmp229
	movl	%eax, %edi	# tmp229, ii
	movl	%eax, 24(%rsp)	# ii, %sfp
	leal	-1(%rcx), %eax	#, _85
	movl	%eax, 4(%rsp)	# _85, %sfp
	leal	-1(%r12), %eax	#, _103
	movl	%eax, 84(%rsp)	# _103, %sfp
	movl	%r12d, %eax	# nj, bnd.462
	shrl	$2, %eax	#, bnd.462
	sall	$5, %edx	#, tmp230
	decl	%eax	# _157
	movl	%r12d, %ecx	# nj, niters_vector_mult_vf.463
	movl	%edx, 64(%rsp)	# tmp230, %sfp
	andl	$-4, %ecx	#, niters_vector_mult_vf.463
	movslq	120(%rsp), %rdx	# %sfp, _74
	incq	%rax	# tmp283
	movl	%ecx, 80(%rsp)	# niters_vector_mult_vf.463, %sfp
	salq	$5, %rax	#, tmp283
	leal	32(%rdi), %ecx	#, _50
	movq	%rdx, %r13	# _74, _169
	leaq	0(,%rdx,8), %rbx	#, _167
	movl	%ecx, 28(%rsp)	# _50, %sfp
	leaq	(%rdx,%rdx,2), %rdx	#, tmp235
	movq	%rax, 32(%rsp)	# tmp283, %sfp
	leaq	0(,%rdx,8), %r14	#, tmp236
	movl	64(%rsp), %edx	# %sfp, jj
	salq	$5, %r13	#, _169
.L190:
# benchmark_2mm.c:228:             int i_end = (ii + TILE < ni) ? ii + TILE : ni;
	movl	16(%rsp), %ecx	# %sfp, ni
	movl	28(%rsp), %eax	# %sfp, _50
# benchmark_2mm.c:229:             int j_end = (jj + TILE < nl) ? jj + TILE : nl;
	movl	120(%rsp), %edi	# %sfp, nl
# benchmark_2mm.c:228:             int i_end = (ii + TILE < ni) ? ii + TILE : ni;
	cmpl	%eax, %ecx	# _50, ni
	cmovle	%ecx, %eax	# ni,, i_end
# benchmark_2mm.c:229:             int j_end = (jj + TILE < nl) ? jj + TILE : nl;
	leal	32(%rdx), %r9d	#, _52
# benchmark_2mm.c:231:             for (int i = ii; i < i_end; i++) {
	movl	24(%rsp), %ecx	# %sfp, ii
# benchmark_2mm.c:229:             int j_end = (jj + TILE < nl) ? jj + TILE : nl;
	cmpl	%r9d, %edi	# _52, nl
# benchmark_2mm.c:229:             int j_end = (jj + TILE < nl) ? jj + TILE : nl;
	movl	%r9d, 48(%rsp)	# _52, %sfp
# benchmark_2mm.c:228:             int i_end = (ii + TILE < ni) ? ii + TILE : ni;
	movl	%eax, 52(%rsp)	# i_end, %sfp
# benchmark_2mm.c:229:             int j_end = (jj + TILE < nl) ? jj + TILE : nl;
	cmovle	%edi, %r9d	# nl,, j_end
# benchmark_2mm.c:231:             for (int i = ii; i < i_end; i++) {
	cmpl	%eax, %ecx	# i_end, ii
	jge	.L191	#,
	cmpl	%r9d, %edx	# j_end, jj
	jge	.L191	#,
	movslq	%edx, %rax	# jj, _101
	movl	%ecx, %edx	# ii, tmp237
	imull	%edi, %edx	# nl, tmp237
	movq	8(%rsp), %rdi	# %sfp, D
	movl	%ecx, %r11d	# ii, ivtmp.502
	movslq	%edx, %rdx	# tmp237, tmp238
	addq	%rax, %rdx	# _101, tmp239
	leaq	(%rdi,%rdx,8), %rdx	#, ivtmp.500
	movq	%rdx, 56(%rsp)	# ivtmp.500, %sfp
	movq	112(%rsp), %rdx	# %sfp, C
	imull	%r12d, %r11d	# nj, ivtmp.502
	leaq	(%rdx,%rax,8), %rax	#, ivtmp.493
	movq	%rax, 40(%rsp)	# ivtmp.493, %sfp
	leaq	128(%rsp), %rax	#, tmp460
	movl	%ecx, 68(%rsp)	# ii, %sfp
	movq	%rax, 96(%rsp)	# tmp460, %sfp
	.p2align 4,,10
	.p2align 3
.L194:
	movq	72(%rsp), %rcx	# %sfp, tmp
	movq	32(%rsp), %r8	# %sfp, _144
	movslq	%r11d, %rax	# ivtmp.502, _76
	leaq	(%rcx,%rax,8), %rax	#, ivtmp.478
	addq	%rax, %r8	# ivtmp.478, _144
	movq	%r8, 104(%rsp)	# _144, %sfp
	movq	%rax, 88(%rsp)	# ivtmp.478, %sfp
	movq	40(%rsp), %rcx	# %sfp, ivtmp.493
	movq	56(%rsp), %rdx	# %sfp, ivtmp.491
	movl	64(%rsp), %eax	# %sfp, j
	.p2align 4,,10
	.p2align 3
.L192:
# benchmark_2mm.c:234:                     #pragma omp simd reduction(+:sum)
	movq	96(%rsp), %rdi	# %sfp, tmp278
	vpxor	%xmm0, %xmm0, %xmm0	# tmp291
# benchmark_2mm.c:233:                     double sum = beta * D[i*nl + j];
	vmulsd	(%rdx), %xmm3, %xmm2	# MEM[(double *)_90], beta, sum
# benchmark_2mm.c:234:                     #pragma omp simd reduction(+:sum)
	vmovdqa	%xmm0, (%rdi)	# tmp291, MEM <char[1:32]> [(void *)&D.36876]
	vmovdqa	%xmm0, 16(%rdi)	# tmp291, MEM <char[1:32]> [(void *)&D.36876]
	testl	%r12d, %r12d	# nj
	jle	.L200	#,
	cmpl	$2, 84(%rsp)	#, %sfp
	jbe	.L204	#,
	movq	88(%rsp), %rdi	# %sfp, ivtmp.478
	movq	104(%rsp), %r8	# %sfp, _144
	movq	%rcx, %rsi	# ivtmp.493, ivtmp.479
	vxorpd	%xmm4, %xmm4, %xmm4	# vect__90.470
	.p2align 4,,10
	.p2align 3
.L198:
# benchmark_2mm.c:236:                         sum += tmp[i*nj + k] * C[k*nl + j];
	vmovsd	(%rsi,%rbx,2), %xmm1	# MEM[(double *)_218 + _167 * 2], MEM[(double *)_218 + _167 * 2]
	vmovsd	(%rsi), %xmm0	# MEM[(double *)_218], MEM[(double *)_218]
	vmovhpd	(%rsi,%r14), %xmm1, %xmm1	# MEM[(double *)_218 + _214 * 1], MEM[(double *)_218 + _167 * 2], tmp256
	vmovhpd	(%rsi,%rbx), %xmm0, %xmm0	# MEM[(double *)_218 + _167 * 1], MEM[(double *)_218], tmp259
	vinsertf128	$0x1, %xmm1, %ymm0, %ymm0	# tmp256, tmp259, tmp255
# benchmark_2mm.c:236:                         sum += tmp[i*nj + k] * C[k*nl + j];
	vfmadd231pd	(%rdi), %ymm0, %ymm4	# MEM <vector(4) double> [(double *)_219], tmp255, vect__90.470
	addq	$32, %rdi	#, ivtmp.478
	addq	%r13, %rsi	# _169, ivtmp.479
	vmovapd	%ymm4, 128(%rsp)	# vect__90.470, MEM <vector(4) double> [(double *)&D.36876]
	cmpq	%rdi, %r8	# ivtmp.478, _144
	jne	.L198	#,
	movl	80(%rsp), %esi	# %sfp, niters_vector_mult_vf.463
	movq	%r8, 104(%rsp)	# _144, %sfp
	cmpl	%r12d, %esi	# nj, niters_vector_mult_vf.463
	je	.L200	#,
	movl	%esi, 124(%rsp)	# niters_vector_mult_vf.463, %sfp
.L197:
# benchmark_2mm.c:236:                         sum += tmp[i*nj + k] * C[k*nl + j];
	movl	120(%rsp), %edi	# %sfp, _16
# benchmark_2mm.c:236:                         sum += tmp[i*nj + k] * C[k*nl + j];
	leal	(%r11,%rsi), %r15d	#, tmp262
# benchmark_2mm.c:236:                         sum += tmp[i*nj + k] * C[k*nl + j];
	imull	%esi, %edi	# k, _16
# benchmark_2mm.c:236:                         sum += tmp[i*nj + k] * C[k*nl + j];
	movq	72(%rsp), %r8	# %sfp, tmp
# benchmark_2mm.c:236:                         sum += tmp[i*nj + k] * C[k*nl + j];
	movslq	%r15d, %r15	# tmp262, tmp263
# benchmark_2mm.c:236:                         sum += tmp[i*nj + k] * C[k*nl + j];
	vmovsd	(%r8,%r15,8), %xmm0	# *_173, _91
	vmovsd	128(%rsp), %xmm5	# MEM[(double *)&D.36876], tmp406
	movq	112(%rsp), %rsi	# %sfp, C
# benchmark_2mm.c:236:                         sum += tmp[i*nj + k] * C[k*nl + j];
	leal	(%rdi,%rax), %r10d	#, tmp264
	movslq	%r10d, %r10	# tmp264, tmp265
# benchmark_2mm.c:236:                         sum += tmp[i*nj + k] * C[k*nl + j];
	vfmadd132sd	(%rsi,%r10,8), %xmm5, %xmm0	# *_139, tmp406, _91
	movl	124(%rsp), %r15d	# %sfp, k
	leal	1(%r15), %r10d	#, k
	vmovsd	%xmm0, 128(%rsp)	# _91, D.36876[0]
	cmpl	%r10d, %r12d	# k, nj
	jle	.L200	#,
# benchmark_2mm.c:236:                         sum += tmp[i*nj + k] * C[k*nl + j];
	addl	%r11d, %r10d	# ivtmp.502, tmp266
# benchmark_2mm.c:236:                         sum += tmp[i*nj + k] * C[k*nl + j];
	addl	120(%rsp), %edi	# %sfp, _201
# benchmark_2mm.c:236:                         sum += tmp[i*nj + k] * C[k*nl + j];
	movslq	%r10d, %r10	# tmp266, tmp267
# benchmark_2mm.c:236:                         sum += tmp[i*nj + k] * C[k*nl + j];
	vmovsd	(%r8,%r10,8), %xmm6	# *_199, tmp409
# benchmark_2mm.c:236:                         sum += tmp[i*nj + k] * C[k*nl + j];
	leal	(%rax,%rdi), %r15d	#, tmp268
	movslq	%r15d, %r15	# tmp268, tmp269
# benchmark_2mm.c:236:                         sum += tmp[i*nj + k] * C[k*nl + j];
	vfmadd231sd	(%rsi,%r15,8), %xmm6, %xmm0	# *_205, tmp409, _209
	movl	124(%rsp), %esi	# %sfp, k
	addl	$2, %esi	#, k
	vmovsd	%xmm0, 128(%rsp)	# _209, D.36876[0]
	cmpl	%esi, %r12d	# k, nj
	jle	.L200	#,
# benchmark_2mm.c:236:                         sum += tmp[i*nj + k] * C[k*nl + j];
	addl	120(%rsp), %edi	# %sfp, tmp270
# benchmark_2mm.c:236:                         sum += tmp[i*nj + k] * C[k*nl + j];
	movq	112(%rsp), %r15	# %sfp, C
# benchmark_2mm.c:236:                         sum += tmp[i*nj + k] * C[k*nl + j];
	addl	%eax, %edi	# j, tmp271
	movslq	%edi, %rdi	# tmp271, tmp272
# benchmark_2mm.c:236:                         sum += tmp[i*nj + k] * C[k*nl + j];
	vmovsd	(%r15,%rdi,8), %xmm7	# *_124, tmp412
# benchmark_2mm.c:236:                         sum += tmp[i*nj + k] * C[k*nl + j];
	addl	%r11d, %esi	# ivtmp.502, tmp273
	movslq	%esi, %rsi	# tmp273, tmp274
# benchmark_2mm.c:236:                         sum += tmp[i*nj + k] * C[k*nl + j];
	vfmadd231sd	(%r8,%rsi,8), %xmm7, %xmm0	# *_130, tmp412, _120
	vmovsd	%xmm0, 128(%rsp)	# _120, D.36876[0]
.L200:
# benchmark_2mm.c:234:                     #pragma omp simd reduction(+:sum)
	vmovapd	128(%rsp), %ymm0	# MEM <vector(4) double> [(double *)&D.36876], vect__72.459
# benchmark_2mm.c:232:                 for (int j = jj; j < j_end; j++) {
	incl	%eax	# j
	vaddsd	%xmm0, %xmm2, %xmm2	# stmp_sum_73.460, sum, stmp_sum_73.460
	vunpckhpd	%xmm0, %xmm0, %xmm1	# tmp247, stmp_sum_73.460
	vextractf128	$0x1, %ymm0, %xmm0	# vect__72.459, tmp249
	vaddsd	%xmm2, %xmm1, %xmm1	# stmp_sum_73.460, stmp_sum_73.460, stmp_sum_73.460
# benchmark_2mm.c:232:                 for (int j = jj; j < j_end; j++) {
	addq	$8, %rdx	#, ivtmp.491
	addq	$8, %rcx	#, ivtmp.493
# benchmark_2mm.c:234:                     #pragma omp simd reduction(+:sum)
	vaddsd	%xmm1, %xmm0, %xmm1	# stmp_sum_73.460, stmp_sum_73.460, stmp_sum_73.460
	vunpckhpd	%xmm0, %xmm0, %xmm0	# tmp249, stmp_sum_73.460
	vaddsd	%xmm0, %xmm1, %xmm1	# stmp_sum_73.460, stmp_sum_73.460, sum
# benchmark_2mm.c:238:                     D[i*nl + j] = sum;
	vmovsd	%xmm1, -8(%rdx)	# sum, MEM[(double *)_90]
# benchmark_2mm.c:232:                 for (int j = jj; j < j_end; j++) {
	cmpl	%eax, %r9d	# j, j_end
	jne	.L192	#,
# benchmark_2mm.c:231:             for (int i = ii; i < i_end; i++) {
	incl	68(%rsp)	# %sfp
# benchmark_2mm.c:231:             for (int i = ii; i < i_end; i++) {
	addq	%rbx, 56(%rsp)	# _167, %sfp
	addl	%r12d, %r11d	# nj, ivtmp.502
# benchmark_2mm.c:231:             for (int i = ii; i < i_end; i++) {
	movl	68(%rsp), %eax	# %sfp, i
# benchmark_2mm.c:231:             for (int i = ii; i < i_end; i++) {
	cmpl	%eax, 52(%rsp)	# i, %sfp
	jne	.L194	#,
.L191:
	movl	20(%rsp), %ecx	# %sfp, ivtmp.503
	cmpl	%ecx, 4(%rsp)	# ivtmp.503, %sfp
	je	.L209	#,
	movl	48(%rsp), %ecx	# %sfp, _52
	cmpl	%ecx, 120(%rsp)	# _52, %sfp
	jle	.L212	#,
.L193:
# benchmark_2mm.c:225:     #pragma omp parallel for collapse(2) schedule(static)
	movl	48(%rsp), %eax	# %sfp, _52
	incl	20(%rsp)	# %sfp
	movl	%eax, 64(%rsp)	# _52, %sfp
	movl	%eax, %edx	# _52, jj
	jmp	.L190	#
.L209:
	vzeroupper
.L186:
	movq	184(%rsp), %rax	# D.38262, tmp297
	subq	%fs:40, %rax	# MEM[(<address-space-1> long unsigned int *)40B], tmp297
	jne	.L213	#,
	leaq	-40(%rbp), %rsp	#,
	popq	%rbx	#
	popq	%r12	#
	popq	%r13	#
	popq	%r14	#
	popq	%r15	#
	popq	%rbp	#
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret	
.L204:
	.cfi_restore_state
# benchmark_2mm.c:234:                     #pragma omp simd reduction(+:sum)
	movl	$0, 124(%rsp)	#, %sfp
	movl	124(%rsp), %esi	# %sfp, k
	jmp	.L197	#
.L212:
# benchmark_2mm.c:225:     #pragma omp parallel for collapse(2) schedule(static)
	movl	28(%rsp), %eax	# %sfp, _50
	movl	$0, 48(%rsp)	#, %sfp
	movl	%eax, 24(%rsp)	# _50, %sfp
	addl	$32, %eax	#, _50
	movl	%eax, 28(%rsp)	# _50, %sfp
	jmp	.L193	#
.L189:
	incl	%ecx	# q.56_1
# benchmark_2mm.c:225:     #pragma omp parallel for collapse(2) schedule(static)
	xorl	%edx, %edx	# tt.57_2
	jmp	.L202	#
.L213:
# benchmark_2mm.c:225:     #pragma omp parallel for collapse(2) schedule(static)
	call	__stack_chk_fail@PLT	#
	.cfi_endproc
.LFE5558:
	.size	kernel_2mm_tiled._omp_fn.2, .-kernel_2mm_tiled._omp_fn.2
	.p2align 4
	.type	kernel_2mm_simd._omp_fn.0, @function
kernel_2mm_simd._omp_fn.0:
.LFB5559:
	.cfi_startproc
	endbr64	
	pushq	%rbp	#
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp	#,
	.cfi_def_cfa_register 6
	pushq	%r15	#
	pushq	%r14	#
	pushq	%r13	#
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	movq	%rdi, %r13	# tmp223, .omp_data_i
	pushq	%r12	#
	pushq	%rbx	#
	andq	$-32, %rsp	#,
	subq	$96, %rsp	#,
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	call	omp_get_num_threads@PLT	#
	movl	%eax, %ebx	# tmp224, _21
	call	omp_get_thread_num@PLT	#
	movl	%eax, %ecx	# tmp225, _22
	movl	32(%r13), %eax	# *.omp_data_i_13(D).ni, *.omp_data_i_13(D).ni
	cltd
	idivl	%ebx	# _21
	cmpl	%edx, %ecx	# tt.84_2, _22
	jl	.L215	#,
.L228:
	imull	%eax, %ecx	# q.83_1, tmp181
	movl	%eax, %esi	# q.83_1, q.83_1
	leal	(%rcx,%rdx), %edi	#, i
	addl	%edi, %esi	# i, q.83_1
	movl	%edi, 68(%rsp)	# i, %sfp
	movl	%esi, 56(%rsp)	# _28, %sfp
	movl	%edi, %eax	# i, i
	cmpl	%esi, %edi	# _28, i
	jge	.L237	#,
# benchmark_2mm.c:259:     #pragma omp parallel for
	movq	8(%r13), %rdi	# *.omp_data_i_13(D).A, A
	movl	36(%r13), %r11d	# *.omp_data_i_13(D).nj, nj
	movq	%rdi, 32(%rsp)	# A, %sfp
	movl	40(%r13), %edi	# *.omp_data_i_13(D).nk, nk
	movl	%eax, %r14d	# i, ivtmp.562
	movl	%edi, %esi	# nk, ivtmp.563
	imull	%eax, %esi	# i, ivtmp.563
# benchmark_2mm.c:264:             tmp[i*nj + j] = 0.0;
	leal	-1(%r11), %eax	#,
	movl	%eax, 84(%rsp)	# _75, %sfp
	leaq	8(,%rax,8), %rax	#, _74
	movq	%rax, 40(%rsp)	# _74, %sfp
	movl	%r11d, %eax	# nj, bnd.514
	shrl	$2, %eax	#, bnd.514
# benchmark_2mm.c:259:     #pragma omp parallel for
	movq	24(%r13), %r12	# *.omp_data_i_13(D).tmp, tmp
	movq	16(%r13), %rbx	# *.omp_data_i_13(D).B, B
	vmovsd	0(%r13), %xmm3	# *.omp_data_i_13(D).alpha, alpha
	leal	-1(%rax), %r13d	#, tmp186
	movl	%r11d, %eax	# nj, niters_vector_mult_vf.515
	andl	$-4, %eax	#, niters_vector_mult_vf.515
	movl	%eax, 80(%rsp)	# niters_vector_mult_vf.515, %sfp
	movslq	%r11d, %rax	# nj, _175
	movq	%rax, 72(%rsp)	# _175, %sfp
	leal	-1(%rdi), %eax	#, tmp220
	movl	%edi, 64(%rsp)	# nk, %sfp
	imull	%r11d, %r14d	# nj, ivtmp.562
	movl	%esi, 60(%rsp)	# ivtmp.563, %sfp
	movl	%eax, 28(%rsp)	# tmp220, %sfp
	incq	%r13	# tmp187
	salq	$5, %r13	#, _67
	movl	%r11d, %r15d	# nj, nj
	.p2align 4,,10
	.p2align 3
.L218:
	testl	%r15d, %r15d	# nj
	jle	.L222	#,
# benchmark_2mm.c:264:             tmp[i*nj + j] = 0.0;
	movslq	%r14d, %r8	# ivtmp.562, ivtmp.562
	leaq	(%r12,%r8,8), %rcx	#, tmp205
	movq	40(%rsp), %rdx	# %sfp,
	xorl	%esi, %esi	#
	movq	%rcx, %rdi	# tmp205,
	movq	%r8, 88(%rsp)	# ivtmp.562, %sfp
	vmovsd	%xmm3, 48(%rsp)	# alpha, %sfp
	call	memset@PLT	#
	movq	%rax, %rcx	#, tmp205
# benchmark_2mm.c:268:         for (int k = 0; k < nk; k++) {
	movl	64(%rsp), %eax	# %sfp,
	movq	88(%rsp), %r8	# %sfp, ivtmp.562
	testl	%eax, %eax	#
	vmovsd	48(%rsp), %xmm3	# %sfp, alpha
	jle	.L222	#,
	movslq	60(%rsp), %rdx	# %sfp, _154
	movl	28(%rsp), %eax	# %sfp, tmp212
	movq	32(%rsp), %rdi	# %sfp, A
	addq	%rdx, %rax	# _154, tmp213
	leaq	8(%rdi,%rax,8), %rax	#, _42
	movq	%rax, 88(%rsp)	# _42, %sfp
	leaq	(%rdi,%rdx,8), %r11	#, ivtmp.553
# benchmark_2mm.c:269:             double aik = alpha * A[i*nk + k];
	xorl	%esi, %esi	# ivtmp.556
	xorl	%edi, %edi	# ivtmp.557
	.p2align 4,,10
	.p2align 3
.L221:
	cmpl	$2, 84(%rsp)	#, %sfp
# benchmark_2mm.c:269:             double aik = alpha * A[i*nk + k];
	vmulsd	(%r11), %xmm3, %xmm0	# MEM[(double *)_186], alpha, aik
	jbe	.L239	#,
	leaq	(%rbx,%rdi,8), %rdx	#, _47
	vbroadcastsd	%xmm0, %ymm2	# aik, vect_cst__77
# benchmark_2mm.c:272:                 tmp[i*nj + j] += aik * B[k*nj + j];
	xorl	%eax, %eax	# ivtmp.546
	.p2align 4,,10
	.p2align 3
.L223:
# benchmark_2mm.c:272:                 tmp[i*nj + j] += aik * B[k*nj + j];
	vmovupd	(%rdx,%rax), %ymm1	# MEM <vector(4) double> [(double *)_47 + ivtmp.546_137 * 1], vect__51.524
	vfmadd213pd	(%rcx,%rax), %ymm2, %ymm1	# MEM <vector(4) double> [(double *)vectp.518_89 + ivtmp.546_137 * 1], vect_cst__77, vect__51.524
	vmovupd	%ymm1, (%rcx,%rax)	# vect__51.524, MEM <vector(4) double> [(double *)vectp.518_89 + ivtmp.546_137 * 1]
	addq	$32, %rax	#, ivtmp.546
	cmpq	%rax, %r13	# ivtmp.546, _67
	jne	.L223	#,
	movl	80(%rsp), %eax	# %sfp, niters_vector_mult_vf.515
# benchmark_2mm.c:272:                 tmp[i*nj + j] += aik * B[k*nj + j];
	movl	%eax, %edx	# niters_vector_mult_vf.515,
	cmpl	%eax, %r15d	# niters_vector_mult_vf.515, nj
	je	.L224	#,
.L219:
	movl	%r15d, %r9d	# nj, niters.527
	subl	%edx, %r9d	# _91, niters.527
	cmpl	$1, %r9d	#, niters.527
	je	.L225	#,
	leaq	(%r8,%rdx), %r10	#, tmp192
	leaq	(%r12,%r10,8), %r10	#, vectp.532
	vmovupd	(%r10), %xmm5	# MEM <vector(2) double> [(double *)vectp.532_162], tmp285
# benchmark_2mm.c:272:                 tmp[i*nj + j] += aik * B[k*nj + j];
	addq	%rdi, %rdx	# ivtmp.557, tmp194
# benchmark_2mm.c:272:                 tmp[i*nj + j] += aik * B[k*nj + j];
	vmovddup	%xmm0, %xmm1	# aik, tmp196
	vfmadd132pd	(%rbx,%rdx,8), %xmm5, %xmm1	# MEM <vector(2) double> [(double *)vectp.535_170], tmp285, vect__103.538
	movl	%r9d, %edx	# niters.527, niters_vector_mult_vf.529
	andl	$-2, %edx	#, niters_vector_mult_vf.529
	addl	%edx, %eax	# niters_vector_mult_vf.529, tmp.530
	vmovupd	%xmm1, (%r10)	# vect__103.538, MEM <vector(2) double> [(double *)vectp.532_162]
	cmpl	%edx, %r9d	# niters_vector_mult_vf.529, niters.527
	je	.L224	#,
.L225:
	leal	(%r14,%rax), %edx	#, tmp198
	movslq	%edx, %rdx	# tmp198, tmp199
	leaq	(%r12,%rdx,8), %rdx	#, _141
	vmovsd	(%rdx), %xmm4	# *_141, tmp287
# benchmark_2mm.c:272:                 tmp[i*nj + j] += aik * B[k*nj + j];
	addl	%esi, %eax	# ivtmp.556, tmp201
	cltq
# benchmark_2mm.c:272:                 tmp[i*nj + j] += aik * B[k*nj + j];
	vfmadd132sd	(%rbx,%rax,8), %xmm4, %xmm0	# *_146, tmp287, _149
	vmovsd	%xmm0, (%rdx)	# _149, *_141
.L224:
# benchmark_2mm.c:268:         for (int k = 0; k < nk; k++) {
	addq	$8, %r11	#, ivtmp.553
	addl	%r15d, %esi	# nj, ivtmp.556
	addq	72(%rsp), %rdi	# %sfp, ivtmp.557
	cmpq	%r11, 88(%rsp)	# ivtmp.553, %sfp
	jne	.L221	#,
	vzeroupper
.L222:
	incl	68(%rsp)	# %sfp
	movl	64(%rsp), %esi	# %sfp, nk
	addl	%r15d, %r14d	# nj, ivtmp.562
	addl	%esi, 60(%rsp)	# nk, %sfp
	movl	68(%rsp), %eax	# %sfp, i
	cmpl	%eax, 56(%rsp)	# i, %sfp
	jne	.L218	#,
.L237:
# benchmark_2mm.c:259:     #pragma omp parallel for
	leaq	-40(%rbp), %rsp	#,
	popq	%rbx	#
	popq	%r12	#
	popq	%r13	#
	popq	%r14	#
	popq	%r15	#
	popq	%rbp	#
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret	
	.p2align 4,,10
	.p2align 3
.L239:
	.cfi_restore_state
# benchmark_2mm.c:272:                 tmp[i*nj + j] += aik * B[k*nj + j];
	xorl	%edx, %edx	#
	xorl	%eax, %eax	# tmp.530
	jmp	.L219	#
.L215:
	incl	%eax	# q.83_1
# benchmark_2mm.c:259:     #pragma omp parallel for
	xorl	%edx, %edx	# tt.84_2
	jmp	.L228	#
	.cfi_endproc
.LFE5559:
	.size	kernel_2mm_simd._omp_fn.0, .-kernel_2mm_simd._omp_fn.0
	.p2align 4
	.type	kernel_2mm_simd._omp_fn.1, @function
kernel_2mm_simd._omp_fn.1:
.LFB5560:
	.cfi_startproc
	endbr64	
	pushq	%rbp	#
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp	#,
	.cfi_def_cfa_register 6
	pushq	%r15	#
	pushq	%r14	#
	pushq	%r13	#
	pushq	%r12	#
	pushq	%rbx	#
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movq	%rdi, %rbx	# tmp248, .omp_data_i
	andq	$-32, %rsp	#,
	subq	$160, %rsp	#,
# benchmark_2mm.c:278:     #pragma omp parallel for
	movq	%fs:40, %rax	# MEM[(<address-space-1> long unsigned int *)40B], tmp251
	movq	%rax, 152(%rsp)	# tmp251, D.38389
	xorl	%eax, %eax	# tmp251
	call	omp_get_num_threads@PLT	#
	movl	%eax, %r12d	# tmp249, _23
	call	omp_get_thread_num@PLT	#
	movl	%eax, %ecx	# tmp250, _24
	movl	32(%rbx), %eax	# *.omp_data_i_15(D).ni, *.omp_data_i_15(D).ni
	cltd
	idivl	%r12d	# _23
	cmpl	%edx, %ecx	# tt.82_2, _24
	jl	.L241	#,
.L252:
	imull	%eax, %ecx	# q.81_1, tmp196
	addl	%edx, %ecx	# tt.82_2, i
	movl	%eax, %edx	# q.81_1, q.81_1
	addl	%ecx, %edx	# i, q.81_1
	movl	%ecx, 36(%rsp)	# i, %sfp
	movl	%edx, 32(%rsp)	# _30, %sfp
	movl	%ecx, %eax	# i, i
	cmpl	%edx, %ecx	# _30, i
	jge	.L240	#,
	movq	24(%rbx), %rcx	# *.omp_data_i_15(D).tmp, tmp
	movq	8(%rbx), %rdx	# *.omp_data_i_15(D).C, C
	movl	40(%rbx), %r9d	# *.omp_data_i_15(D).nl, nl
	movq	%rcx, 48(%rsp)	# tmp, %sfp
	movq	%rdx, 40(%rsp)	# C, %sfp
	movq	16(%rbx), %rcx	# *.omp_data_i_15(D).D, D
	vmovsd	(%rbx), %xmm3	# *.omp_data_i_15(D).beta, beta
	movl	36(%rbx), %r13d	# *.omp_data_i_15(D).nj, nj
	testl	%r9d, %r9d	# nl
	jle	.L240	#,
	movl	%eax, %ebx	# i, i
	imull	%r9d, %eax	# nl, tmp197
	movslq	%r9d, %rdx	# nl, _146
	leaq	(%rdx,%rdx,2), %r11	#, tmp204
	cltq
	leaq	(%rcx,%rax,8), %rax	#, ivtmp.616
	movq	%rax, 24(%rsp)	# ivtmp.616, %sfp
	movl	%ebx, %eax	# i, ivtmp.618
	imull	%r13d, %eax	# nj, ivtmp.618
	movl	%r13d, %ecx	# nj, niters_vector_mult_vf.582
	andl	$-4, %ecx	#, niters_vector_mult_vf.582
	movl	%eax, %r14d	# ivtmp.618, ivtmp.618
	leal	-1(%r13), %eax	#, _126
	movl	%eax, 60(%rsp)	# _126, %sfp
	movl	%r13d, %eax	# nj, bnd.581
	shrl	$2, %eax	#, bnd.581
	decl	%eax	# _81
	incq	%rax	# tmp241
	movl	%ecx, 56(%rsp)	# niters_vector_mult_vf.582, %sfp
	salq	$5, %rax	#, tmp241
	leaq	96(%rsp), %rcx	#, tmp339
	movq	%rcx, 72(%rsp)	# tmp339, %sfp
	movq	%rax, 16(%rsp)	# tmp241, %sfp
	movq	%rdx, %rbx	# _146, _143
	leaq	0(,%rdx,8), %rsi	#, _4
	salq	$5, %rbx	#, _143
	salq	$3, %r11	#, tmp205
	.p2align 4,,10
	.p2align 3
.L244:
	movq	48(%rsp), %rdx	# %sfp, tmp
	movq	16(%rsp), %r10	# %sfp, _11
	movslq	%r14d, %rax	# ivtmp.618, _49
	leaq	(%rdx,%rax,8), %rax	#, ivtmp.595
	addq	%rax, %r10	# ivtmp.595, _11
	movq	%rsi, 88(%rsp)	# _4, %sfp
	movq	%r10, 80(%rsp)	# _11, %sfp
	movq	40(%rsp), %rcx	# %sfp, ivtmp.610
	movq	24(%rsp), %rdx	# %sfp, ivtmp.608
	movq	%rax, 64(%rsp)	# ivtmp.595, %sfp
# benchmark_2mm.c:280:         for (int j = 0; j < nl; j++) {
	xorl	%eax, %eax	# j
	.p2align 4,,10
	.p2align 3
.L243:
# benchmark_2mm.c:282:             #pragma omp simd reduction(+:sum) aligned(tmp,C:ALIGN_SIZE)
	movq	72(%rsp), %rdi	# %sfp, tmp240
	vpxor	%xmm0, %xmm0, %xmm0	# tmp246
# benchmark_2mm.c:281:             double sum = beta * D[i*nl + j];
	vmulsd	(%rdx), %xmm3, %xmm2	# MEM[(double *)_61], beta, sum
# benchmark_2mm.c:282:             #pragma omp simd reduction(+:sum) aligned(tmp,C:ALIGN_SIZE)
	vmovdqa	%xmm0, (%rdi)	# tmp246, MEM <char[1:32]> [(void *)&D.37169]
	vmovdqa	%xmm0, 16(%rdi)	# tmp246, MEM <char[1:32]> [(void *)&D.37169]
	testl	%r13d, %r13d	# nj
	jle	.L250	#,
	cmpl	$2, 60(%rsp)	#, %sfp
	jbe	.L254	#,
	movq	64(%rsp), %r8	# %sfp, ivtmp.595
	movq	88(%rsp), %rsi	# %sfp, _4
	movq	80(%rsp), %r10	# %sfp, _11
	movq	%rcx, %rdi	# ivtmp.610, ivtmp.598
	vxorpd	%xmm4, %xmm4, %xmm4	# vect__63.589
	.p2align 4,,10
	.p2align 3
.L248:
# benchmark_2mm.c:284:                 sum += tmp[i*nj + k] * C[k*nl + j];
	vmovsd	(%rdi,%rsi,2), %xmm1	# MEM[(double *)_114 + _4 * 2], MEM[(double *)_114 + _4 * 2]
	vmovsd	(%rdi), %xmm0	# MEM[(double *)_114], MEM[(double *)_114]
	vmovhpd	(%rdi,%r11), %xmm1, %xmm1	# MEM[(double *)_114 + _101 * 1], MEM[(double *)_114 + _4 * 2], tmp220
	vmovhpd	(%rdi,%rsi), %xmm0, %xmm0	# MEM[(double *)_114 + _4 * 1], MEM[(double *)_114], tmp223
	vinsertf128	$0x1, %xmm1, %ymm0, %ymm0	# tmp220, tmp223, tmp219
# benchmark_2mm.c:284:                 sum += tmp[i*nj + k] * C[k*nl + j];
	vfmadd231pd	(%r8), %ymm0, %ymm4	# MEM <vector(4) double> [(double *)_186], tmp219, vect__63.589
	addq	$32, %r8	#, ivtmp.595
	addq	%rbx, %rdi	# _143, ivtmp.598
	vmovapd	%ymm4, 96(%rsp)	# vect__63.589, MEM <vector(4) double> [(double *)&D.37169]
	cmpq	%r8, %r10	# ivtmp.595, _11
	jne	.L248	#,
	movq	%rsi, 88(%rsp)	# _4, %sfp
	movl	56(%rsp), %esi	# %sfp, niters_vector_mult_vf.582
	movq	%r10, 80(%rsp)	# _11, %sfp
	movl	%esi, %edi	# niters_vector_mult_vf.582, k
	cmpl	%esi, %r13d	# niters_vector_mult_vf.582, nj
	je	.L250	#,
.L247:
# benchmark_2mm.c:284:                 sum += tmp[i*nj + k] * C[k*nl + j];
	movl	%r9d, %r8d	# nl, _12
	imull	%edi, %r8d	# k, _12
# benchmark_2mm.c:284:                 sum += tmp[i*nj + k] * C[k*nl + j];
	movq	48(%rsp), %r10	# %sfp, tmp
# benchmark_2mm.c:284:                 sum += tmp[i*nj + k] * C[k*nl + j];
	leal	(%r14,%rdi), %r15d	#, tmp226
	movslq	%r15d, %r15	# tmp226, tmp227
# benchmark_2mm.c:284:                 sum += tmp[i*nj + k] * C[k*nl + j];
	vmovsd	96(%rsp), %xmm5	# MEM[(double *)&D.37169], tmp314
	movq	40(%rsp), %rsi	# %sfp, C
	vmovsd	(%r10,%r15,8), %xmm0	# *_124, _64
# benchmark_2mm.c:284:                 sum += tmp[i*nj + k] * C[k*nl + j];
	leal	(%r8,%rax), %r12d	#, tmp228
	movslq	%r12d, %r12	# tmp228, tmp229
# benchmark_2mm.c:284:                 sum += tmp[i*nj + k] * C[k*nl + j];
	vfmadd132sd	(%rsi,%r12,8), %xmm5, %xmm0	# *_96, tmp314, _64
	leal	1(%rdi), %r12d	#, k
	vmovsd	%xmm0, 96(%rsp)	# _64, D.37169[0]
	cmpl	%r12d, %r13d	# k, nj
	jle	.L250	#,
# benchmark_2mm.c:284:                 sum += tmp[i*nj + k] * C[k*nl + j];
	addl	%r14d, %r12d	# ivtmp.618, tmp230
# benchmark_2mm.c:284:                 sum += tmp[i*nj + k] * C[k*nl + j];
	addl	%r9d, %r8d	# nl, _174
# benchmark_2mm.c:284:                 sum += tmp[i*nj + k] * C[k*nl + j];
	movslq	%r12d, %r12	# tmp230, tmp231
# benchmark_2mm.c:284:                 sum += tmp[i*nj + k] * C[k*nl + j];
	vmovsd	(%r10,%r12,8), %xmm6	# *_172, tmp317
# benchmark_2mm.c:284:                 sum += tmp[i*nj + k] * C[k*nl + j];
	leal	(%rax,%r8), %r15d	#, tmp232
	movslq	%r15d, %r15	# tmp232, tmp233
# benchmark_2mm.c:284:                 sum += tmp[i*nj + k] * C[k*nl + j];
	vfmadd231sd	(%rsi,%r15,8), %xmm6, %xmm0	# *_178, tmp317, _182
	addl	$2, %edi	#, k
	vmovsd	%xmm0, 96(%rsp)	# _182, D.37169[0]
	cmpl	%edi, %r13d	# k, nj
	jle	.L250	#,
# benchmark_2mm.c:284:                 sum += tmp[i*nj + k] * C[k*nl + j];
	addl	%r9d, %r8d	# nl, tmp234
# benchmark_2mm.c:284:                 sum += tmp[i*nj + k] * C[k*nl + j];
	addl	%eax, %r8d	# j, tmp235
	movslq	%r8d, %r8	# tmp235, tmp236
# benchmark_2mm.c:284:                 sum += tmp[i*nj + k] * C[k*nl + j];
	vmovsd	(%rsi,%r8,8), %xmm7	# *_69, tmp320
# benchmark_2mm.c:284:                 sum += tmp[i*nj + k] * C[k*nl + j];
	addl	%r14d, %edi	# ivtmp.618, tmp237
	movslq	%edi, %rdi	# tmp237, tmp238
# benchmark_2mm.c:284:                 sum += tmp[i*nj + k] * C[k*nl + j];
	vfmadd231sd	(%r10,%rdi,8), %xmm7, %xmm0	# *_87, tmp320, _8
	vmovsd	%xmm0, 96(%rsp)	# _8, D.37169[0]
.L250:
# benchmark_2mm.c:282:             #pragma omp simd reduction(+:sum) aligned(tmp,C:ALIGN_SIZE)
	vmovapd	96(%rsp), %ymm0	# MEM <vector(4) double> [(double *)&D.37169], vect__45.578
# benchmark_2mm.c:280:         for (int j = 0; j < nl; j++) {
	incl	%eax	# j
	vaddsd	%xmm0, %xmm2, %xmm2	# stmp_sum_46.579, sum, stmp_sum_46.579
	vunpckhpd	%xmm0, %xmm0, %xmm1	# tmp211, stmp_sum_46.579
	vextractf128	$0x1, %ymm0, %xmm0	# vect__45.578, tmp213
	vaddsd	%xmm2, %xmm1, %xmm1	# stmp_sum_46.579, stmp_sum_46.579, stmp_sum_46.579
# benchmark_2mm.c:280:         for (int j = 0; j < nl; j++) {
	addq	$8, %rdx	#, ivtmp.608
	addq	$8, %rcx	#, ivtmp.610
# benchmark_2mm.c:282:             #pragma omp simd reduction(+:sum) aligned(tmp,C:ALIGN_SIZE)
	vaddsd	%xmm1, %xmm0, %xmm1	# stmp_sum_46.579, stmp_sum_46.579, stmp_sum_46.579
	vunpckhpd	%xmm0, %xmm0, %xmm0	# tmp213, stmp_sum_46.579
	vaddsd	%xmm0, %xmm1, %xmm1	# stmp_sum_46.579, stmp_sum_46.579, sum
# benchmark_2mm.c:286:             D[i*nl + j] = sum;
	vmovsd	%xmm1, -8(%rdx)	# sum, MEM[(double *)_61]
# benchmark_2mm.c:280:         for (int j = 0; j < nl; j++) {
	cmpl	%eax, %r9d	# j, nl
	jne	.L243	#,
	incl	36(%rsp)	# %sfp
	movq	88(%rsp), %rsi	# %sfp, _4
	addl	%r13d, %r14d	# nj, ivtmp.618
	addq	%rsi, 24(%rsp)	# _4, %sfp
	movl	36(%rsp), %eax	# %sfp, i
	cmpl	%eax, 32(%rsp)	# i, %sfp
	jne	.L244	#,
	vzeroupper
.L240:
# benchmark_2mm.c:278:     #pragma omp parallel for
	movq	152(%rsp), %rax	# D.38389, tmp252
	subq	%fs:40, %rax	# MEM[(<address-space-1> long unsigned int *)40B], tmp252
	jne	.L262	#,
	leaq	-40(%rbp), %rsp	#,
	popq	%rbx	#
	popq	%r12	#
	popq	%r13	#
	popq	%r14	#
	popq	%r15	#
	popq	%rbp	#
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret	
.L254:
	.cfi_restore_state
# benchmark_2mm.c:282:             #pragma omp simd reduction(+:sum) aligned(tmp,C:ALIGN_SIZE)
	xorl	%edi, %edi	# k
	jmp	.L247	#
.L241:
	incl	%eax	# q.81_1
# benchmark_2mm.c:278:     #pragma omp parallel for
	xorl	%edx, %edx	# tt.82_2
	jmp	.L252	#
.L262:
# benchmark_2mm.c:278:     #pragma omp parallel for
	call	__stack_chk_fail@PLT	#
	.cfi_endproc
.LFE5560:
	.size	kernel_2mm_simd._omp_fn.1, .-kernel_2mm_simd._omp_fn.1
	.p2align 4
	.globl	kernel_2mm_simd
	.type	kernel_2mm_simd, @function
kernel_2mm_simd:
.LFB5549:
	.cfi_startproc
	endbr64	
	pushq	%r15	#
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	vmovd	%edi, %xmm3	# tmp109, tmp109
# benchmark_2mm.c:259:     #pragma omp parallel for
	vmovq	%r8, %xmm4	# tmp115, tmp115
# benchmark_2mm.c:250:                     double *__restrict__ tmp) {
	pushq	%r14	#
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	vmovsd	%xmm0, %xmm0, %xmm2	# alpha, tmp113
	leaq	kernel_2mm_simd._omp_fn.0(%rip), %rdi	#, tmp104
	pushq	%r13	#
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	vpinsrd	$1, %esi, %xmm3, %xmm0	# tmp110, tmp109, tmp101
	pushq	%r12	#
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp	#
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	vmovq	%xmm1, %rbp	# tmp114, beta
# benchmark_2mm.c:259:     #pragma omp parallel for
	vpinsrq	$1, %r9, %xmm4, %xmm1	# tmp116, tmp115, tmp102
# benchmark_2mm.c:250:                     double *__restrict__ tmp) {
	pushq	%rbx	#
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movl	%ecx, %ebx	# tmp112, nl
	xorl	%ecx, %ecx	#
	subq	$88, %rsp	#,
	.cfi_def_cfa_offset 144
# benchmark_2mm.c:250:                     double *__restrict__ tmp) {
	movq	160(%rsp), %r15	# tmp, tmp
	movq	144(%rsp), %r12	# C, C
	leaq	16(%rsp), %r14	#, tmp103
	movq	%fs:40, %rax	# MEM[(<address-space-1> long unsigned int *)40B], tmp117
	movq	%rax, 72(%rsp)	# tmp117, D.38414
	xorl	%eax, %eax	# tmp117
	movq	152(%rsp), %r13	# D, D
# benchmark_2mm.c:259:     #pragma omp parallel for
	movl	%edx, 56(%rsp)	# tmp111, MEM[(struct .omp_data_s.71 *)_36].nk
	movq	%r14, %rsi	# tmp103,
	xorl	%edx, %edx	#
	vmovdqu	%xmm1, 24(%rsp)	# tmp102, MEM <vector(2) long unsigned int> [(double * *)_36]
	vmovq	%xmm0, 48(%rsp)	# tmp101, MEM <vector(2) int> [(int *)_36]
	vmovq	%xmm0, 8(%rsp)	# tmp101, %sfp
	movq	%r15, 40(%rsp)	# tmp, MEM[(struct .omp_data_s.71 *)_36].tmp
	vmovsd	%xmm2, 16(%rsp)	# tmp113, MEM[(struct .omp_data_s.71 *)_36].alpha
	call	GOMP_parallel@PLT	#
# benchmark_2mm.c:278:     #pragma omp parallel for
	vmovq	8(%rsp), %xmm0	# %sfp, tmp101
	vmovq	%r12, %xmm5	# C, C
	vpinsrq	$1, %r13, %xmm5, %xmm1	# D, C, tmp105
	xorl	%ecx, %ecx	#
	xorl	%edx, %edx	#
	movq	%r14, %rsi	# tmp103,
	leaq	kernel_2mm_simd._omp_fn.1(%rip), %rdi	#, tmp107
	movq	%r15, 40(%rsp)	# tmp, MEM[(struct .omp_data_s.72 *)_36].tmp
	movq	%rbp, 16(%rsp)	# beta, MEM[(struct .omp_data_s.72 *)_36].beta
	movl	%ebx, 56(%rsp)	# nl, MEM[(struct .omp_data_s.72 *)_36].nl
	vmovdqu	%xmm1, 24(%rsp)	# tmp105, MEM <vector(2) long unsigned int> [(double * *)_36]
	vmovq	%xmm0, 48(%rsp)	# tmp101, MEM <vector(2) int> [(int *)_36]
	call	GOMP_parallel@PLT	#
# benchmark_2mm.c:289: }
	movq	72(%rsp), %rax	# D.38414, tmp118
	subq	%fs:40, %rax	# MEM[(<address-space-1> long unsigned int *)40B], tmp118
	jne	.L267	#,
	addq	$88, %rsp	#,
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx	#
	.cfi_def_cfa_offset 48
	popq	%rbp	#
	.cfi_def_cfa_offset 40
	popq	%r12	#
	.cfi_def_cfa_offset 32
	popq	%r13	#
	.cfi_def_cfa_offset 24
	popq	%r14	#
	.cfi_def_cfa_offset 16
	popq	%r15	#
	.cfi_def_cfa_offset 8
	ret	
.L267:
	.cfi_restore_state
	call	__stack_chk_fail@PLT	#
	.cfi_endproc
.LFE5549:
	.size	kernel_2mm_simd, .-kernel_2mm_simd
	.p2align 4
	.type	kernel_2mm_tasks._omp_fn.1, @function
kernel_2mm_tasks._omp_fn.1:
.LFB5562:
	.cfi_startproc
	endbr64	
	pushq	%r15	#
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	movq	%rdi, %rax	# tmp201, .omp_data_i
	pushq	%r14	#
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13	#
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12	#
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp	#
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx	#
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
# benchmark_2mm.c:307:                         int i_end = (i + CHUNK < ni) ? i + CHUNK : ni;
	movq	(%rdi), %rcx	# *.omp_data_i_8(D).ni, *.omp_data_i_8(D).ni
# benchmark_2mm.c:305:                     #pragma omp task depend(out:tmp[i*nj+j:CHUNK*CHUNK])
	movl	56(%rdi), %ebx	# *.omp_data_i_8(D).i, i
# benchmark_2mm.c:307:                         int i_end = (i + CHUNK < ni) ? i + CHUNK : ni;
	movl	(%rcx), %ecx	# *_12, tmp202
# benchmark_2mm.c:307:                         int i_end = (i + CHUNK < ni) ? i + CHUNK : ni;
	leal	32(%rbx), %esi	#, tmp166
# benchmark_2mm.c:307:                         int i_end = (i + CHUNK < ni) ? i + CHUNK : ni;
	cmpl	%ecx, %esi	# tmp202, tmp166
	cmovle	%esi, %ecx	# tmp166,, tmp202
# benchmark_2mm.c:305:                     #pragma omp task depend(out:tmp[i*nj+j:CHUNK*CHUNK])
	movl	60(%rdi), %edx	# *.omp_data_i_8(D).j, j
# benchmark_2mm.c:307:                         int i_end = (i + CHUNK < ni) ? i + CHUNK : ni;
	movl	%ecx, -44(%rsp)	# i_end, %sfp
	movl	%ecx, %esi	# tmp202, i_end
# benchmark_2mm.c:308:                         int j_end = (j + CHUNK < nj) ? j + CHUNK : nj;
	movq	8(%rdi), %rcx	# *.omp_data_i_8(D).nj, *.omp_data_i_8(D).nj
# benchmark_2mm.c:305:                     #pragma omp task depend(out:tmp[i*nj+j:CHUNK*CHUNK])
	movl	%ebx, -52(%rsp)	# i, %sfp
# benchmark_2mm.c:308:                         int j_end = (j + CHUNK < nj) ? j + CHUNK : nj;
	movl	(%rcx), %edi	# *_16, _17
# benchmark_2mm.c:308:                         int j_end = (j + CHUNK < nj) ? j + CHUNK : nj;
	leal	32(%rdx), %ecx	#, tmp168
# benchmark_2mm.c:308:                         int j_end = (j + CHUNK < nj) ? j + CHUNK : nj;
	cmpl	%edi, %ecx	# _17, tmp168
	cmovg	%edi, %ecx	# tmp168,, _17, tmp168
	movl	%ecx, -56(%rsp)	# j_end, %sfp
# benchmark_2mm.c:310:                         for (int ii = i; ii < i_end; ii++) {
	cmpl	%esi, %ebx	# i_end, i
	jge	.L283	#,
	movl	%ecx, %r14d	# tmp168, j_end
# benchmark_2mm.c:313:                                 for (int k = 0; k < nk; k++) {
	movq	16(%rax), %rcx	# *.omp_data_i_8(D).nk, *.omp_data_i_8(D).nk
# benchmark_2mm.c:314:                                     sum += alpha * A[ii*nk + k] * B[k*nj + jj];
	movq	24(%rax), %rsi	# *.omp_data_i_8(D).alpha, _40
# benchmark_2mm.c:313:                                 for (int k = 0; k < nk; k++) {
	movl	(%rcx), %r13d	# *_59, _60
# benchmark_2mm.c:314:                                     sum += alpha * A[ii*nk + k] * B[k*nj + jj];
	movq	32(%rax), %rcx	# *.omp_data_i_8(D).A, *.omp_data_i_8(D).A
# benchmark_2mm.c:314:                                     sum += alpha * A[ii*nk + k] * B[k*nj + jj];
	movq	%rsi, -64(%rsp)	# _40, %sfp
# benchmark_2mm.c:314:                                     sum += alpha * A[ii*nk + k] * B[k*nj + jj];
	movq	(%rcx), %r8	# *_36, _37
# benchmark_2mm.c:314:                                     sum += alpha * A[ii*nk + k] * B[k*nj + jj];
	movq	40(%rax), %rcx	# *.omp_data_i_8(D).B, *.omp_data_i_8(D).B
# benchmark_2mm.c:316:                                 tmp[ii*nj + jj] = sum;
	movq	48(%rax), %rax	# *.omp_data_i_8(D).tmp, *.omp_data_i_8(D).tmp
# benchmark_2mm.c:314:                                     sum += alpha * A[ii*nk + k] * B[k*nj + jj];
	movq	(%rcx), %r9	# *_47, _48
# benchmark_2mm.c:316:                                 tmp[ii*nj + jj] = sum;
	movq	(%rax), %rax	# *_27, _28
	movq	%rax, -40(%rsp)	# _28, %sfp
	cmpl	%r14d, %edx	# j_end, j
	jge	.L283	#,
	movl	%ebx, %eax	# i, i
	imull	%edi, %ebx	# _17, ivtmp.671
	movl	%r13d, %r15d	# _60, niters_vector_mult_vf.638
	andl	$-2, %r15d	#, niters_vector_mult_vf.638
	movl	%ebx, -48(%rsp)	# ivtmp.671, %sfp
	movl	%eax, %ebx	# i, ivtmp.672
	movslq	%edx, %rax	# j, _53
	movq	%rax, -32(%rsp)	# _53, %sfp
	leaq	(%r9,%rax,8), %rax	#, ivtmp.664
	movq	%rax, -24(%rsp)	# ivtmp.664, %sfp
	movl	%r13d, %eax	# _60, bnd.637
	shrl	%eax	# bnd.637
	imull	%r13d, %ebx	# _60, ivtmp.672
	decl	%eax	# _22
	incq	%rax	# tmp195
	salq	$4, %rax	#, tmp195
	movl	%ebx, %esi	# ivtmp.672, ivtmp.672
	movq	%rax, -16(%rsp)	# tmp195, %sfp
	movslq	%edi, %rbx	# _17, _86
	movq	%rbx, %rbp	# _86, _85
	salq	$4, %rbp	#, _85
	salq	$3, %rbx	#, _81
	vxorpd	%xmm5, %xmm5, %xmm5	# sum
	.p2align 4,,10
	.p2align 3
.L274:
	movq	-16(%rsp), %r11	# %sfp, _128
	movslq	%esi, %rax	# ivtmp.672, _32
	leaq	(%r8,%rax,8), %rax	#, ivtmp.655
	movq	%rax, -72(%rsp)	# ivtmp.655, %sfp
	movq	-40(%rsp), %rcx	# %sfp, _28
	addq	%rax, %r11	# ivtmp.655, _128
	movslq	-48(%rsp), %rax	# %sfp, ivtmp.671
# benchmark_2mm.c:305:                     #pragma omp task depend(out:tmp[i*nj+j:CHUNK*CHUNK])
	movq	-24(%rsp), %r12	# %sfp, ivtmp.664
	movq	-32(%rsp), %r10	# %sfp, ivtmp.660
	leaq	(%rcx,%rax,8), %r14	#, _45
	.p2align 4,,10
	.p2align 3
.L271:
	movl	%r10d, %ecx	# ivtmp.660, jj
# benchmark_2mm.c:313:                                 for (int k = 0; k < nk; k++) {
	testl	%r13d, %r13d	# _60
	jle	.L285	#,
# benchmark_2mm.c:314:                                     sum += alpha * A[ii*nk + k] * B[k*nj + jj];
	movq	-64(%rsp), %rax	# %sfp, _40
	vmovsd	(%rax), %xmm3	# *_40, _41
	cmpl	$1, %r13d	#, _60
	je	.L279	#,
	movq	-72(%rsp), %rdx	# %sfp, ivtmp.655
	vmovddup	%xmm3, %xmm4	# _41, vect_cst__91
	movq	%r12, %rax	# ivtmp.664, ivtmp.652
# benchmark_2mm.c:312:                                 double sum = 0.0;
	vmovsd	%xmm5, %xmm5, %xmm1	# sum, sum
	.p2align 4,,10
	.p2align 3
.L276:
# benchmark_2mm.c:314:                                     sum += alpha * A[ii*nk + k] * B[k*nj + jj];
	vmulpd	(%rdx), %xmm4, %xmm2	# MEM <vector(2) double> [(double *)_2], vect_cst__91, vect__42.643
# benchmark_2mm.c:314:                                     sum += alpha * A[ii*nk + k] * B[k*nj + jj];
	vmovsd	(%rax), %xmm0	# MEM[(double *)_3], MEM[(double *)_3]
	addq	$16, %rdx	#, ivtmp.655
	vmovhpd	(%rax,%rbx), %xmm0, %xmm0	# MEM[(double *)_3 + _81 * 1], MEM[(double *)_3], tmp183
	addq	%rbp, %rax	# _85, ivtmp.652
# benchmark_2mm.c:314:                                     sum += alpha * A[ii*nk + k] * B[k*nj + jj];
	vmulpd	%xmm2, %xmm0, %xmm0	# vect__42.643, tmp183, vect__51.644
	vaddsd	%xmm0, %xmm1, %xmm1	# stmp_sum_52.645, sum, stmp_sum_52.645
# benchmark_2mm.c:314:                                     sum += alpha * A[ii*nk + k] * B[k*nj + jj];
	vunpckhpd	%xmm0, %xmm0, %xmm0	# vect__51.644, stmp_sum_52.645
	vaddsd	%xmm1, %xmm0, %xmm1	# stmp_sum_52.645, stmp_sum_52.645, sum
	cmpq	%r11, %rdx	# _128, ivtmp.655
	jne	.L276	#,
# benchmark_2mm.c:313:                                 for (int k = 0; k < nk; k++) {
	movl	%r15d, %eax	# niters_vector_mult_vf.638, k
	cmpl	%r13d, %r15d	# _60, niters_vector_mult_vf.638
	je	.L273	#,
.L275:
# benchmark_2mm.c:314:                                     sum += alpha * A[ii*nk + k] * B[k*nj + jj];
	movl	%edi, %edx	# _17, tmp189
	imull	%eax, %edx	# k, tmp189
# benchmark_2mm.c:314:                                     sum += alpha * A[ii*nk + k] * B[k*nj + jj];
	addl	%esi, %eax	# ivtmp.672, tmp192
	cltq
# benchmark_2mm.c:314:                                     sum += alpha * A[ii*nk + k] * B[k*nj + jj];
	vmulsd	(%r8,%rax,8), %xmm3, %xmm3	# *_116, _41, tmp194
# benchmark_2mm.c:314:                                     sum += alpha * A[ii*nk + k] * B[k*nj + jj];
	addl	%ecx, %edx	# jj, tmp190
	movslq	%edx, %rdx	# tmp190, tmp191
# benchmark_2mm.c:314:                                     sum += alpha * A[ii*nk + k] * B[k*nj + jj];
	vfmadd231sd	(%r9,%rdx,8), %xmm3, %xmm1	# *_109, tmp194, sum
.L273:
# benchmark_2mm.c:316:                                 tmp[ii*nj + jj] = sum;
	vmovsd	%xmm1, (%r14,%r10,8)	# sum, MEM[(double *)_45 + ivtmp.660_127 * 8]
# benchmark_2mm.c:311:                             for (int jj = j; jj < j_end; jj++) {
	incq	%r10	# ivtmp.660
	addq	$8, %r12	#, ivtmp.664
	cmpl	%r10d, -56(%rsp)	# ivtmp.660, %sfp
	jg	.L271	#,
# benchmark_2mm.c:310:                         for (int ii = i; ii < i_end; ii++) {
	incl	-52(%rsp)	# %sfp
# benchmark_2mm.c:310:                         for (int ii = i; ii < i_end; ii++) {
	addl	%edi, -48(%rsp)	# _17, %sfp
	addl	%r13d, %esi	# _60, ivtmp.672
# benchmark_2mm.c:310:                         for (int ii = i; ii < i_end; ii++) {
	movl	-52(%rsp), %eax	# %sfp, i
# benchmark_2mm.c:310:                         for (int ii = i; ii < i_end; ii++) {
	cmpl	%eax, -44(%rsp)	# i, %sfp
	jne	.L274	#,
.L283:
# benchmark_2mm.c:305:                     #pragma omp task depend(out:tmp[i*nj+j:CHUNK*CHUNK])
	popq	%rbx	#
	.cfi_remember_state
	.cfi_def_cfa_offset 48
	popq	%rbp	#
	.cfi_def_cfa_offset 40
	popq	%r12	#
	.cfi_def_cfa_offset 32
	popq	%r13	#
	.cfi_def_cfa_offset 24
	popq	%r14	#
	.cfi_def_cfa_offset 16
	popq	%r15	#
	.cfi_def_cfa_offset 8
	ret	
	.p2align 4,,10
	.p2align 3
.L285:
	.cfi_restore_state
# benchmark_2mm.c:312:                                 double sum = 0.0;
	vmovsd	%xmm5, %xmm5, %xmm1	# sum, sum
	jmp	.L273	#
.L279:
# benchmark_2mm.c:313:                                 for (int k = 0; k < nk; k++) {
	xorl	%eax, %eax	# k
# benchmark_2mm.c:312:                                 double sum = 0.0;
	vmovsd	%xmm5, %xmm5, %xmm1	# sum, sum
	jmp	.L275	#
	.cfi_endproc
.LFE5562:
	.size	kernel_2mm_tasks._omp_fn.1, .-kernel_2mm_tasks._omp_fn.1
	.p2align 4
	.type	kernel_2mm_tasks._omp_fn.2, @function
kernel_2mm_tasks._omp_fn.2:
.LFB5563:
	.cfi_startproc
	endbr64	
	pushq	%r15	#
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	movq	%rdi, %rax	# tmp163, .omp_data_i
	pushq	%r14	#
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13	#
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12	#
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp	#
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx	#
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
# benchmark_2mm.c:329:                         int i_end = (i + CHUNK < ni) ? i + CHUNK : ni;
	movq	(%rdi), %rdx	# *.omp_data_i_8(D).ni, *.omp_data_i_8(D).ni
# benchmark_2mm.c:326:                     #pragma omp task depend(in:tmp[i*nj:CHUNK*nj]) \
	movl	56(%rdi), %r12d	# *.omp_data_i_8(D).i, i
# benchmark_2mm.c:329:                         int i_end = (i + CHUNK < ni) ? i + CHUNK : ni;
	movl	(%rdx), %edx	# *_12, tmp164
# benchmark_2mm.c:329:                         int i_end = (i + CHUNK < ni) ? i + CHUNK : ni;
	leal	32(%r12), %ecx	#, tmp142
# benchmark_2mm.c:329:                         int i_end = (i + CHUNK < ni) ? i + CHUNK : ni;
	cmpl	%edx, %ecx	# tmp164, tmp142
# benchmark_2mm.c:326:                     #pragma omp task depend(in:tmp[i*nj:CHUNK*nj]) \
	movl	60(%rdi), %esi	# *.omp_data_i_8(D).j, j
# benchmark_2mm.c:329:                         int i_end = (i + CHUNK < ni) ? i + CHUNK : ni;
	movl	%edx, %edi	# tmp164, tmp164
# benchmark_2mm.c:330:                         int j_end = (j + CHUNK < nl) ? j + CHUNK : nl;
	movq	16(%rax), %rdx	# *.omp_data_i_8(D).nl, *.omp_data_i_8(D).nl
# benchmark_2mm.c:329:                         int i_end = (i + CHUNK < ni) ? i + CHUNK : ni;
	cmovle	%ecx, %edi	# tmp142,, tmp164
# benchmark_2mm.c:330:                         int j_end = (j + CHUNK < nl) ? j + CHUNK : nl;
	movl	(%rdx), %edx	# *_16, _17
# benchmark_2mm.c:330:                         int j_end = (j + CHUNK < nl) ? j + CHUNK : nl;
	leal	32(%rsi), %ebx	#, tmp144
# benchmark_2mm.c:330:                         int j_end = (j + CHUNK < nl) ? j + CHUNK : nl;
	cmpl	%edx, %ebx	# _17, tmp144
# benchmark_2mm.c:329:                         int i_end = (i + CHUNK < ni) ? i + CHUNK : ni;
	movl	%edi, -28(%rsp)	# i_end, %sfp
# benchmark_2mm.c:330:                         int j_end = (j + CHUNK < nl) ? j + CHUNK : nl;
	cmovg	%edx, %ebx	# tmp144,, _17, j_end
# benchmark_2mm.c:332:                         for (int ii = i; ii < i_end; ii++) {
	cmpl	%edi, %r12d	# i_end, i
	jge	.L300	#,
# benchmark_2mm.c:334:                                 double sum = beta * D[ii*nl + jj];
	movq	40(%rax), %rcx	# *.omp_data_i_8(D).D, *.omp_data_i_8(D).D
# benchmark_2mm.c:334:                                 double sum = beta * D[ii*nl + jj];
	movq	24(%rax), %rbp	# *.omp_data_i_8(D).beta, _29
# benchmark_2mm.c:334:                                 double sum = beta * D[ii*nl + jj];
	movq	(%rcx), %r8	# *_25, _26
# benchmark_2mm.c:335:                                 for (int k = 0; k < nj; k++) {
	movq	8(%rax), %rcx	# *.omp_data_i_8(D).nj, *.omp_data_i_8(D).nj
	movl	(%rcx), %r10d	# *_58, _61
# benchmark_2mm.c:336:                                     sum += tmp[ii*nj + k] * C[k*nl + jj];
	movq	48(%rax), %rcx	# *.omp_data_i_8(D).tmp, *.omp_data_i_8(D).tmp
# benchmark_2mm.c:336:                                     sum += tmp[ii*nj + k] * C[k*nl + jj];
	movq	32(%rax), %rax	# *.omp_data_i_8(D).C, *.omp_data_i_8(D).C
# benchmark_2mm.c:336:                                     sum += tmp[ii*nj + k] * C[k*nl + jj];
	movq	(%rcx), %r14	# *_40, _41
# benchmark_2mm.c:336:                                     sum += tmp[ii*nj + k] * C[k*nl + jj];
	movq	(%rax), %rdi	# *_48, _49
	cmpl	%ebx, %esi	# j_end, j
	jge	.L300	#,
	movl	%edx, %eax	# _17, _17
	imull	%r12d, %eax	# i, _17
	movl	%r12d, %r13d	# i, ivtmp.701
	imull	%r10d, %r13d	# _61, ivtmp.701
	cltq
	leaq	(%r8,%rax,8), %r9	#, ivtmp.699
	movslq	%esi, %rax	# j, _95
	movq	%rax, -24(%rsp)	# _95, %sfp
	leaq	(%rdi,%rax,8), %rax	#, ivtmp.693
	movq	%rax, -16(%rsp)	# ivtmp.693, %sfp
	leal	-1(%r10), %eax	#, tmp156
	movq	%rax, -8(%rsp)	# tmp156, %sfp
	movslq	%edx, %rcx	# _17, _17
	salq	$3, %rcx	#, _81
	leaq	8(%r14), %r15	#, tmp162
	.p2align 4,,10
	.p2align 3
.L289:
	movslq	%r13d, %rax	# ivtmp.701, _125
	leaq	(%r14,%rax,8), %r11	#, ivtmp.683
# benchmark_2mm.c:326:                     #pragma omp task depend(in:tmp[i*nj:CHUNK*nj]) \
	movq	-16(%rsp), %r8	# %sfp, ivtmp.693
	addq	-8(%rsp), %rax	# %sfp, tmp157
	movq	-24(%rsp), %rdi	# %sfp, ivtmp.689
	leaq	(%r15,%rax,8), %rsi	#, _102
	.p2align 4,,10
	.p2align 3
.L288:
# benchmark_2mm.c:334:                                 double sum = beta * D[ii*nl + jj];
	vmovsd	(%r9,%rdi,8), %xmm0	# MEM[(double *)_90 + ivtmp.689_101 * 8], MEM[(double *)_90 + ivtmp.689_101 * 8]
# benchmark_2mm.c:335:                                 for (int k = 0; k < nj; k++) {
	movq	%r8, %rdx	# ivtmp.693, ivtmp.684
# benchmark_2mm.c:334:                                 double sum = beta * D[ii*nl + jj];
	vmulsd	0(%rbp), %xmm0, %xmm0	# *_29, MEM[(double *)_90 + ivtmp.689_101 * 8], sum
# benchmark_2mm.c:335:                                 for (int k = 0; k < nj; k++) {
	movq	%r11, %rax	# ivtmp.683, ivtmp.683
	testl	%r10d, %r10d	# _61
	jle	.L292	#,
	.p2align 4,,10
	.p2align 3
.L290:
# benchmark_2mm.c:336:                                     sum += tmp[ii*nj + k] * C[k*nl + jj];
	vmovsd	(%rax), %xmm1	# MEM[(double *)_113], tmp213
# benchmark_2mm.c:335:                                 for (int k = 0; k < nj; k++) {
	addq	$8, %rax	#, ivtmp.683
# benchmark_2mm.c:336:                                     sum += tmp[ii*nj + k] * C[k*nl + jj];
	vfmadd231sd	(%rdx), %xmm1, %xmm0	# MEM[(double *)_112], tmp213, sum
# benchmark_2mm.c:335:                                 for (int k = 0; k < nj; k++) {
	addq	%rcx, %rdx	# _81, ivtmp.684
	cmpq	%rax, %rsi	# ivtmp.683, _102
	jne	.L290	#,
.L292:
# benchmark_2mm.c:338:                                 D[ii*nl + jj] = sum;
	vmovsd	%xmm0, (%r9,%rdi,8)	# sum, MEM[(double *)_90 + ivtmp.689_101 * 8]
# benchmark_2mm.c:333:                             for (int jj = j; jj < j_end; jj++) {
	incq	%rdi	# ivtmp.689
	addq	$8, %r8	#, ivtmp.693
	cmpl	%edi, %ebx	# ivtmp.689, j_end
	jg	.L288	#,
# benchmark_2mm.c:332:                         for (int ii = i; ii < i_end; ii++) {
	incl	%r12d	# i
# benchmark_2mm.c:332:                         for (int ii = i; ii < i_end; ii++) {
	addq	%rcx, %r9	# _81, ivtmp.699
	addl	%r10d, %r13d	# _61, ivtmp.701
	cmpl	%r12d, -28(%rsp)	# i, %sfp
	jne	.L289	#,
.L300:
# benchmark_2mm.c:326:                     #pragma omp task depend(in:tmp[i*nj:CHUNK*nj]) \
	popq	%rbx	#
	.cfi_def_cfa_offset 48
	popq	%rbp	#
	.cfi_def_cfa_offset 40
	popq	%r12	#
	.cfi_def_cfa_offset 32
	popq	%r13	#
	.cfi_def_cfa_offset 24
	popq	%r14	#
	.cfi_def_cfa_offset 16
	popq	%r15	#
	.cfi_def_cfa_offset 8
	ret	
	.cfi_endproc
.LFE5563:
	.size	kernel_2mm_tasks._omp_fn.2, .-kernel_2mm_tasks._omp_fn.2
	.p2align 4
	.type	kernel_2mm_tasks._omp_fn.0, @function
kernel_2mm_tasks._omp_fn.0:
.LFB5561:
	.cfi_startproc
	endbr64	
	pushq	%r15	#
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14	#
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13	#
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12	#
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	movq	%rdi, %r12	# tmp176, .omp_data_i
	pushq	%rbp	#
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx	#
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$184, %rsp	#,
	.cfi_def_cfa_offset 240
# benchmark_2mm.c:298:     #pragma omp parallel
	movq	%fs:40, %rax	# MEM[(<address-space-1> long unsigned int *)40B], tmp178
	movq	%rax, 168(%rsp)	# tmp178, D.38542
	xorl	%eax, %eax	# tmp178
	call	GOMP_single_start@PLT	#
	testb	%al, %al	# tmp177
	je	.L305	#,
# benchmark_2mm.c:303:             for (int i = 0; i < ni; i += CHUNK) {
	movl	56(%r12), %edx	# *.omp_data_i_13(D).ni, prephitmp_122
	testl	%edx, %edx	# prephitmp_122
	jle	.L305	#,
# benchmark_2mm.c:305:                     #pragma omp task depend(out:tmp[i*nj+j:CHUNK*CHUNK])
	leaq	16(%r12), %rsi	#, tmp172
	leaq	24(%r12), %rcx	#, tmp173
	vmovq	%rsi, %xmm6	# tmp172, tmp172
# benchmark_2mm.c:303:             for (int i = 0; i < ni; i += CHUNK) {
	movl	$0, (%rsp)	#, %sfp
	vpinsrq	$1, %rcx, %xmm6, %xmm5	# tmp173, tmp172, _152
# benchmark_2mm.c:304:                 for (int j = 0; j < nj; j += CHUNK) {
	movl	60(%r12), %eax	# *.omp_data_i_13(D).nj, _54
	vmovdqa	%xmm5, 32(%rsp)	# _152, %sfp
	leaq	kernel_2mm_tasks._omp_fn.1(%rip), %rbx	#, tmp175
	.p2align 4,,10
	.p2align 3
.L313:
	testl	%eax, %eax	# _54
	jle	.L321	#,
# benchmark_2mm.c:326:                     #pragma omp task depend(in:tmp[i*nj:CHUNK*nj]) \
	leaq	56(%r12), %rdi	#, pretmp_128
	leaq	48(%r12), %rcx	#, pretmp_140
	leaq	64(%rsp), %rbp	#, tmp166
	movq	%rdi, 8(%rsp)	# pretmp_128, %sfp
	movq	%rcx, 16(%rsp)	# pretmp_140, %sfp
	leaq	128(%rsp), %rdi	#, tmp167
# benchmark_2mm.c:304:                 for (int j = 0; j < nj; j += CHUNK) {
	xorl	%r13d, %r13d	# j
	movq	%rdi, 24(%rsp)	# tmp167, %sfp
	movq	%rbp, %rdi	# tmp166, tmp166
# benchmark_2mm.c:326:                     #pragma omp task depend(in:tmp[i*nj:CHUNK*nj]) \
	leaq	60(%r12), %r15	#, pretmp_130
	movl	%r13d, %ebp	# j, j
# benchmark_2mm.c:305:                     #pragma omp task depend(out:tmp[i*nj+j:CHUNK*CHUNK])
	leaq	64(%r12), %r14	#, pretmp_132
	movq	%rdi, %r13	# tmp166, tmp166
	.p2align 4,,10
	.p2align 3
.L312:
# benchmark_2mm.c:305:                     #pragma omp task depend(out:tmp[i*nj+j:CHUNK*CHUNK])
	movl	(%rsp), %esi	# %sfp, i
# benchmark_2mm.c:305:                     #pragma omp task depend(out:tmp[i*nj+j:CHUNK*CHUNK])
	movq	48(%r12), %rdx	# *.omp_data_i_13(D).tmp, *.omp_data_i_13(D).tmp
# benchmark_2mm.c:305:                     #pragma omp task depend(out:tmp[i*nj+j:CHUNK*CHUNK])
	imull	%esi, %eax	# i, tmp154
# benchmark_2mm.c:305:                     #pragma omp task depend(out:tmp[i*nj+j:CHUNK*CHUNK])
	vmovdqa	32(%rsp), %xmm4	# %sfp, _152
	vmovd	%esi, %xmm3	# i, i
# benchmark_2mm.c:305:                     #pragma omp task depend(out:tmp[i*nj+j:CHUNK*CHUNK])
	addl	%ebp, %eax	# j, tmp155
	cltq
# benchmark_2mm.c:305:                     #pragma omp task depend(out:tmp[i*nj+j:CHUNK*CHUNK])
	leaq	(%rdx,%rax,8), %rax	#, _61
	movq	%rax, 144(%rsp)	# _61, MEM[(void *[3] *)_32][2]
# benchmark_2mm.c:305:                     #pragma omp task depend(out:tmp[i*nj+j:CHUNK*CHUNK])
	movq	8(%rsp), %rax	# %sfp, pretmp_128
	vpinsrd	$1, %ebp, %xmm3, %xmm0	# j, i, tmp159
	movq	%rax, 64(%rsp)	# pretmp_128, MEM[(struct .omp_data_s.87 *)_102].ni
	movq	16(%rsp), %rax	# %sfp, pretmp_140
	movq	$1, 128(%rsp)	#, MEM[(void *[3] *)_32][0]
	movq	%rax, 112(%rsp)	# pretmp_140, MEM[(struct .omp_data_s.87 *)_102].tmp
	movq	$1, 136(%rsp)	#, MEM[(void *[3] *)_32][1]
	movq	%r15, 72(%rsp)	# pretmp_130, MEM[(struct .omp_data_s.87 *)_102].nj
	movq	%r14, 80(%rsp)	# pretmp_132, MEM[(struct .omp_data_s.87 *)_102].nk
	movq	%r12, 88(%rsp)	# .omp_data_i, MEM[(struct .omp_data_s.87 *)_102].alpha
	vmovq	%xmm0, 120(%rsp)	# tmp159, MEM <vector(2) int> [(int *)_102]
	vmovdqa	%xmm4, 96(%rsp)	# _152, MEM <vector(2) long unsigned int> [(double * * *)_102]
	pushq	$0	#
	.cfi_def_cfa_offset 248
	movl	$1, %r9d	#,
	movl	$8, %r8d	#,
	pushq	$0	#
	.cfi_def_cfa_offset 256
	movl	$64, %ecx	#,
	xorl	%edx, %edx	#
	pushq	40(%rsp)	# %sfp
	.cfi_def_cfa_offset 264
	movq	%r13, %rsi	# tmp166,
	movq	%rbx, %rdi	# tmp175,
	pushq	$8	#
	.cfi_def_cfa_offset 272
# benchmark_2mm.c:304:                 for (int j = 0; j < nj; j += CHUNK) {
	addl	$32, %ebp	#, j
	call	GOMP_task@PLT	#
# benchmark_2mm.c:304:                 for (int j = 0; j < nj; j += CHUNK) {
	movl	60(%r12), %eax	# *.omp_data_i_13(D).nj, _54
	addq	$32, %rsp	#,
	.cfi_def_cfa_offset 240
	cmpl	%ebp, %eax	# j, _54
	jg	.L312	#,
# benchmark_2mm.c:303:             for (int i = 0; i < ni; i += CHUNK) {
	addl	$32, (%rsp)	#, %sfp
# benchmark_2mm.c:303:             for (int i = 0; i < ni; i += CHUNK) {
	movl	56(%r12), %edx	# *.omp_data_i_13(D).ni, prephitmp_122
# benchmark_2mm.c:303:             for (int i = 0; i < ni; i += CHUNK) {
	movl	(%rsp), %ecx	# %sfp, i
# benchmark_2mm.c:303:             for (int i = 0; i < ni; i += CHUNK) {
	cmpl	%edx, %ecx	# prephitmp_122, i
	jl	.L313	#,
.L311:
# benchmark_2mm.c:324:             for (int i = 0; i < ni; i += CHUNK) {
	testl	%edx, %edx	# prephitmp_122
	jg	.L310	#,
	.p2align 4,,10
	.p2align 3
.L305:
	movq	168(%rsp), %rax	# D.38542, tmp179
	subq	%fs:40, %rax	# MEM[(<address-space-1> long unsigned int *)40B], tmp179
	jne	.L322	#,
# benchmark_2mm.c:298:     #pragma omp parallel
	addq	$184, %rsp	#,
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx	#
	.cfi_def_cfa_offset 48
	popq	%rbp	#
	.cfi_def_cfa_offset 40
	popq	%r12	#
	.cfi_def_cfa_offset 32
	popq	%r13	#
	.cfi_def_cfa_offset 24
	popq	%r14	#
	.cfi_def_cfa_offset 16
	popq	%r15	#
	.cfi_def_cfa_offset 8
	jmp	GOMP_barrier@PLT	#
	.p2align 4,,10
	.p2align 3
.L321:
	.cfi_restore_state
# benchmark_2mm.c:303:             for (int i = 0; i < ni; i += CHUNK) {
	movl	(%rsp), %eax	# %sfp, i
	addl	$32, %eax	#, i
# benchmark_2mm.c:303:             for (int i = 0; i < ni; i += CHUNK) {
	cmpl	%edx, %eax	# prephitmp_122, i
	jge	.L311	#,
.L310:
# benchmark_2mm.c:326:                     #pragma omp task depend(in:tmp[i*nj:CHUNK*nj]) \
	leaq	32(%r12), %rcx	#, tmp168
	leaq	40(%r12), %rdx	#, tmp169
	vmovq	%rcx, %xmm5	# tmp168, tmp168
	vpinsrq	$1, %rdx, %xmm5, %xmm7	# tmp169, tmp168, _197
# benchmark_2mm.c:325:                 for (int j = 0; j < nl; j += CHUNK) {
	movl	68(%r12), %eax	# *.omp_data_i_13(D).nl, _16
# benchmark_2mm.c:324:             for (int i = 0; i < ni; i += CHUNK) {
	xorl	%r13d, %r13d	# i
	vmovdqa	%xmm7, 48(%rsp)	# _197, %sfp
	leaq	kernel_2mm_tasks._omp_fn.2(%rip), %rbp	#, tmp171
	.p2align 4,,10
	.p2align 3
.L309:
# benchmark_2mm.c:325:                 for (int j = 0; j < nl; j += CHUNK) {
	testl	%eax, %eax	# _16
	jle	.L305	#,
# benchmark_2mm.c:326:                     #pragma omp task depend(in:tmp[i*nj:CHUNK*nj]) \
	leaq	68(%r12), %rdi	#, pretmp_157
	movq	%rdi, (%rsp)	# pretmp_157, %sfp
	leaq	56(%r12), %rdi	#, pretmp_165
	movq	%rdi, 8(%rsp)	# pretmp_165, %sfp
	leaq	60(%r12), %rdi	#, pretmp_167
	movq	%rdi, 16(%rsp)	# pretmp_167, %sfp
	leaq	48(%r12), %rdi	#, pretmp_169
	leaq	64(%rsp), %rbx	#, tmp166
	leaq	128(%rsp), %rcx	#, tmp167
	movq	%rdi, 24(%rsp)	# pretmp_169, %sfp
# benchmark_2mm.c:325:                 for (int j = 0; j < nl; j += CHUNK) {
	xorl	%r14d, %r14d	# j
	movq	%rcx, 32(%rsp)	# tmp167, %sfp
	movq	%rbx, %rcx	# tmp166, tmp166
# benchmark_2mm.c:326:                     #pragma omp task depend(in:tmp[i*nj:CHUNK*nj]) \
	leaq	8(%r12), %r15	#, pretmp_159
	movl	%r14d, %ebx	# j, j
	movq	%rcx, %r14	# tmp166, tmp166
	.p2align 4,,10
	.p2align 3
.L308:
# benchmark_2mm.c:327:                                      depend(inout:D[i*nl+j:CHUNK*CHUNK])
	imull	%r13d, %eax	# i, tmp138
# benchmark_2mm.c:326:                     #pragma omp task depend(in:tmp[i*nj:CHUNK*nj]) \
	vmovdqa	48(%rsp), %xmm2	# %sfp, _197
	vmovd	%r13d, %xmm1	# i, i
# benchmark_2mm.c:327:                                      depend(inout:D[i*nl+j:CHUNK*CHUNK])
	addl	%ebx, %eax	# j, tmp139
	cltq
# benchmark_2mm.c:327:                                      depend(inout:D[i*nl+j:CHUNK*CHUNK])
	salq	$3, %rax	#, tmp141
	vmovq	%rax, %xmm0	# tmp141, tmp141
# benchmark_2mm.c:326:                     #pragma omp task depend(in:tmp[i*nj:CHUNK*nj]) \
	movl	60(%r12), %eax	# *.omp_data_i_13(D).nj, tmp142
	movq	$2, 128(%rsp)	#, MEM[(void *[4] *)_32][0]
	imull	%r13d, %eax	# i, tmp142
	movq	$1, 136(%rsp)	#, MEM[(void *[4] *)_32][1]
# benchmark_2mm.c:326:                     #pragma omp task depend(in:tmp[i*nj:CHUNK*nj]) \
	movq	%r15, 88(%rsp)	# pretmp_159, MEM[(struct .omp_data_s.88 *)_102].beta
# benchmark_2mm.c:326:                     #pragma omp task depend(in:tmp[i*nj:CHUNK*nj]) \
	cltq
# benchmark_2mm.c:326:                     #pragma omp task depend(in:tmp[i*nj:CHUNK*nj]) \
	salq	$3, %rax	#, tmp144
# benchmark_2mm.c:327:                                      depend(inout:D[i*nl+j:CHUNK*CHUNK])
	vpinsrq	$1, %rax, %xmm0, %xmm0	# tmp144, tmp141, tmp137
# benchmark_2mm.c:326:                     #pragma omp task depend(in:tmp[i*nj:CHUNK*nj]) \
	movq	8(%rsp), %rax	# %sfp, pretmp_165
# benchmark_2mm.c:327:                                      depend(inout:D[i*nl+j:CHUNK*CHUNK])
	vpaddq	40(%r12), %xmm0, %xmm0	# MEM <vector(2) long unsigned int> [(double * *).omp_data_i_13(D) + 40B], tmp137, vect__23.719
# benchmark_2mm.c:326:                     #pragma omp task depend(in:tmp[i*nj:CHUNK*nj]) \
	movq	%rax, 64(%rsp)	# pretmp_165, MEM[(struct .omp_data_s.88 *)_102].ni
	movq	16(%rsp), %rax	# %sfp, pretmp_167
	vmovdqa	%xmm0, 144(%rsp)	# vect__23.719, MEM <vector(2) long unsigned int> [(void * *)_32]
	movq	%rax, 72(%rsp)	# pretmp_167, MEM[(struct .omp_data_s.88 *)_102].nj
	movq	(%rsp), %rax	# %sfp, pretmp_157
	vpinsrd	$1, %ebx, %xmm1, %xmm0	# j, i, tmp146
	movq	%rax, 80(%rsp)	# pretmp_157, MEM[(struct .omp_data_s.88 *)_102].nl
	movq	24(%rsp), %rax	# %sfp, pretmp_169
	vmovq	%xmm0, 120(%rsp)	# tmp146, MEM <vector(2) int> [(int *)_102]
	movq	%rax, 112(%rsp)	# pretmp_169, MEM[(struct .omp_data_s.88 *)_102].tmp
	vmovdqa	%xmm2, 96(%rsp)	# _197, MEM <vector(2) long unsigned int> [(double * * *)_102]
	pushq	$0	#
	.cfi_def_cfa_offset 248
	movl	$1, %r9d	#,
	movl	$8, %r8d	#,
	pushq	$0	#
	.cfi_def_cfa_offset 256
	movl	$64, %ecx	#,
	xorl	%edx, %edx	#
	pushq	48(%rsp)	# %sfp
	.cfi_def_cfa_offset 264
	movq	%r14, %rsi	# tmp166,
	movq	%rbp, %rdi	# tmp171,
	pushq	$8	#
	.cfi_def_cfa_offset 272
# benchmark_2mm.c:325:                 for (int j = 0; j < nl; j += CHUNK) {
	addl	$32, %ebx	#, j
	call	GOMP_task@PLT	#
# benchmark_2mm.c:325:                 for (int j = 0; j < nl; j += CHUNK) {
	movl	68(%r12), %eax	# *.omp_data_i_13(D).nl, _16
	addq	$32, %rsp	#,
	.cfi_def_cfa_offset 240
	cmpl	%ebx, %eax	# j, _16
	jg	.L308	#,
# benchmark_2mm.c:324:             for (int i = 0; i < ni; i += CHUNK) {
	addl	$32, %r13d	#, i
# benchmark_2mm.c:324:             for (int i = 0; i < ni; i += CHUNK) {
	cmpl	%r13d, 56(%r12)	# i, *.omp_data_i_13(D).ni
	jg	.L309	#,
	jmp	.L305	#
.L322:
	call	__stack_chk_fail@PLT	#
	.cfi_endproc
.LFE5561:
	.size	kernel_2mm_tasks._omp_fn.0, .-kernel_2mm_tasks._omp_fn.0
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC1:
	.string	"Error: aligned allocation failed\n"
	.text
	.p2align 4
	.type	aligned_malloc, @function
aligned_malloc:
.LFB5541:
	.cfi_startproc
	subq	$24, %rsp	#,
	.cfi_def_cfa_offset 32
# benchmark_2mm.c:58: static void* aligned_malloc(size_t size) {
	movq	%rdi, %rdx	# tmp93, size
# benchmark_2mm.c:60:     if (posix_memalign(&ptr, ALIGN_SIZE, size) != 0) {
	movl	$64, %esi	#,
	movq	%rsp, %rdi	#, tmp87
# benchmark_2mm.c:58: static void* aligned_malloc(size_t size) {
	movq	%fs:40, %rax	# MEM[(<address-space-1> long unsigned int *)40B], tmp95
	movq	%rax, 8(%rsp)	# tmp95, D.38552
	xorl	%eax, %eax	# tmp95
# benchmark_2mm.c:60:     if (posix_memalign(&ptr, ALIGN_SIZE, size) != 0) {
	call	posix_memalign@PLT	#
	testl	%eax, %eax	# tmp94
	jne	.L324	#,
# benchmark_2mm.c:64:     return ptr;
	movq	(%rsp), %rax	# D.37659, D.37659
# benchmark_2mm.c:65: }
	movq	8(%rsp), %rdx	# D.38552, tmp96
	subq	%fs:40, %rdx	# MEM[(<address-space-1> long unsigned int *)40B], tmp96
	jne	.L328	#,
	addq	$24, %rsp	#,
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret	
.L324:
	.cfi_restore_state
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:105:   return __fprintf_chk (__stream, __USE_FORTIFY_LEVEL - 1, __fmt,
	movq	stderr(%rip), %rcx	# stderr,
	leaq	.LC1(%rip), %rdi	#, tmp91
	movl	$33, %edx	#,
	movl	$1, %esi	#,
	call	fwrite@PLT	#
# benchmark_2mm.c:62:         exit(1);
	movl	$1, %edi	#,
	call	exit@PLT	#
.L328:
# benchmark_2mm.c:65: }
	call	__stack_chk_fail@PLT	#
	.cfi_endproc
.LFE5541:
	.size	aligned_malloc, .-aligned_malloc
	.p2align 4
	.globl	kernel_2mm_sequential
	.type	kernel_2mm_sequential, @function
kernel_2mm_sequential:
.LFB5545:
	.cfi_startproc
	endbr64	
	pushq	%r15	#
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14	#
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13	#
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12	#
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp	#
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx	#
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
# benchmark_2mm.c:111:                           double *tmp) {
	movl	%edi, -8(%rsp)	# ni, %sfp
	movq	%r8, -32(%rsp)	# tmp184, %sfp
	movq	%r9, -24(%rsp)	# tmp185, %sfp
# benchmark_2mm.c:113:     for (int i = 0; i < ni; i++) {
	testl	%edi, %edi	# ni
	jle	.L358	#,
	movl	%esi, %r10d	# tmp179, nj
	movl	%ecx, %ebx	# tmp181, nl
	testl	%esi, %esi	# nj
	jle	.L339	#,
	leal	-1(%rdx), %eax	#, tmp162
	movq	%rax, -16(%rsp)	# tmp162, %sfp
	movl	%ecx, -4(%rsp)	# nl, %sfp
	movslq	%esi, %r15	# nj, _196
	movl	%edx, %r11d	# tmp180, nk
	leaq	0(,%r15,8), %rdi	#, _118
	xorl	%r14d, %r14d	# ivtmp.798
	xorl	%r13d, %r13d	# ivtmp.797
# benchmark_2mm.c:113:     for (int i = 0; i < ni; i++) {
	xorl	%r12d, %r12d	# i
	vxorpd	%xmm4, %xmm4, %xmm4	# tmp177
	leaq	8(%r8), %rbp	#, tmp175
	.p2align 4,,10
	.p2align 3
.L336:
	movq	72(%rsp), %rax	# tmp, tmp278
	movq	-32(%rsp), %rbx	# %sfp, A
	leaq	(%rax,%r14,8), %rcx	#, ivtmp.788
	movslq	%r13d, %rax	# ivtmp.797, _124
	leaq	(%rbx,%rax,8), %rbx	#, ivtmp.782
	movq	-24(%rsp), %r9	# %sfp, ivtmp.790
	addq	-16(%rsp), %rax	# %sfp, tmp163
	leaq	0(%rbp,%rax,8), %rsi	#, _84
# benchmark_2mm.c:114:         for (int j = 0; j < nj; j++) {
	xorl	%r8d, %r8d	# j
	.p2align 4,,10
	.p2align 3
.L333:
# benchmark_2mm.c:115:             tmp[i*nj + j] = 0.0;
	movq	$0x000000000, (%rcx)	#, MEM[(double *)_48]
# benchmark_2mm.c:116:             for (int k = 0; k < nk; k++) {
	movq	%r9, %rdx	# ivtmp.790, ivtmp.783
	movq	%rbx, %rax	# ivtmp.782, ivtmp.782
	vmovsd	%xmm4, %xmm4, %xmm2	# tmp177, _21
	testl	%r11d, %r11d	# nk
	jle	.L335	#,
	.p2align 4,,10
	.p2align 3
.L332:
# benchmark_2mm.c:117:                 tmp[i*nj + j] += alpha * A[i*nk + k] * B[k*nj + j];
	vmulsd	(%rax), %xmm0, %xmm3	# MEM[(double *)_113], alpha, tmp158
# benchmark_2mm.c:116:             for (int k = 0; k < nk; k++) {
	addq	$8, %rax	#, ivtmp.782
# benchmark_2mm.c:117:                 tmp[i*nj + j] += alpha * A[i*nk + k] * B[k*nj + j];
	vfmadd231sd	(%rdx), %xmm3, %xmm2	# MEM[(double *)_112], tmp158, _21
# benchmark_2mm.c:116:             for (int k = 0; k < nk; k++) {
	addq	%rdi, %rdx	# _118, ivtmp.783
# benchmark_2mm.c:117:                 tmp[i*nj + j] += alpha * A[i*nk + k] * B[k*nj + j];
	vmovsd	%xmm2, (%rcx)	# _21, MEM[(double *)_48]
# benchmark_2mm.c:116:             for (int k = 0; k < nk; k++) {
	cmpq	%rax, %rsi	# ivtmp.782, _84
	jne	.L332	#,
.L335:
# benchmark_2mm.c:114:         for (int j = 0; j < nj; j++) {
	incl	%r8d	# j
# benchmark_2mm.c:114:         for (int j = 0; j < nj; j++) {
	addq	$8, %rcx	#, ivtmp.788
	addq	$8, %r9	#, ivtmp.790
	cmpl	%r8d, %r10d	# j, nj
	jne	.L333	#,
# benchmark_2mm.c:113:     for (int i = 0; i < ni; i++) {
	incl	%r12d	# i
# benchmark_2mm.c:113:     for (int i = 0; i < ni; i++) {
	addl	%r11d, %r13d	# nk, ivtmp.797
	addq	%r15, %r14	# _196, ivtmp.798
	cmpl	%r12d, -8(%rsp)	# i, %sfp
	jne	.L336	#,
	movl	-4(%rsp), %ebx	# %sfp, nl
.L339:
	testl	%ebx, %ebx	# nl
	jle	.L358	#,
	movq	72(%rsp), %rax	# tmp, tmp273
	movslq	%ebx, %r14	# nl, _132
	leaq	8(%rax), %r15	#, tmp176
	leal	-1(%r10), %eax	#, tmp169
	movq	%rax, -32(%rsp)	# tmp169, %sfp
	leaq	0(,%r14,8), %rdi	#, _170
# benchmark_2mm.c:126:             for (int k = 0; k < nj; k++) {
	xorl	%r13d, %r13d	# ivtmp.778
	xorl	%r12d, %r12d	# ivtmp.777
	xorl	%ebp, %ebp	# i
	.p2align 4,,10
	.p2align 3
.L338:
	movq	64(%rsp), %rax	# D, tmp282
	movq	72(%rsp), %rsi	# tmp, tmp283
	leaq	(%rax,%r13,8), %rcx	#, ivtmp.768
	movslq	%r12d, %rax	# ivtmp.777, _176
	leaq	(%rsi,%rax,8), %r11	#, ivtmp.762
	movq	56(%rsp), %r9	# C, ivtmp.770
	addq	-32(%rsp), %rax	# %sfp, tmp170
	leaq	(%r15,%rax,8), %rsi	#, _153
# benchmark_2mm.c:124:         for (int j = 0; j < nl; j++) {
	xorl	%r8d, %r8d	# j
	.p2align 4,,10
	.p2align 3
.L341:
# benchmark_2mm.c:125:             D[i*nl + j] *= beta;
	vmulsd	(%rcx), %xmm1, %xmm0	# MEM[(double *)_143], beta, _43
# benchmark_2mm.c:126:             for (int k = 0; k < nj; k++) {
	movq	%r9, %rdx	# ivtmp.770, ivtmp.763
	movq	%r11, %rax	# ivtmp.762, ivtmp.762
# benchmark_2mm.c:125:             D[i*nl + j] *= beta;
	vmovsd	%xmm0, (%rcx)	# _43, MEM[(double *)_143]
# benchmark_2mm.c:126:             for (int k = 0; k < nj; k++) {
	testl	%r10d, %r10d	# nj
	jle	.L343	#,
	.p2align 4,,10
	.p2align 3
.L340:
# benchmark_2mm.c:127:                 D[i*nl + j] += tmp[i*nj + k] * C[k*nl + j];
	vmovsd	(%rax), %xmm5	# MEM[(double *)_165], tmp280
# benchmark_2mm.c:126:             for (int k = 0; k < nj; k++) {
	addq	$8, %rax	#, ivtmp.762
# benchmark_2mm.c:127:                 D[i*nl + j] += tmp[i*nj + k] * C[k*nl + j];
	vfmadd231sd	(%rdx), %xmm5, %xmm0	# MEM[(double *)_164], tmp280, _43
# benchmark_2mm.c:126:             for (int k = 0; k < nj; k++) {
	addq	%rdi, %rdx	# _170, ivtmp.763
# benchmark_2mm.c:127:                 D[i*nl + j] += tmp[i*nj + k] * C[k*nl + j];
	vmovsd	%xmm0, (%rcx)	# _43, MEM[(double *)_143]
# benchmark_2mm.c:126:             for (int k = 0; k < nj; k++) {
	cmpq	%rax, %rsi	# ivtmp.762, _153
	jne	.L340	#,
.L343:
# benchmark_2mm.c:124:         for (int j = 0; j < nl; j++) {
	incl	%r8d	# j
# benchmark_2mm.c:124:         for (int j = 0; j < nl; j++) {
	addq	$8, %rcx	#, ivtmp.768
	addq	$8, %r9	#, ivtmp.770
	cmpl	%r8d, %ebx	# j, nl
	jne	.L341	#,
# benchmark_2mm.c:123:     for (int i = 0; i < ni; i++) {
	incl	%ebp	# i
# benchmark_2mm.c:123:     for (int i = 0; i < ni; i++) {
	addl	%r10d, %r12d	# nj, ivtmp.777
	addq	%r14, %r13	# _132, ivtmp.778
	cmpl	%ebp, -8(%rsp)	# i, %sfp
	jne	.L338	#,
.L358:
# benchmark_2mm.c:131: }
	popq	%rbx	#
	.cfi_def_cfa_offset 48
	popq	%rbp	#
	.cfi_def_cfa_offset 40
	popq	%r12	#
	.cfi_def_cfa_offset 32
	popq	%r13	#
	.cfi_def_cfa_offset 24
	popq	%r14	#
	.cfi_def_cfa_offset 16
	popq	%r15	#
	.cfi_def_cfa_offset 8
	ret	
	.cfi_endproc
.LFE5545:
	.size	kernel_2mm_sequential, .-kernel_2mm_sequential
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC54:
	.string	"Warming up CPU..."
	.section	.rodata.str1.8
	.align 8
.LC57:
	.string	"\n=== Running 2MM Benchmark ==="
	.align 8
.LC58:
	.string	"Problem size: NI=%d, NJ=%d, NK=%d, NL=%d\n"
	.section	.rodata.str1.1
.LC59:
	.string	"Total FLOPS: %lld\n"
.LC61:
	.string	"Memory footprint: %.2f MB\n\n"
	.section	.rodata.str1.8
	.align 8
.LC66:
	.string	"Sequential: %.4f seconds (%.2f GFLOPS)\n"
	.section	.rodata.str1.1
.LC68:
	.string	"Speedup"
.LC69:
	.string	"Time (s)"
.LC70:
	.string	"Threads"
.LC71:
	.string	"Strategy"
	.section	.rodata.str1.8
	.align 8
.LC72:
	.string	"\n%-20s %-10s %-12s %-12s %-12s %-10s\n"
	.section	.rodata.str1.1
.LC73:
	.string	"GFLOPS"
.LC74:
	.string	"Efficiency"
.LC75:
	.string	"-------"
.LC76:
	.string	"--------"
	.section	.rodata.str1.8
	.align 8
.LC77:
	.string	"%-20s %-10s %-12s %-12s %-12s %-10s\n"
	.section	.rodata.str1.1
.LC78:
	.string	"------"
.LC79:
	.string	"----------"
.LC80:
	.string	"Basic Parallel"
.LC81:
	.string	"Collapsed"
.LC82:
	.string	"Tiled"
.LC83:
	.string	"SIMD"
.LC84:
	.string	"Task-based"
	.section	.rodata.str1.8
	.align 8
.LC88:
	.string	"%-20s %-10d %-12.4f %-12.2f %-12.1f%% %-10.2f"
	.section	.rodata.str1.1
.LC90:
	.string	" [ERROR: %.2e]"
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB5551:
	.cfi_startproc
	endbr64	
	leaq	8(%rsp), %r10	#,
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp	#,
	pushq	-8(%r10)	#
	pushq	%rbp	#
	movq	%rsp, %rbp	#,
	.cfi_escape 0x10,0x6,0x2,0x76,0
	pushq	%r15	#
	pushq	%r14	#
	pushq	%r13	#
	pushq	%r12	#
	pushq	%r10	#
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	pushq	%rbx	#
	leaq	-16384(%rsp), %r11	#,
.LPSRL0:
	subq	$4096, %rsp	#,
	orq	$0, (%rsp)	#,
	cmpq	%r11, %rsp	#,
	jne	.LPSRL0
	subq	$352, %rsp	#,
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
# benchmark_2mm.c:351:     double *A = (double*)aligned_malloc(NI * NK * sizeof(double));
	movl	$112000, %edi	#,
# benchmark_2mm.c:349: int main(int argc, char** argv) {
	movq	%fs:40, %rax	# MEM[(<address-space-1> long unsigned int *)40B], tmp1810
	movq	%rax, -56(%rbp)	# tmp1810, D.38804
	xorl	%eax, %eax	# tmp1810
# benchmark_2mm.c:351:     double *A = (double*)aligned_malloc(NI * NK * sizeof(double));
	call	aligned_malloc	#
# benchmark_2mm.c:352:     double *B = (double*)aligned_malloc(NK * NJ * sizeof(double));
	movl	$134400, %edi	#,
# benchmark_2mm.c:351:     double *A = (double*)aligned_malloc(NI * NK * sizeof(double));
	movq	%rax, %r14	# tmp1795, A
# benchmark_2mm.c:352:     double *B = (double*)aligned_malloc(NK * NJ * sizeof(double));
	call	aligned_malloc	#
# benchmark_2mm.c:353:     double *C = (double*)aligned_malloc(NJ * NL * sizeof(double));
	movl	$153600, %edi	#,
# benchmark_2mm.c:352:     double *B = (double*)aligned_malloc(NK * NJ * sizeof(double));
	movq	%rax, %r13	# tmp1796, B
# benchmark_2mm.c:353:     double *C = (double*)aligned_malloc(NJ * NL * sizeof(double));
	call	aligned_malloc	#
# benchmark_2mm.c:354:     double *D_ref = (double*)aligned_malloc(NI * NL * sizeof(double));
	movl	$128000, %edi	#,
# benchmark_2mm.c:353:     double *C = (double*)aligned_malloc(NJ * NL * sizeof(double));
	movq	%rax, -16696(%rbp)	# tmp1797, %sfp
# benchmark_2mm.c:354:     double *D_ref = (double*)aligned_malloc(NI * NL * sizeof(double));
	call	aligned_malloc	#
# benchmark_2mm.c:355:     double *D = (double*)aligned_malloc(NI * NL * sizeof(double));
	movl	$128000, %edi	#,
# benchmark_2mm.c:354:     double *D_ref = (double*)aligned_malloc(NI * NL * sizeof(double));
	movq	%rax, %r15	# tmp1798, D_ref
# benchmark_2mm.c:355:     double *D = (double*)aligned_malloc(NI * NL * sizeof(double));
	call	aligned_malloc	#
# benchmark_2mm.c:356:     double *tmp = (double*)aligned_malloc(NI * NJ * sizeof(double));
	movl	$96000, %edi	#,
# benchmark_2mm.c:355:     double *D = (double*)aligned_malloc(NI * NL * sizeof(double));
	movq	%rax, %r12	# tmp1799, D
# benchmark_2mm.c:356:     double *tmp = (double*)aligned_malloc(NI * NJ * sizeof(double));
	call	aligned_malloc	#
	vmovdqa	.LC2(%rip), %ymm11	#, vect_vec_iv_.805
	vmovdqa	.LC3(%rip), %ymm5	#, tmp1778
	vmovdqa	.LC4(%rip), %ymm1	#, tmp1779
	vmovdqa	.LC5(%rip), %ymm2	#, tmp1781
	vmovdqa	.LC6(%rip), %ymm3	#, tmp1782
	vmovapd	.LC7(%rip), %ymm0	#, tmp1783
	vmovdqa	.LC8(%rip), %ymm15	#, tmp1784
	vmovdqa	.LC9(%rip), %ymm14	#, tmp1785
	vmovdqa	.LC10(%rip), %ymm13	#, tmp1762
	vmovdqa	.LC11(%rip), %ymm12	#, tmp1763
	vmovdqa	.LC26(%rip), %xmm10	#, tmp1731
	vmovapd	.LC29(%rip), %xmm9	#, tmp1734
	movq	%rax, -16704(%rbp)	# tmp1800, %sfp
# benchmark_2mm.c:70:     for (int i = 0; i < ni; i++)
	xorl	%edx, %edx	# i
	movq	%r14, %rax	# A, ivtmp.962
	.p2align 4,,10
	.p2align 3
.L361:
	vmovd	%edx, %xmm4	# i, tmp530
	vpbroadcastd	%xmm4, %ymm4	# tmp530, tmp530
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpmulld	%ymm11, %ymm4, %ymm6	# vect_vec_iv_.805, tmp530, vect__131.844
# benchmark_2mm.c:70:     for (int i = 0; i < ni; i++)
	addq	$1120, %rax	#, ivtmp.962
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpaddd	%ymm5, %ymm6, %ymm6	# tmp1778, vect__131.844, vect__132.845
	vpsrlq	$32, %ymm6, %ymm7	#, vect__132.845, tmp538
	vpmuldq	%ymm1, %ymm6, %ymm8	# tmp1779, vect__132.845, tmp534
	vpmuldq	%ymm1, %ymm7, %ymm7	# tmp1779, tmp538, tmp536
	vpshufb	%ymm2, %ymm8, %ymm8	# tmp1781, tmp534, tmp547
	vpshufb	%ymm3, %ymm7, %ymm7	# tmp1782, tmp536, tmp549
	vpor	%ymm7, %ymm8, %ymm8	# tmp549, tmp547, tmp542
	vpsrad	$5, %ymm8, %ymm8	#, tmp542, vect_patt_429.847
	vpslld	$1, %ymm8, %ymm7	#, vect_patt_429.847, tmp552
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp552, tmp553
	vpslld	$3, %ymm7, %ymm7	#, tmp553, tmp554
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp554, vect_patt_430.848
	vpslld	$2, %ymm7, %ymm7	#, vect_patt_430.848, tmp556
	vpsubd	%ymm7, %ymm6, %ymm6	# tmp556, vect__132.845, vect_patt_431.849
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vcvtdq2pd	%xmm6, %ymm7	# vect_patt_431.849, vect__134.850
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_431.849, tmp561
	vcvtdq2pd	%xmm6, %ymm6	# tmp561, vect__134.850
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1783, vect__134.850, vect__141.851
	vdivpd	%ymm0, %ymm7, %ymm7	# tmp1783, vect__134.850, vect__141.851
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm6, -1088(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 32B]
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpmulld	%ymm15, %ymm4, %ymm6	# tmp1784, tmp530, vect__131.844
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpaddd	%ymm5, %ymm6, %ymm6	# tmp1778, vect__131.844, vect__132.845
	vpmuldq	%ymm1, %ymm6, %ymm8	# tmp1779, vect__132.845, tmp567
	vpshufb	%ymm2, %ymm8, %ymm8	# tmp1781, tmp567, tmp580
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm7, -1120(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826]
	vpsrlq	$32, %ymm6, %ymm7	#, vect__132.845, tmp571
	vpmuldq	%ymm1, %ymm7, %ymm7	# tmp1779, tmp571, tmp569
	vpshufb	%ymm3, %ymm7, %ymm7	# tmp1782, tmp569, tmp582
	vpor	%ymm7, %ymm8, %ymm8	# tmp582, tmp580, tmp575
	vpsrad	$5, %ymm8, %ymm8	#, tmp575, vect_patt_429.847
	vpslld	$1, %ymm8, %ymm7	#, vect_patt_429.847, tmp585
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp585, tmp586
	vpslld	$3, %ymm7, %ymm7	#, tmp586, tmp587
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp587, vect_patt_430.848
	vpslld	$2, %ymm7, %ymm7	#, vect_patt_430.848, tmp589
	vpsubd	%ymm7, %ymm6, %ymm6	# tmp589, vect__132.845, vect_patt_431.849
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vcvtdq2pd	%xmm6, %ymm7	# vect_patt_431.849, vect__134.850
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_431.849, tmp594
	vcvtdq2pd	%xmm6, %ymm6	# tmp594, vect__134.850
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1783, vect__134.850, vect__141.851
	vdivpd	%ymm0, %ymm7, %ymm7	# tmp1783, vect__134.850, vect__141.851
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm6, -1024(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 96B]
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpmulld	%ymm14, %ymm4, %ymm6	# tmp1785, tmp530, vect__131.844
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpaddd	%ymm5, %ymm6, %ymm6	# tmp1778, vect__131.844, vect__132.845
	vpmuldq	%ymm1, %ymm6, %ymm8	# tmp1779, vect__132.845, tmp600
	vpshufb	%ymm2, %ymm8, %ymm8	# tmp1781, tmp600, tmp613
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm7, -1056(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 64B]
	vpsrlq	$32, %ymm6, %ymm7	#, vect__132.845, tmp604
	vpmuldq	%ymm1, %ymm7, %ymm7	# tmp1779, tmp604, tmp602
	vpshufb	%ymm3, %ymm7, %ymm7	# tmp1782, tmp602, tmp615
	vpor	%ymm7, %ymm8, %ymm8	# tmp615, tmp613, tmp608
	vpsrad	$5, %ymm8, %ymm8	#, tmp608, vect_patt_429.847
	vpslld	$1, %ymm8, %ymm7	#, vect_patt_429.847, tmp618
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp618, tmp619
	vpslld	$3, %ymm7, %ymm7	#, tmp619, tmp620
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp620, vect_patt_430.848
	vpslld	$2, %ymm7, %ymm7	#, vect_patt_430.848, tmp622
	vpsubd	%ymm7, %ymm6, %ymm6	# tmp622, vect__132.845, vect_patt_431.849
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vcvtdq2pd	%xmm6, %ymm7	# vect_patt_431.849, vect__134.850
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_431.849, tmp627
	vcvtdq2pd	%xmm6, %ymm6	# tmp627, vect__134.850
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1783, vect__134.850, vect__141.851
	vdivpd	%ymm0, %ymm7, %ymm7	# tmp1783, vect__134.850, vect__141.851
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm6, -960(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 160B]
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpmulld	%ymm13, %ymm4, %ymm6	# tmp1762, tmp530, vect__131.844
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpaddd	%ymm5, %ymm6, %ymm6	# tmp1778, vect__131.844, vect__132.845
	vpmuldq	%ymm1, %ymm6, %ymm8	# tmp1779, vect__132.845, tmp633
	vpshufb	%ymm2, %ymm8, %ymm8	# tmp1781, tmp633, tmp646
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm7, -992(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 128B]
	vpsrlq	$32, %ymm6, %ymm7	#, vect__132.845, tmp637
	vpmuldq	%ymm1, %ymm7, %ymm7	# tmp1779, tmp637, tmp635
	vpshufb	%ymm3, %ymm7, %ymm7	# tmp1782, tmp635, tmp648
	vpor	%ymm7, %ymm8, %ymm8	# tmp648, tmp646, tmp641
	vpsrad	$5, %ymm8, %ymm8	#, tmp641, vect_patt_429.847
	vpslld	$1, %ymm8, %ymm7	#, vect_patt_429.847, tmp651
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp651, tmp652
	vpslld	$3, %ymm7, %ymm7	#, tmp652, tmp653
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp653, vect_patt_430.848
	vpslld	$2, %ymm7, %ymm7	#, vect_patt_430.848, tmp655
	vpsubd	%ymm7, %ymm6, %ymm6	# tmp655, vect__132.845, vect_patt_431.849
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vcvtdq2pd	%xmm6, %ymm7	# vect_patt_431.849, vect__134.850
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_431.849, tmp660
	vcvtdq2pd	%xmm6, %ymm6	# tmp660, vect__134.850
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1783, vect__134.850, vect__141.851
	vdivpd	%ymm0, %ymm7, %ymm7	# tmp1783, vect__134.850, vect__141.851
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm6, -896(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 224B]
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpmulld	%ymm12, %ymm4, %ymm6	# tmp1763, tmp530, vect__131.844
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpaddd	%ymm5, %ymm6, %ymm6	# tmp1778, vect__131.844, vect__132.845
	vpmuldq	%ymm1, %ymm6, %ymm8	# tmp1779, vect__132.845, tmp666
	vpshufb	%ymm2, %ymm8, %ymm8	# tmp1781, tmp666, tmp679
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm7, -928(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 192B]
	vpsrlq	$32, %ymm6, %ymm7	#, vect__132.845, tmp670
	vpmuldq	%ymm1, %ymm7, %ymm7	# tmp1779, tmp670, tmp668
	vpshufb	%ymm3, %ymm7, %ymm7	# tmp1782, tmp668, tmp681
	vpor	%ymm7, %ymm8, %ymm8	# tmp681, tmp679, tmp674
	vpsrad	$5, %ymm8, %ymm8	#, tmp674, vect_patt_429.847
	vpslld	$1, %ymm8, %ymm7	#, vect_patt_429.847, tmp684
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp684, tmp685
	vpslld	$3, %ymm7, %ymm7	#, tmp685, tmp686
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp686, vect_patt_430.848
	vpslld	$2, %ymm7, %ymm7	#, vect_patt_430.848, tmp688
	vpsubd	%ymm7, %ymm6, %ymm6	# tmp688, vect__132.845, vect_patt_431.849
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vcvtdq2pd	%xmm6, %ymm7	# vect_patt_431.849, vect__134.850
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_431.849, tmp693
	vcvtdq2pd	%xmm6, %ymm6	# tmp693, vect__134.850
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1783, vect__134.850, vect__141.851
	vdivpd	%ymm0, %ymm7, %ymm7	# tmp1783, vect__134.850, vect__141.851
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm6, -832(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 288B]
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpmulld	.LC12(%rip), %ymm4, %ymm6	#, tmp530, vect__131.844
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpaddd	%ymm5, %ymm6, %ymm6	# tmp1778, vect__131.844, vect__132.845
	vpmuldq	%ymm1, %ymm6, %ymm8	# tmp1779, vect__132.845, tmp699
	vpshufb	%ymm2, %ymm8, %ymm8	# tmp1781, tmp699, tmp712
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm7, -864(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 256B]
	vpsrlq	$32, %ymm6, %ymm7	#, vect__132.845, tmp703
	vpmuldq	%ymm1, %ymm7, %ymm7	# tmp1779, tmp703, tmp701
	vpshufb	%ymm3, %ymm7, %ymm7	# tmp1782, tmp701, tmp714
	vpor	%ymm7, %ymm8, %ymm8	# tmp714, tmp712, tmp707
	vpsrad	$5, %ymm8, %ymm8	#, tmp707, vect_patt_429.847
	vpslld	$1, %ymm8, %ymm7	#, vect_patt_429.847, tmp717
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp717, tmp718
	vpslld	$3, %ymm7, %ymm7	#, tmp718, tmp719
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp719, vect_patt_430.848
	vpslld	$2, %ymm7, %ymm7	#, vect_patt_430.848, tmp721
	vpsubd	%ymm7, %ymm6, %ymm6	# tmp721, vect__132.845, vect_patt_431.849
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vcvtdq2pd	%xmm6, %ymm7	# vect_patt_431.849, vect__134.850
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_431.849, tmp726
	vcvtdq2pd	%xmm6, %ymm6	# tmp726, vect__134.850
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1783, vect__134.850, vect__141.851
	vdivpd	%ymm0, %ymm7, %ymm7	# tmp1783, vect__134.850, vect__141.851
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm6, -768(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 352B]
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpmulld	.LC13(%rip), %ymm4, %ymm6	#, tmp530, vect__131.844
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpaddd	%ymm5, %ymm6, %ymm6	# tmp1778, vect__131.844, vect__132.845
	vpmuldq	%ymm1, %ymm6, %ymm8	# tmp1779, vect__132.845, tmp732
	vpshufb	%ymm2, %ymm8, %ymm8	# tmp1781, tmp732, tmp745
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm7, -800(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 320B]
	vpsrlq	$32, %ymm6, %ymm7	#, vect__132.845, tmp736
	vpmuldq	%ymm1, %ymm7, %ymm7	# tmp1779, tmp736, tmp734
	vpshufb	%ymm3, %ymm7, %ymm7	# tmp1782, tmp734, tmp747
	vpor	%ymm7, %ymm8, %ymm8	# tmp747, tmp745, tmp740
	vpsrad	$5, %ymm8, %ymm8	#, tmp740, vect_patt_429.847
	vpslld	$1, %ymm8, %ymm7	#, vect_patt_429.847, tmp750
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp750, tmp751
	vpslld	$3, %ymm7, %ymm7	#, tmp751, tmp752
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp752, vect_patt_430.848
	vpslld	$2, %ymm7, %ymm7	#, vect_patt_430.848, tmp754
	vpsubd	%ymm7, %ymm6, %ymm6	# tmp754, vect__132.845, vect_patt_431.849
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vcvtdq2pd	%xmm6, %ymm7	# vect_patt_431.849, vect__134.850
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_431.849, tmp759
	vcvtdq2pd	%xmm6, %ymm6	# tmp759, vect__134.850
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1783, vect__134.850, vect__141.851
	vdivpd	%ymm0, %ymm7, %ymm7	# tmp1783, vect__134.850, vect__141.851
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm6, -704(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 416B]
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpmulld	.LC14(%rip), %ymm4, %ymm6	#, tmp530, vect__131.844
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpaddd	%ymm5, %ymm6, %ymm6	# tmp1778, vect__131.844, vect__132.845
	vpmuldq	%ymm1, %ymm6, %ymm8	# tmp1779, vect__132.845, tmp765
	vpshufb	%ymm2, %ymm8, %ymm8	# tmp1781, tmp765, tmp778
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm7, -736(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 384B]
	vpsrlq	$32, %ymm6, %ymm7	#, vect__132.845, tmp769
	vpmuldq	%ymm1, %ymm7, %ymm7	# tmp1779, tmp769, tmp767
	vpshufb	%ymm3, %ymm7, %ymm7	# tmp1782, tmp767, tmp780
	vpor	%ymm7, %ymm8, %ymm8	# tmp780, tmp778, tmp773
	vpsrad	$5, %ymm8, %ymm8	#, tmp773, vect_patt_429.847
	vpslld	$1, %ymm8, %ymm7	#, vect_patt_429.847, tmp783
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp783, tmp784
	vpslld	$3, %ymm7, %ymm7	#, tmp784, tmp785
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp785, vect_patt_430.848
	vpslld	$2, %ymm7, %ymm7	#, vect_patt_430.848, tmp787
	vpsubd	%ymm7, %ymm6, %ymm6	# tmp787, vect__132.845, vect_patt_431.849
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vcvtdq2pd	%xmm6, %ymm7	# vect_patt_431.849, vect__134.850
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_431.849, tmp792
	vcvtdq2pd	%xmm6, %ymm6	# tmp792, vect__134.850
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1783, vect__134.850, vect__141.851
	vdivpd	%ymm0, %ymm7, %ymm7	# tmp1783, vect__134.850, vect__141.851
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm6, -640(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 480B]
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpmulld	.LC15(%rip), %ymm4, %ymm6	#, tmp530, vect__131.844
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpaddd	%ymm5, %ymm6, %ymm6	# tmp1778, vect__131.844, vect__132.845
	vpmuldq	%ymm1, %ymm6, %ymm8	# tmp1779, vect__132.845, tmp798
	vpshufb	%ymm2, %ymm8, %ymm8	# tmp1781, tmp798, tmp811
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm7, -672(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 448B]
	vpsrlq	$32, %ymm6, %ymm7	#, vect__132.845, tmp802
	vpmuldq	%ymm1, %ymm7, %ymm7	# tmp1779, tmp802, tmp800
	vpshufb	%ymm3, %ymm7, %ymm7	# tmp1782, tmp800, tmp813
	vpor	%ymm7, %ymm8, %ymm8	# tmp813, tmp811, tmp806
	vpsrad	$5, %ymm8, %ymm8	#, tmp806, vect_patt_429.847
	vpslld	$1, %ymm8, %ymm7	#, vect_patt_429.847, tmp816
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp816, tmp817
	vpslld	$3, %ymm7, %ymm7	#, tmp817, tmp818
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp818, vect_patt_430.848
	vpslld	$2, %ymm7, %ymm7	#, vect_patt_430.848, tmp820
	vpsubd	%ymm7, %ymm6, %ymm6	# tmp820, vect__132.845, vect_patt_431.849
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vcvtdq2pd	%xmm6, %ymm7	# vect_patt_431.849, vect__134.850
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_431.849, tmp825
	vcvtdq2pd	%xmm6, %ymm6	# tmp825, vect__134.850
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1783, vect__134.850, vect__141.851
	vdivpd	%ymm0, %ymm7, %ymm7	# tmp1783, vect__134.850, vect__141.851
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm6, -576(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 544B]
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpmulld	.LC16(%rip), %ymm4, %ymm6	#, tmp530, vect__131.844
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpaddd	%ymm5, %ymm6, %ymm6	# tmp1778, vect__131.844, vect__132.845
	vpmuldq	%ymm1, %ymm6, %ymm8	# tmp1779, vect__132.845, tmp831
	vpshufb	%ymm2, %ymm8, %ymm8	# tmp1781, tmp831, tmp844
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm7, -608(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 512B]
	vpsrlq	$32, %ymm6, %ymm7	#, vect__132.845, tmp835
	vpmuldq	%ymm1, %ymm7, %ymm7	# tmp1779, tmp835, tmp833
	vpshufb	%ymm3, %ymm7, %ymm7	# tmp1782, tmp833, tmp846
	vpor	%ymm7, %ymm8, %ymm8	# tmp846, tmp844, tmp839
	vpsrad	$5, %ymm8, %ymm8	#, tmp839, vect_patt_429.847
	vpslld	$1, %ymm8, %ymm7	#, vect_patt_429.847, tmp849
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp849, tmp850
	vpslld	$3, %ymm7, %ymm7	#, tmp850, tmp851
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp851, vect_patt_430.848
	vpslld	$2, %ymm7, %ymm7	#, vect_patt_430.848, tmp853
	vpsubd	%ymm7, %ymm6, %ymm6	# tmp853, vect__132.845, vect_patt_431.849
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vcvtdq2pd	%xmm6, %ymm7	# vect_patt_431.849, vect__134.850
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_431.849, tmp858
	vcvtdq2pd	%xmm6, %ymm6	# tmp858, vect__134.850
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1783, vect__134.850, vect__141.851
	vdivpd	%ymm0, %ymm7, %ymm7	# tmp1783, vect__134.850, vect__141.851
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm6, -512(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 608B]
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpmulld	.LC17(%rip), %ymm4, %ymm6	#, tmp530, vect__131.844
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpaddd	%ymm5, %ymm6, %ymm6	# tmp1778, vect__131.844, vect__132.845
	vpmuldq	%ymm1, %ymm6, %ymm8	# tmp1779, vect__132.845, tmp864
	vpshufb	%ymm2, %ymm8, %ymm8	# tmp1781, tmp864, tmp877
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm7, -544(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 576B]
	vpsrlq	$32, %ymm6, %ymm7	#, vect__132.845, tmp868
	vpmuldq	%ymm1, %ymm7, %ymm7	# tmp1779, tmp868, tmp866
	vpshufb	%ymm3, %ymm7, %ymm7	# tmp1782, tmp866, tmp879
	vpor	%ymm7, %ymm8, %ymm8	# tmp879, tmp877, tmp872
	vpsrad	$5, %ymm8, %ymm8	#, tmp872, vect_patt_429.847
	vpslld	$1, %ymm8, %ymm7	#, vect_patt_429.847, tmp882
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp882, tmp883
	vpslld	$3, %ymm7, %ymm7	#, tmp883, tmp884
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp884, vect_patt_430.848
	vpslld	$2, %ymm7, %ymm7	#, vect_patt_430.848, tmp886
	vpsubd	%ymm7, %ymm6, %ymm6	# tmp886, vect__132.845, vect_patt_431.849
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vcvtdq2pd	%xmm6, %ymm7	# vect_patt_431.849, vect__134.850
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_431.849, tmp891
	vcvtdq2pd	%xmm6, %ymm6	# tmp891, vect__134.850
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1783, vect__134.850, vect__141.851
	vdivpd	%ymm0, %ymm7, %ymm7	# tmp1783, vect__134.850, vect__141.851
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm6, -448(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 672B]
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpmulld	.LC18(%rip), %ymm4, %ymm6	#, tmp530, vect__131.844
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpaddd	%ymm5, %ymm6, %ymm6	# tmp1778, vect__131.844, vect__132.845
	vpmuldq	%ymm1, %ymm6, %ymm8	# tmp1779, vect__132.845, tmp897
	vpshufb	%ymm2, %ymm8, %ymm8	# tmp1781, tmp897, tmp910
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm7, -480(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 640B]
	vpsrlq	$32, %ymm6, %ymm7	#, vect__132.845, tmp901
	vpmuldq	%ymm1, %ymm7, %ymm7	# tmp1779, tmp901, tmp899
	vpshufb	%ymm3, %ymm7, %ymm7	# tmp1782, tmp899, tmp912
	vpor	%ymm7, %ymm8, %ymm8	# tmp912, tmp910, tmp905
	vpsrad	$5, %ymm8, %ymm8	#, tmp905, vect_patt_429.847
	vpslld	$1, %ymm8, %ymm7	#, vect_patt_429.847, tmp915
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp915, tmp916
	vpslld	$3, %ymm7, %ymm7	#, tmp916, tmp917
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp917, vect_patt_430.848
	vpslld	$2, %ymm7, %ymm7	#, vect_patt_430.848, tmp919
	vpsubd	%ymm7, %ymm6, %ymm6	# tmp919, vect__132.845, vect_patt_431.849
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vcvtdq2pd	%xmm6, %ymm7	# vect_patt_431.849, vect__134.850
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_431.849, tmp924
	vcvtdq2pd	%xmm6, %ymm6	# tmp924, vect__134.850
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1783, vect__134.850, vect__141.851
	vdivpd	%ymm0, %ymm7, %ymm7	# tmp1783, vect__134.850, vect__141.851
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm6, -384(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 736B]
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpmulld	.LC19(%rip), %ymm4, %ymm6	#, tmp530, vect__131.844
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpaddd	%ymm5, %ymm6, %ymm6	# tmp1778, vect__131.844, vect__132.845
	vpmuldq	%ymm1, %ymm6, %ymm8	# tmp1779, vect__132.845, tmp930
	vpshufb	%ymm2, %ymm8, %ymm8	# tmp1781, tmp930, tmp943
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm7, -416(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 704B]
	vpsrlq	$32, %ymm6, %ymm7	#, vect__132.845, tmp934
	vpmuldq	%ymm1, %ymm7, %ymm7	# tmp1779, tmp934, tmp932
	vpshufb	%ymm3, %ymm7, %ymm7	# tmp1782, tmp932, tmp945
	vpor	%ymm7, %ymm8, %ymm8	# tmp945, tmp943, tmp938
	vpsrad	$5, %ymm8, %ymm8	#, tmp938, vect_patt_429.847
	vpslld	$1, %ymm8, %ymm7	#, vect_patt_429.847, tmp948
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp948, tmp949
	vpslld	$3, %ymm7, %ymm7	#, tmp949, tmp950
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp950, vect_patt_430.848
	vpslld	$2, %ymm7, %ymm7	#, vect_patt_430.848, tmp952
	vpsubd	%ymm7, %ymm6, %ymm6	# tmp952, vect__132.845, vect_patt_431.849
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vcvtdq2pd	%xmm6, %ymm7	# vect_patt_431.849, vect__134.850
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_431.849, tmp957
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vdivpd	%ymm0, %ymm7, %ymm7	# tmp1783, vect__134.850, vect__141.851
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vcvtdq2pd	%xmm6, %ymm6	# tmp957, vect__134.850
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1783, vect__134.850, vect__141.851
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm7, -352(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 768B]
	vmovupd	%ymm6, -320(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 800B]
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpmulld	.LC20(%rip), %ymm4, %ymm6	#, tmp530, vect__131.844
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpaddd	%ymm5, %ymm6, %ymm6	# tmp1778, vect__131.844, vect__132.845
	vpsrlq	$32, %ymm6, %ymm7	#, vect__132.845, tmp967
	vpmuldq	%ymm1, %ymm6, %ymm8	# tmp1779, vect__132.845, tmp963
	vpmuldq	%ymm1, %ymm7, %ymm7	# tmp1779, tmp967, tmp965
	vpshufb	%ymm2, %ymm8, %ymm8	# tmp1781, tmp963, tmp976
	vpshufb	%ymm3, %ymm7, %ymm7	# tmp1782, tmp965, tmp978
	vpor	%ymm7, %ymm8, %ymm8	# tmp978, tmp976, tmp971
	vpsrad	$5, %ymm8, %ymm8	#, tmp971, vect_patt_429.847
	vpslld	$1, %ymm8, %ymm7	#, vect_patt_429.847, tmp981
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp981, tmp982
	vpslld	$3, %ymm7, %ymm7	#, tmp982, tmp983
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp983, vect_patt_430.848
	vpslld	$2, %ymm7, %ymm7	#, vect_patt_430.848, tmp985
	vpsubd	%ymm7, %ymm6, %ymm6	# tmp985, vect__132.845, vect_patt_431.849
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vcvtdq2pd	%xmm6, %ymm7	# vect_patt_431.849, vect__134.850
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_431.849, tmp990
	vcvtdq2pd	%xmm6, %ymm6	# tmp990, vect__134.850
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1783, vect__134.850, vect__141.851
	vdivpd	%ymm0, %ymm7, %ymm7	# tmp1783, vect__134.850, vect__141.851
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm6, -256(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 864B]
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpmulld	.LC21(%rip), %ymm4, %ymm6	#, tmp530, vect__131.844
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpaddd	%ymm5, %ymm6, %ymm6	# tmp1778, vect__131.844, vect__132.845
	vpmuldq	%ymm1, %ymm6, %ymm8	# tmp1779, vect__132.845, tmp996
	vpshufb	%ymm2, %ymm8, %ymm8	# tmp1781, tmp996, tmp1009
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm7, -288(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 832B]
	vpsrlq	$32, %ymm6, %ymm7	#, vect__132.845, tmp1000
	vpmuldq	%ymm1, %ymm7, %ymm7	# tmp1779, tmp1000, tmp998
	vpshufb	%ymm3, %ymm7, %ymm7	# tmp1782, tmp998, tmp1011
	vpor	%ymm7, %ymm8, %ymm8	# tmp1011, tmp1009, tmp1004
	vpsrad	$5, %ymm8, %ymm8	#, tmp1004, vect_patt_429.847
	vpslld	$1, %ymm8, %ymm7	#, vect_patt_429.847, tmp1014
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp1014, tmp1015
	vpslld	$3, %ymm7, %ymm7	#, tmp1015, tmp1016
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp1016, vect_patt_430.848
	vpslld	$2, %ymm7, %ymm7	#, vect_patt_430.848, tmp1018
	vpsubd	%ymm7, %ymm6, %ymm6	# tmp1018, vect__132.845, vect_patt_431.849
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vcvtdq2pd	%xmm6, %ymm7	# vect_patt_431.849, vect__134.850
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_431.849, tmp1023
	vcvtdq2pd	%xmm6, %ymm6	# tmp1023, vect__134.850
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1783, vect__134.850, vect__141.851
	vdivpd	%ymm0, %ymm7, %ymm7	# tmp1783, vect__134.850, vect__141.851
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm6, -192(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 928B]
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpmulld	.LC22(%rip), %ymm4, %ymm6	#, tmp530, vect__131.844
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpaddd	%ymm5, %ymm6, %ymm6	# tmp1778, vect__131.844, vect__132.845
	vpmuldq	%ymm1, %ymm6, %ymm8	# tmp1779, vect__132.845, tmp1029
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpmulld	.LC23(%rip), %ymm4, %ymm4	#, tmp530, vect__131.844
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpaddd	%ymm5, %ymm4, %ymm4	# tmp1778, vect__131.844, vect__132.845
	vpshufb	%ymm2, %ymm8, %ymm8	# tmp1781, tmp1029, tmp1042
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm7, -224(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 896B]
	vpsrlq	$32, %ymm6, %ymm7	#, vect__132.845, tmp1033
	vpmuldq	%ymm1, %ymm7, %ymm7	# tmp1779, tmp1033, tmp1031
	vpshufb	%ymm3, %ymm7, %ymm7	# tmp1782, tmp1031, tmp1044
	vpor	%ymm7, %ymm8, %ymm8	# tmp1044, tmp1042, tmp1037
	vpsrad	$5, %ymm8, %ymm8	#, tmp1037, vect_patt_429.847
	vpslld	$1, %ymm8, %ymm7	#, vect_patt_429.847, tmp1047
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp1047, tmp1048
	vpslld	$3, %ymm7, %ymm7	#, tmp1048, tmp1049
	vpaddd	%ymm8, %ymm7, %ymm7	# vect_patt_429.847, tmp1049, vect_patt_430.848
	vpslld	$2, %ymm7, %ymm7	#, vect_patt_430.848, tmp1051
	vpsubd	%ymm7, %ymm6, %ymm6	# tmp1051, vect__132.845, vect_patt_431.849
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vcvtdq2pd	%xmm6, %ymm7	# vect_patt_431.849, vect__134.850
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_431.849, tmp1056
	vcvtdq2pd	%xmm6, %ymm6	# tmp1056, vect__134.850
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1783, vect__134.850, vect__141.851
	vdivpd	%ymm0, %ymm7, %ymm7	# tmp1783, vect__134.850, vect__141.851
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm6, -128(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 992B]
	vpsrlq	$32, %ymm4, %ymm6	#, vect__132.845, tmp1066
	vpmuldq	%ymm1, %ymm6, %ymm6	# tmp1779, tmp1066, tmp1064
	vpshufb	%ymm3, %ymm6, %ymm6	# tmp1782, tmp1064, tmp1077
	vmovupd	%ymm7, -160(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 960B]
	vpmuldq	%ymm1, %ymm4, %ymm7	# tmp1779, vect__132.845, tmp1062
	vpshufb	%ymm2, %ymm7, %ymm7	# tmp1781, tmp1062, tmp1075
	vpor	%ymm6, %ymm7, %ymm7	# tmp1077, tmp1075, tmp1070
	vpsrad	$5, %ymm7, %ymm7	#, tmp1070, vect_patt_429.847
	vpslld	$1, %ymm7, %ymm6	#, vect_patt_429.847, tmp1080
	vpaddd	%ymm7, %ymm6, %ymm6	# vect_patt_429.847, tmp1080, tmp1081
	vpslld	$3, %ymm6, %ymm6	#, tmp1081, tmp1082
	vpaddd	%ymm7, %ymm6, %ymm6	# vect_patt_429.847, tmp1082, vect_patt_430.848
	vpslld	$2, %ymm6, %ymm6	#, vect_patt_430.848, tmp1084
	vpsubd	%ymm6, %ymm4, %ymm4	# tmp1084, vect__132.845, vect_patt_431.849
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vcvtdq2pd	%xmm4, %ymm6	# vect_patt_431.849, vect__134.850
	vextracti128	$0x1, %ymm4, %xmm4	# vect_patt_431.849, tmp1089
	vcvtdq2pd	%xmm4, %ymm4	# tmp1089, vect__134.850
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vdivpd	%ymm0, %ymm4, %ymm4	# tmp1783, vect__134.850, vect__141.851
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1783, vect__134.850, vect__141.851
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm4, -64(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 1056B]
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovd	%edx, %xmm4	# i, i
	vpshufd	$0, %xmm4, %xmm4	# i, tmp1092
	vpmulld	.LC24(%rip), %xmm4, %xmm4	#, tmp1092, vect__446.855
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vpaddd	.LC25(%rip), %xmm4, %xmm4	#, vect__446.855, vect__447.856
# benchmark_2mm.c:70:     for (int i = 0; i < ni; i++)
	incl	%edx	# i
	vpmuldq	%xmm10, %xmm4, %xmm7	# tmp1731, vect__447.856, tmp1096
	vpshufb	.LC27(%rip), %xmm7, %xmm7	#, tmp1096, tmp1105
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%ymm6, -96(%rax)	# vect__141.851, MEM <vector(4) double> [(double *)_826 + 1024B]
	vpsrlq	$32, %xmm4, %xmm6	#, vect__447.856, tmp1100
	vpmuldq	%xmm10, %xmm6, %xmm6	# tmp1731, tmp1100, tmp1098
	vpshufb	.LC28(%rip), %xmm6, %xmm6	#, tmp1098, tmp1107
	vpor	%xmm6, %xmm7, %xmm7	# tmp1107, tmp1105, tmp1108
	vpsrad	$5, %xmm7, %xmm7	#, tmp1108, vect_patt_436.858
	vpslld	$1, %xmm7, %xmm6	#, vect_patt_436.858, tmp1111
	vpaddd	%xmm7, %xmm6, %xmm6	# vect_patt_436.858, tmp1111, tmp1112
	vpslld	$3, %xmm6, %xmm6	#, tmp1112, tmp1113
	vpaddd	%xmm7, %xmm6, %xmm6	# vect_patt_436.858, tmp1113, vect_patt_437.859
	vpslld	$2, %xmm6, %xmm6	#, vect_patt_437.859, tmp1115
	vpsubd	%xmm6, %xmm4, %xmm4	# tmp1115, vect__447.856, vect_patt_438.860
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vcvtdq2pd	%xmm4, %xmm6	# vect_patt_438.860, vect__449.861
	vpshufd	$238, %xmm4, %xmm4	#, vect_patt_438.860, tmp1120
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vdivpd	%xmm9, %xmm6, %xmm6	# tmp1734, vect__449.861, vect__454.862
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vcvtdq2pd	%xmm4, %xmm4	# tmp1120, vect__449.861
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vdivpd	%xmm9, %xmm4, %xmm4	# tmp1734, vect__449.861, vect__454.862
# benchmark_2mm.c:72:             A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
	vmovupd	%xmm6, -32(%rax)	# vect__454.862, MEM <vector(2) double> [(double *)_826 + 1088B]
	vmovupd	%xmm4, -16(%rax)	# vect__454.862, MEM <vector(2) double> [(double *)_826 + 1104B]
# benchmark_2mm.c:70:     for (int i = 0; i < ni; i++)
	cmpl	$100, %edx	#, i
	jne	.L361	#,
	vmovdqa	.LC30(%rip), %ymm13	#, tmp1739
	vmovdqa	.LC31(%rip), %ymm1	#, tmp1740
	vmovapd	.LC32(%rip), %ymm0	#, tmp1741
	vmovdqa	.LC33(%rip), %ymm12	#, tmp1742
	vmovdqa	.LC34(%rip), %ymm10	#, tmp1743
	vmovdqa	.LC35(%rip), %ymm9	#, tmp1744
	vmovdqa	.LC36(%rip), %ymm8	#, tmp1745
	vmovdqa	.LC37(%rip), %ymm7	#, tmp1746
	movq	%r13, %rax	# B, ivtmp.955
# benchmark_2mm.c:74:     for (int i = 0; i < nk; i++)
	xorl	%edx, %edx	# i
.L362:
	vmovd	%edx, %xmm4	# i, tmp1123
	vpbroadcastd	%xmm4, %ymm4	# tmp1123, tmp1123
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vpmulld	%ymm13, %ymm4, %ymm6	# tmp1739, tmp1123, vect__147.831
# benchmark_2mm.c:74:     for (int i = 0; i < nk; i++)
	incl	%edx	# i
# benchmark_2mm.c:74:     for (int i = 0; i < nk; i++)
	addq	$960, %rax	#, ivtmp.955
	vpsrlq	$32, %ymm6, %ymm15	#, vect__147.831, tmp1129
	vpmuldq	%ymm1, %ymm6, %ymm14	# tmp1740, vect__147.831, tmp1125
	vpmuldq	%ymm1, %ymm15, %ymm15	# tmp1740, tmp1129, tmp1127
	vpshufb	%ymm2, %ymm14, %ymm14	# tmp1781, tmp1125, tmp1138
	vpshufb	%ymm3, %ymm15, %ymm15	# tmp1782, tmp1127, tmp1140
	vpor	%ymm15, %ymm14, %ymm14	# tmp1140, tmp1138, tmp1133
	vpaddd	%ymm14, %ymm6, %ymm14	# tmp1133, vect__147.831, vect_patt_385.833
	vpsrad	$6, %ymm14, %ymm14	#, vect_patt_385.833, vect_patt_386.834
	vpslld	$4, %ymm14, %ymm15	#, vect_patt_386.834, tmp1144
	vpsubd	%ymm14, %ymm15, %ymm14	# vect_patt_386.834, tmp1144, vect_patt_387.835
	vpslld	$3, %ymm14, %ymm14	#, vect_patt_387.835, tmp1146
	vpsubd	%ymm14, %ymm6, %ymm6	# tmp1146, vect__147.831, vect_patt_388.836
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vcvtdq2pd	%xmm6, %ymm14	# vect_patt_388.836, vect__149.837
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_388.836, tmp1151
	vcvtdq2pd	%xmm6, %ymm6	# tmp1151, vect__149.837
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1741, vect__149.837, vect__156.838
	vdivpd	%ymm0, %ymm14, %ymm14	# tmp1741, vect__149.837, vect__156.838
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vmovupd	%ymm6, -928(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703 + 32B]
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vpmulld	%ymm12, %ymm4, %ymm6	# tmp1742, tmp1123, vect__147.831
	vpsrlq	$32, %ymm6, %ymm15	#, vect__147.831, tmp1159
	vpmuldq	%ymm1, %ymm15, %ymm15	# tmp1740, tmp1159, tmp1157
	vpshufb	%ymm3, %ymm15, %ymm15	# tmp1782, tmp1157, tmp1170
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vmovupd	%ymm14, -960(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703]
	vpmuldq	%ymm1, %ymm6, %ymm14	# tmp1740, vect__147.831, tmp1155
	vpshufb	%ymm2, %ymm14, %ymm14	# tmp1781, tmp1155, tmp1168
	vpor	%ymm15, %ymm14, %ymm14	# tmp1170, tmp1168, tmp1163
	vpaddd	%ymm14, %ymm6, %ymm14	# tmp1163, vect__147.831, vect_patt_385.833
	vpsrad	$6, %ymm14, %ymm14	#, vect_patt_385.833, vect_patt_386.834
	vpslld	$4, %ymm14, %ymm15	#, vect_patt_386.834, tmp1174
	vpsubd	%ymm14, %ymm15, %ymm14	# vect_patt_386.834, tmp1174, vect_patt_387.835
	vpslld	$3, %ymm14, %ymm14	#, vect_patt_387.835, tmp1176
	vpsubd	%ymm14, %ymm6, %ymm6	# tmp1176, vect__147.831, vect_patt_388.836
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vcvtdq2pd	%xmm6, %ymm14	# vect_patt_388.836, vect__149.837
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_388.836, tmp1181
	vcvtdq2pd	%xmm6, %ymm6	# tmp1181, vect__149.837
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1741, vect__149.837, vect__156.838
	vdivpd	%ymm0, %ymm14, %ymm14	# tmp1741, vect__149.837, vect__156.838
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vmovupd	%ymm6, -864(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703 + 96B]
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vpmulld	%ymm10, %ymm4, %ymm6	# tmp1743, tmp1123, vect__147.831
	vpsrlq	$32, %ymm6, %ymm15	#, vect__147.831, tmp1189
	vpmuldq	%ymm1, %ymm15, %ymm15	# tmp1740, tmp1189, tmp1187
	vpshufb	%ymm3, %ymm15, %ymm15	# tmp1782, tmp1187, tmp1200
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vmovupd	%ymm14, -896(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703 + 64B]
	vpmuldq	%ymm1, %ymm6, %ymm14	# tmp1740, vect__147.831, tmp1185
	vpshufb	%ymm2, %ymm14, %ymm14	# tmp1781, tmp1185, tmp1198
	vpor	%ymm15, %ymm14, %ymm14	# tmp1200, tmp1198, tmp1193
	vpaddd	%ymm14, %ymm6, %ymm14	# tmp1193, vect__147.831, vect_patt_385.833
	vpsrad	$6, %ymm14, %ymm14	#, vect_patt_385.833, vect_patt_386.834
	vpslld	$4, %ymm14, %ymm15	#, vect_patt_386.834, tmp1204
	vpsubd	%ymm14, %ymm15, %ymm14	# vect_patt_386.834, tmp1204, vect_patt_387.835
	vpslld	$3, %ymm14, %ymm14	#, vect_patt_387.835, tmp1206
	vpsubd	%ymm14, %ymm6, %ymm6	# tmp1206, vect__147.831, vect_patt_388.836
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vcvtdq2pd	%xmm6, %ymm14	# vect_patt_388.836, vect__149.837
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_388.836, tmp1211
	vcvtdq2pd	%xmm6, %ymm6	# tmp1211, vect__149.837
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1741, vect__149.837, vect__156.838
	vdivpd	%ymm0, %ymm14, %ymm14	# tmp1741, vect__149.837, vect__156.838
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vmovupd	%ymm6, -800(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703 + 160B]
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vpmulld	%ymm9, %ymm4, %ymm6	# tmp1744, tmp1123, vect__147.831
	vpsrlq	$32, %ymm6, %ymm15	#, vect__147.831, tmp1219
	vpmuldq	%ymm1, %ymm15, %ymm15	# tmp1740, tmp1219, tmp1217
	vpshufb	%ymm3, %ymm15, %ymm15	# tmp1782, tmp1217, tmp1230
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vmovupd	%ymm14, -832(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703 + 128B]
	vpmuldq	%ymm1, %ymm6, %ymm14	# tmp1740, vect__147.831, tmp1215
	vpshufb	%ymm2, %ymm14, %ymm14	# tmp1781, tmp1215, tmp1228
	vpor	%ymm15, %ymm14, %ymm14	# tmp1230, tmp1228, tmp1223
	vpaddd	%ymm14, %ymm6, %ymm14	# tmp1223, vect__147.831, vect_patt_385.833
	vpsrad	$6, %ymm14, %ymm14	#, vect_patt_385.833, vect_patt_386.834
	vpslld	$4, %ymm14, %ymm15	#, vect_patt_386.834, tmp1234
	vpsubd	%ymm14, %ymm15, %ymm14	# vect_patt_386.834, tmp1234, vect_patt_387.835
	vpslld	$3, %ymm14, %ymm14	#, vect_patt_387.835, tmp1236
	vpsubd	%ymm14, %ymm6, %ymm6	# tmp1236, vect__147.831, vect_patt_388.836
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vcvtdq2pd	%xmm6, %ymm14	# vect_patt_388.836, vect__149.837
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_388.836, tmp1241
	vcvtdq2pd	%xmm6, %ymm6	# tmp1241, vect__149.837
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1741, vect__149.837, vect__156.838
	vdivpd	%ymm0, %ymm14, %ymm14	# tmp1741, vect__149.837, vect__156.838
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vmovupd	%ymm6, -736(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703 + 224B]
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vpmulld	%ymm8, %ymm4, %ymm6	# tmp1745, tmp1123, vect__147.831
	vpsrlq	$32, %ymm6, %ymm15	#, vect__147.831, tmp1249
	vpmuldq	%ymm1, %ymm15, %ymm15	# tmp1740, tmp1249, tmp1247
	vpshufb	%ymm3, %ymm15, %ymm15	# tmp1782, tmp1247, tmp1260
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vmovupd	%ymm14, -768(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703 + 192B]
	vpmuldq	%ymm1, %ymm6, %ymm14	# tmp1740, vect__147.831, tmp1245
	vpshufb	%ymm2, %ymm14, %ymm14	# tmp1781, tmp1245, tmp1258
	vpor	%ymm15, %ymm14, %ymm14	# tmp1260, tmp1258, tmp1253
	vpaddd	%ymm14, %ymm6, %ymm14	# tmp1253, vect__147.831, vect_patt_385.833
	vpsrad	$6, %ymm14, %ymm14	#, vect_patt_385.833, vect_patt_386.834
	vpslld	$4, %ymm14, %ymm15	#, vect_patt_386.834, tmp1264
	vpsubd	%ymm14, %ymm15, %ymm14	# vect_patt_386.834, tmp1264, vect_patt_387.835
	vpslld	$3, %ymm14, %ymm14	#, vect_patt_387.835, tmp1266
	vpsubd	%ymm14, %ymm6, %ymm6	# tmp1266, vect__147.831, vect_patt_388.836
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vcvtdq2pd	%xmm6, %ymm14	# vect_patt_388.836, vect__149.837
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_388.836, tmp1271
	vcvtdq2pd	%xmm6, %ymm6	# tmp1271, vect__149.837
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1741, vect__149.837, vect__156.838
	vdivpd	%ymm0, %ymm14, %ymm14	# tmp1741, vect__149.837, vect__156.838
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vmovupd	%ymm6, -672(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703 + 288B]
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vpmulld	%ymm7, %ymm4, %ymm6	# tmp1746, tmp1123, vect__147.831
	vpsrlq	$32, %ymm6, %ymm15	#, vect__147.831, tmp1279
	vpmuldq	%ymm1, %ymm15, %ymm15	# tmp1740, tmp1279, tmp1277
	vpshufb	%ymm3, %ymm15, %ymm15	# tmp1782, tmp1277, tmp1290
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vmovupd	%ymm14, -704(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703 + 256B]
	vpmuldq	%ymm1, %ymm6, %ymm14	# tmp1740, vect__147.831, tmp1275
	vpshufb	%ymm2, %ymm14, %ymm14	# tmp1781, tmp1275, tmp1288
	vpor	%ymm15, %ymm14, %ymm14	# tmp1290, tmp1288, tmp1283
	vpaddd	%ymm14, %ymm6, %ymm14	# tmp1283, vect__147.831, vect_patt_385.833
	vpsrad	$6, %ymm14, %ymm14	#, vect_patt_385.833, vect_patt_386.834
	vpslld	$4, %ymm14, %ymm15	#, vect_patt_386.834, tmp1294
	vpsubd	%ymm14, %ymm15, %ymm14	# vect_patt_386.834, tmp1294, vect_patt_387.835
	vpslld	$3, %ymm14, %ymm14	#, vect_patt_387.835, tmp1296
	vpsubd	%ymm14, %ymm6, %ymm6	# tmp1296, vect__147.831, vect_patt_388.836
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vcvtdq2pd	%xmm6, %ymm14	# vect_patt_388.836, vect__149.837
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_388.836, tmp1301
	vcvtdq2pd	%xmm6, %ymm6	# tmp1301, vect__149.837
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1741, vect__149.837, vect__156.838
	vdivpd	%ymm0, %ymm14, %ymm14	# tmp1741, vect__149.837, vect__156.838
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vmovupd	%ymm6, -608(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703 + 352B]
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vpmulld	.LC38(%rip), %ymm4, %ymm6	#, tmp1123, vect__147.831
	vpsrlq	$32, %ymm6, %ymm15	#, vect__147.831, tmp1309
	vpmuldq	%ymm1, %ymm15, %ymm15	# tmp1740, tmp1309, tmp1307
	vpshufb	%ymm3, %ymm15, %ymm15	# tmp1782, tmp1307, tmp1320
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vmovupd	%ymm14, -640(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703 + 320B]
	vpmuldq	%ymm1, %ymm6, %ymm14	# tmp1740, vect__147.831, tmp1305
	vpshufb	%ymm2, %ymm14, %ymm14	# tmp1781, tmp1305, tmp1318
	vpor	%ymm15, %ymm14, %ymm14	# tmp1320, tmp1318, tmp1313
	vpaddd	%ymm14, %ymm6, %ymm14	# tmp1313, vect__147.831, vect_patt_385.833
	vpsrad	$6, %ymm14, %ymm14	#, vect_patt_385.833, vect_patt_386.834
	vpslld	$4, %ymm14, %ymm15	#, vect_patt_386.834, tmp1324
	vpsubd	%ymm14, %ymm15, %ymm14	# vect_patt_386.834, tmp1324, vect_patt_387.835
	vpslld	$3, %ymm14, %ymm14	#, vect_patt_387.835, tmp1326
	vpsubd	%ymm14, %ymm6, %ymm6	# tmp1326, vect__147.831, vect_patt_388.836
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vcvtdq2pd	%xmm6, %ymm14	# vect_patt_388.836, vect__149.837
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_388.836, tmp1331
	vcvtdq2pd	%xmm6, %ymm6	# tmp1331, vect__149.837
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1741, vect__149.837, vect__156.838
	vdivpd	%ymm0, %ymm14, %ymm14	# tmp1741, vect__149.837, vect__156.838
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vmovupd	%ymm6, -544(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703 + 416B]
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vpmulld	.LC39(%rip), %ymm4, %ymm6	#, tmp1123, vect__147.831
	vpsrlq	$32, %ymm6, %ymm15	#, vect__147.831, tmp1339
	vpmuldq	%ymm1, %ymm15, %ymm15	# tmp1740, tmp1339, tmp1337
	vpshufb	%ymm3, %ymm15, %ymm15	# tmp1782, tmp1337, tmp1350
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vmovupd	%ymm14, -576(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703 + 384B]
	vpmuldq	%ymm1, %ymm6, %ymm14	# tmp1740, vect__147.831, tmp1335
	vpshufb	%ymm2, %ymm14, %ymm14	# tmp1781, tmp1335, tmp1348
	vpor	%ymm15, %ymm14, %ymm14	# tmp1350, tmp1348, tmp1343
	vpaddd	%ymm14, %ymm6, %ymm14	# tmp1343, vect__147.831, vect_patt_385.833
	vpsrad	$6, %ymm14, %ymm14	#, vect_patt_385.833, vect_patt_386.834
	vpslld	$4, %ymm14, %ymm15	#, vect_patt_386.834, tmp1354
	vpsubd	%ymm14, %ymm15, %ymm14	# vect_patt_386.834, tmp1354, vect_patt_387.835
	vpslld	$3, %ymm14, %ymm14	#, vect_patt_387.835, tmp1356
	vpsubd	%ymm14, %ymm6, %ymm6	# tmp1356, vect__147.831, vect_patt_388.836
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vcvtdq2pd	%xmm6, %ymm14	# vect_patt_388.836, vect__149.837
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_388.836, tmp1361
	vcvtdq2pd	%xmm6, %ymm6	# tmp1361, vect__149.837
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1741, vect__149.837, vect__156.838
	vdivpd	%ymm0, %ymm14, %ymm14	# tmp1741, vect__149.837, vect__156.838
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vmovupd	%ymm6, -480(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703 + 480B]
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vpmulld	.LC40(%rip), %ymm4, %ymm6	#, tmp1123, vect__147.831
	vpsrlq	$32, %ymm6, %ymm15	#, vect__147.831, tmp1369
	vpmuldq	%ymm1, %ymm15, %ymm15	# tmp1740, tmp1369, tmp1367
	vpshufb	%ymm3, %ymm15, %ymm15	# tmp1782, tmp1367, tmp1380
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vmovupd	%ymm14, -512(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703 + 448B]
	vpmuldq	%ymm1, %ymm6, %ymm14	# tmp1740, vect__147.831, tmp1365
	vpshufb	%ymm2, %ymm14, %ymm14	# tmp1781, tmp1365, tmp1378
	vpor	%ymm15, %ymm14, %ymm14	# tmp1380, tmp1378, tmp1373
	vpaddd	%ymm14, %ymm6, %ymm14	# tmp1373, vect__147.831, vect_patt_385.833
	vpsrad	$6, %ymm14, %ymm14	#, vect_patt_385.833, vect_patt_386.834
	vpslld	$4, %ymm14, %ymm15	#, vect_patt_386.834, tmp1384
	vpsubd	%ymm14, %ymm15, %ymm14	# vect_patt_386.834, tmp1384, vect_patt_387.835
	vpslld	$3, %ymm14, %ymm14	#, vect_patt_387.835, tmp1386
	vpsubd	%ymm14, %ymm6, %ymm6	# tmp1386, vect__147.831, vect_patt_388.836
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vcvtdq2pd	%xmm6, %ymm14	# vect_patt_388.836, vect__149.837
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_388.836, tmp1391
	vcvtdq2pd	%xmm6, %ymm6	# tmp1391, vect__149.837
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1741, vect__149.837, vect__156.838
	vdivpd	%ymm0, %ymm14, %ymm14	# tmp1741, vect__149.837, vect__156.838
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vmovupd	%ymm6, -416(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703 + 544B]
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vpmulld	.LC41(%rip), %ymm4, %ymm6	#, tmp1123, vect__147.831
	vpsrlq	$32, %ymm6, %ymm15	#, vect__147.831, tmp1399
	vpmuldq	%ymm1, %ymm15, %ymm15	# tmp1740, tmp1399, tmp1397
	vpshufb	%ymm3, %ymm15, %ymm15	# tmp1782, tmp1397, tmp1410
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vmovupd	%ymm14, -448(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703 + 512B]
	vpmuldq	%ymm1, %ymm6, %ymm14	# tmp1740, vect__147.831, tmp1395
	vpshufb	%ymm2, %ymm14, %ymm14	# tmp1781, tmp1395, tmp1408
	vpor	%ymm15, %ymm14, %ymm14	# tmp1410, tmp1408, tmp1403
	vpaddd	%ymm14, %ymm6, %ymm14	# tmp1403, vect__147.831, vect_patt_385.833
	vpsrad	$6, %ymm14, %ymm14	#, vect_patt_385.833, vect_patt_386.834
	vpslld	$4, %ymm14, %ymm15	#, vect_patt_386.834, tmp1414
	vpsubd	%ymm14, %ymm15, %ymm14	# vect_patt_386.834, tmp1414, vect_patt_387.835
	vpslld	$3, %ymm14, %ymm14	#, vect_patt_387.835, tmp1416
	vpsubd	%ymm14, %ymm6, %ymm6	# tmp1416, vect__147.831, vect_patt_388.836
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vcvtdq2pd	%xmm6, %ymm14	# vect_patt_388.836, vect__149.837
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_388.836, tmp1421
	vcvtdq2pd	%xmm6, %ymm6	# tmp1421, vect__149.837
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1741, vect__149.837, vect__156.838
	vdivpd	%ymm0, %ymm14, %ymm14	# tmp1741, vect__149.837, vect__156.838
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vmovupd	%ymm6, -352(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703 + 608B]
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vpmulld	.LC42(%rip), %ymm4, %ymm6	#, tmp1123, vect__147.831
	vpsrlq	$32, %ymm6, %ymm15	#, vect__147.831, tmp1429
	vpmuldq	%ymm1, %ymm15, %ymm15	# tmp1740, tmp1429, tmp1427
	vpshufb	%ymm3, %ymm15, %ymm15	# tmp1782, tmp1427, tmp1440
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vmovupd	%ymm14, -384(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703 + 576B]
	vpmuldq	%ymm1, %ymm6, %ymm14	# tmp1740, vect__147.831, tmp1425
	vpshufb	%ymm2, %ymm14, %ymm14	# tmp1781, tmp1425, tmp1438
	vpor	%ymm15, %ymm14, %ymm14	# tmp1440, tmp1438, tmp1433
	vpaddd	%ymm14, %ymm6, %ymm14	# tmp1433, vect__147.831, vect_patt_385.833
	vpsrad	$6, %ymm14, %ymm14	#, vect_patt_385.833, vect_patt_386.834
	vpslld	$4, %ymm14, %ymm15	#, vect_patt_386.834, tmp1444
	vpsubd	%ymm14, %ymm15, %ymm14	# vect_patt_386.834, tmp1444, vect_patt_387.835
	vpslld	$3, %ymm14, %ymm14	#, vect_patt_387.835, tmp1446
	vpsubd	%ymm14, %ymm6, %ymm6	# tmp1446, vect__147.831, vect_patt_388.836
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vcvtdq2pd	%xmm6, %ymm14	# vect_patt_388.836, vect__149.837
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_388.836, tmp1451
	vcvtdq2pd	%xmm6, %ymm6	# tmp1451, vect__149.837
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1741, vect__149.837, vect__156.838
	vdivpd	%ymm0, %ymm14, %ymm14	# tmp1741, vect__149.837, vect__156.838
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vmovupd	%ymm6, -288(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703 + 672B]
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vpmulld	.LC43(%rip), %ymm4, %ymm6	#, tmp1123, vect__147.831
	vpsrlq	$32, %ymm6, %ymm15	#, vect__147.831, tmp1459
	vpmuldq	%ymm1, %ymm15, %ymm15	# tmp1740, tmp1459, tmp1457
	vpshufb	%ymm3, %ymm15, %ymm15	# tmp1782, tmp1457, tmp1470
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vmovupd	%ymm14, -320(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703 + 640B]
	vpmuldq	%ymm1, %ymm6, %ymm14	# tmp1740, vect__147.831, tmp1455
	vpshufb	%ymm2, %ymm14, %ymm14	# tmp1781, tmp1455, tmp1468
	vpor	%ymm15, %ymm14, %ymm14	# tmp1470, tmp1468, tmp1463
	vpaddd	%ymm14, %ymm6, %ymm14	# tmp1463, vect__147.831, vect_patt_385.833
	vpsrad	$6, %ymm14, %ymm14	#, vect_patt_385.833, vect_patt_386.834
	vpslld	$4, %ymm14, %ymm15	#, vect_patt_386.834, tmp1474
	vpsubd	%ymm14, %ymm15, %ymm14	# vect_patt_386.834, tmp1474, vect_patt_387.835
	vpslld	$3, %ymm14, %ymm14	#, vect_patt_387.835, tmp1476
	vpsubd	%ymm14, %ymm6, %ymm6	# tmp1476, vect__147.831, vect_patt_388.836
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vcvtdq2pd	%xmm6, %ymm14	# vect_patt_388.836, vect__149.837
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_388.836, tmp1481
	vcvtdq2pd	%xmm6, %ymm6	# tmp1481, vect__149.837
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1741, vect__149.837, vect__156.838
	vdivpd	%ymm0, %ymm14, %ymm14	# tmp1741, vect__149.837, vect__156.838
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vmovupd	%ymm6, -224(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703 + 736B]
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vpmulld	.LC44(%rip), %ymm4, %ymm6	#, tmp1123, vect__147.831
	vpsrlq	$32, %ymm6, %ymm15	#, vect__147.831, tmp1489
	vpmuldq	%ymm1, %ymm15, %ymm15	# tmp1740, tmp1489, tmp1487
	vpshufb	%ymm3, %ymm15, %ymm15	# tmp1782, tmp1487, tmp1500
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vmovupd	%ymm14, -256(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703 + 704B]
	vpmuldq	%ymm1, %ymm6, %ymm14	# tmp1740, vect__147.831, tmp1485
	vpshufb	%ymm2, %ymm14, %ymm14	# tmp1781, tmp1485, tmp1498
	vpor	%ymm15, %ymm14, %ymm14	# tmp1500, tmp1498, tmp1493
	vpaddd	%ymm14, %ymm6, %ymm14	# tmp1493, vect__147.831, vect_patt_385.833
	vpsrad	$6, %ymm14, %ymm14	#, vect_patt_385.833, vect_patt_386.834
	vpslld	$4, %ymm14, %ymm15	#, vect_patt_386.834, tmp1504
	vpsubd	%ymm14, %ymm15, %ymm14	# vect_patt_386.834, tmp1504, vect_patt_387.835
	vpslld	$3, %ymm14, %ymm14	#, vect_patt_387.835, tmp1506
	vpsubd	%ymm14, %ymm6, %ymm6	# tmp1506, vect__147.831, vect_patt_388.836
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vcvtdq2pd	%xmm6, %ymm14	# vect_patt_388.836, vect__149.837
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_388.836, tmp1511
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vdivpd	%ymm0, %ymm14, %ymm14	# tmp1741, vect__149.837, vect__156.838
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vcvtdq2pd	%xmm6, %ymm6	# tmp1511, vect__149.837
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1741, vect__149.837, vect__156.838
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vmovupd	%ymm14, -192(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703 + 768B]
	vmovupd	%ymm6, -160(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703 + 800B]
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vpmulld	.LC45(%rip), %ymm4, %ymm6	#, tmp1123, vect__147.831
	vpsrlq	$32, %ymm6, %ymm15	#, vect__147.831, tmp1519
	vpmuldq	%ymm1, %ymm6, %ymm14	# tmp1740, vect__147.831, tmp1515
	vpmuldq	%ymm1, %ymm15, %ymm15	# tmp1740, tmp1519, tmp1517
	vpmulld	.LC46(%rip), %ymm4, %ymm4	#, tmp1123, vect__147.831
	vpshufb	%ymm2, %ymm14, %ymm14	# tmp1781, tmp1515, tmp1528
	vpshufb	%ymm3, %ymm15, %ymm15	# tmp1782, tmp1517, tmp1530
	vpor	%ymm15, %ymm14, %ymm14	# tmp1530, tmp1528, tmp1523
	vpaddd	%ymm14, %ymm6, %ymm14	# tmp1523, vect__147.831, vect_patt_385.833
	vpsrad	$6, %ymm14, %ymm14	#, vect_patt_385.833, vect_patt_386.834
	vpslld	$4, %ymm14, %ymm15	#, vect_patt_386.834, tmp1534
	vpsubd	%ymm14, %ymm15, %ymm14	# vect_patt_386.834, tmp1534, vect_patt_387.835
	vpslld	$3, %ymm14, %ymm14	#, vect_patt_387.835, tmp1536
	vpsubd	%ymm14, %ymm6, %ymm6	# tmp1536, vect__147.831, vect_patt_388.836
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vcvtdq2pd	%xmm6, %ymm14	# vect_patt_388.836, vect__149.837
	vextracti128	$0x1, %ymm6, %xmm6	# vect_patt_388.836, tmp1541
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vdivpd	%ymm0, %ymm14, %ymm14	# tmp1741, vect__149.837, vect__156.838
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vcvtdq2pd	%xmm6, %ymm6	# tmp1541, vect__149.837
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1741, vect__149.837, vect__156.838
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vmovupd	%ymm14, -128(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703 + 832B]
	vpsrlq	$32, %ymm4, %ymm14	#, vect__147.831, tmp1549
	vpmuldq	%ymm1, %ymm14, %ymm14	# tmp1740, tmp1549, tmp1547
	vpshufb	%ymm3, %ymm14, %ymm14	# tmp1782, tmp1547, tmp1560
	vmovupd	%ymm6, -96(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703 + 864B]
	vpmuldq	%ymm1, %ymm4, %ymm6	# tmp1740, vect__147.831, tmp1545
	vpshufb	%ymm2, %ymm6, %ymm6	# tmp1781, tmp1545, tmp1558
	vpor	%ymm14, %ymm6, %ymm6	# tmp1560, tmp1558, tmp1553
	vpaddd	%ymm6, %ymm4, %ymm6	# tmp1553, vect__147.831, vect_patt_385.833
	vpsrad	$6, %ymm6, %ymm6	#, vect_patt_385.833, vect_patt_386.834
	vpslld	$4, %ymm6, %ymm14	#, vect_patt_386.834, tmp1564
	vpsubd	%ymm6, %ymm14, %ymm6	# vect_patt_386.834, tmp1564, vect_patt_387.835
	vpslld	$3, %ymm6, %ymm6	#, vect_patt_387.835, tmp1566
	vpsubd	%ymm6, %ymm4, %ymm4	# tmp1566, vect__147.831, vect_patt_388.836
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vcvtdq2pd	%xmm4, %ymm6	# vect_patt_388.836, vect__149.837
	vextracti128	$0x1, %ymm4, %xmm4	# vect_patt_388.836, tmp1571
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vdivpd	%ymm0, %ymm6, %ymm6	# tmp1741, vect__149.837, vect__156.838
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vcvtdq2pd	%xmm4, %ymm4	# tmp1571, vect__149.837
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vdivpd	%ymm0, %ymm4, %ymm4	# tmp1741, vect__149.837, vect__156.838
# benchmark_2mm.c:76:             B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
	vmovupd	%ymm6, -64(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703 + 896B]
	vmovupd	%ymm4, -32(%rax)	# vect__156.838, MEM <vector(4) double> [(double *)_703 + 928B]
# benchmark_2mm.c:74:     for (int i = 0; i < nk; i++)
	cmpl	$140, %edx	#, i
	jne	.L362	#,
	movq	-16696(%rbp), %rax	# %sfp, C
	vmovdqa	.LC47(%rip), %ymm4	#, tmp1757
	vmovdqa	.LC48(%rip), %ymm10	#, tmp1758
	vmovdqa	.LC49(%rip), %ymm8	#, tmp1759
	vmovapd	.LC50(%rip), %ymm7	#, tmp1786
	movq	%rax, %rsi	# C, ivtmp.946
	leaq	1280(%rax), %rdx	#, ivtmp.947
# benchmark_2mm.c:78:     for (int i = 0; i < nj; i++)
	xorl	%ecx, %ecx	# i
.L366:
	vmovd	%ecx, %xmm9	# i, vect_cst__351
	vpbroadcastd	%xmm9, %ymm9	# vect_cst__351, vect_cst__351
# benchmark_2mm.c:74:     for (int i = 0; i < nk; i++)
	movq	%rsi, %rax	# ivtmp.946, ivtmp.935
	vmovdqa	%ymm11, %ymm6	# vect_vec_iv_.805, vect_vec_iv_.817
	.p2align 4,,10
	.p2align 3
.L365:
	vmovdqa	%ymm6, %ymm0	# vect_vec_iv_.817, vect_vec_iv_.817
# benchmark_2mm.c:80:             C[i*nl + j] = (double) ((i*(j+3)+1) % nl) / nl;
	vpaddd	%ymm10, %ymm0, %ymm0	# tmp1758, vect_vec_iv_.817, vect__159.818
# benchmark_2mm.c:80:             C[i*nl + j] = (double) ((i*(j+3)+1) % nl) / nl;
	vpmulld	%ymm9, %ymm0, %ymm0	# vect_cst__351, vect__159.818, vect__161.819
	addq	$64, %rax	#, ivtmp.935
	vpaddd	%ymm4, %ymm6, %ymm6	# tmp1757, vect_vec_iv_.817, vect_vec_iv_.817
# benchmark_2mm.c:80:             C[i*nl + j] = (double) ((i*(j+3)+1) % nl) / nl;
	vpaddd	%ymm5, %ymm0, %ymm0	# tmp1778, vect__161.819, vect__162.820
	vpsrlq	$32, %ymm0, %ymm12	#, vect__162.820, tmp1583
	vpmuldq	%ymm8, %ymm0, %ymm1	# tmp1759, vect__162.820, tmp1579
	vpmuldq	%ymm8, %ymm12, %ymm12	# tmp1759, tmp1583, tmp1581
	vpshufb	%ymm2, %ymm1, %ymm1	# tmp1781, tmp1579, tmp1592
	vpshufb	%ymm3, %ymm12, %ymm12	# tmp1782, tmp1581, tmp1594
	vpor	%ymm12, %ymm1, %ymm1	# tmp1594, tmp1592, tmp1587
	vpsrad	$6, %ymm1, %ymm1	#, tmp1587, vect_patt_12.822
	vpslld	$2, %ymm1, %ymm12	#, vect_patt_12.822, tmp1597
	vpaddd	%ymm1, %ymm12, %ymm1	# vect_patt_12.822, tmp1597, vect_patt_7.823
	vpslld	$5, %ymm1, %ymm1	#, vect_patt_7.823, tmp1599
	vpsubd	%ymm1, %ymm0, %ymm0	# tmp1599, vect__162.820, vect_patt_339.824
# benchmark_2mm.c:80:             C[i*nl + j] = (double) ((i*(j+3)+1) % nl) / nl;
	vcvtdq2pd	%xmm0, %ymm1	# vect_patt_339.824, vect__164.825
	vextracti128	$0x1, %ymm0, %xmm0	# vect_patt_339.824, tmp1604
# benchmark_2mm.c:80:             C[i*nl + j] = (double) ((i*(j+3)+1) % nl) / nl;
	vdivpd	%ymm7, %ymm1, %ymm1	# tmp1786, vect__164.825, vect__171.826
# benchmark_2mm.c:80:             C[i*nl + j] = (double) ((i*(j+3)+1) % nl) / nl;
	vcvtdq2pd	%xmm0, %ymm0	# tmp1604, vect__164.825
# benchmark_2mm.c:80:             C[i*nl + j] = (double) ((i*(j+3)+1) % nl) / nl;
	vdivpd	%ymm7, %ymm0, %ymm0	# tmp1786, vect__164.825, vect__171.826
# benchmark_2mm.c:80:             C[i*nl + j] = (double) ((i*(j+3)+1) % nl) / nl;
	vmovupd	%ymm1, -64(%rax)	# vect__171.826, MEM <vector(4) double> [(double *)_644]
	vmovupd	%ymm0, -32(%rax)	# vect__171.826, MEM <vector(4) double> [(double *)_644 + 32B]
	cmpq	%rdx, %rax	# ivtmp.947, ivtmp.935
	jne	.L365	#,
# benchmark_2mm.c:78:     for (int i = 0; i < nj; i++)
	incl	%ecx	# i
# benchmark_2mm.c:78:     for (int i = 0; i < nj; i++)
	addq	$1280, %rsi	#, ivtmp.946
	leaq	1280(%rax), %rdx	#, ivtmp.947
	cmpl	$120, %ecx	#, i
	jne	.L366	#,
	vmovdqa	.LC51(%rip), %ymm9	#, tmp1787
	vmovdqa	.LC52(%rip), %ymm7	#, tmp1788
	vmovapd	.LC53(%rip), %ymm6	#, tmp1789
	movq	%r15, %rsi	# D_ref, ivtmp.930
	leaq	1280(%r15), %rdx	#, ivtmp.931
# benchmark_2mm.c:82:     for (int i = 0; i < ni; i++)
	xorl	%ecx, %ecx	# i
.L367:
	vmovd	%ecx, %xmm8	# i, vect_cst__193
	vpbroadcastd	%xmm8, %ymm8	# vect_cst__193, vect_cst__193
# benchmark_2mm.c:78:     for (int i = 0; i < nj; i++)
	movq	%rsi, %rax	# ivtmp.930, ivtmp.922
	vmovdqa	%ymm11, %ymm5	# vect_vec_iv_.805, vect_vec_iv_.805
	.p2align 4,,10
	.p2align 3
.L368:
	vmovdqa	%ymm5, %ymm0	# vect_vec_iv_.805, vect_vec_iv_.805
# benchmark_2mm.c:84:             D[i*nl + j] = (double) (i*(j+2) % nk) / nk;
	vpaddd	%ymm9, %ymm0, %ymm0	# tmp1787, vect_vec_iv_.805, vect__175.806
# benchmark_2mm.c:84:             D[i*nl + j] = (double) (i*(j+2) % nk) / nk;
	vpmulld	%ymm8, %ymm0, %ymm0	# vect_cst__193, vect__175.806, vect__177.807
	addq	$64, %rax	#, ivtmp.922
	vpaddd	%ymm4, %ymm5, %ymm5	# tmp1757, vect_vec_iv_.805, vect_vec_iv_.805
	vpsrlq	$32, %ymm0, %ymm10	#, vect__177.807, tmp1615
	vpmuldq	%ymm7, %ymm0, %ymm1	# tmp1788, vect__177.807, tmp1611
	vpmuldq	%ymm7, %ymm10, %ymm10	# tmp1788, tmp1615, tmp1613
	vpshufb	%ymm2, %ymm1, %ymm1	# tmp1781, tmp1611, tmp1624
	vpshufb	%ymm3, %ymm10, %ymm10	# tmp1782, tmp1613, tmp1626
	vpor	%ymm10, %ymm1, %ymm1	# tmp1626, tmp1624, tmp1619
	vpaddd	%ymm1, %ymm0, %ymm1	# tmp1619, vect__177.807, vect_patt_217.809
	vpsrad	$7, %ymm1, %ymm1	#, vect_patt_217.809, vect_patt_214.810
	vpslld	$3, %ymm1, %ymm10	#, vect_patt_214.810, tmp1630
	vpaddd	%ymm1, %ymm10, %ymm10	# vect_patt_214.810, tmp1630, tmp1631
	vpslld	$2, %ymm10, %ymm10	#, tmp1631, tmp1632
	vpsubd	%ymm1, %ymm10, %ymm1	# vect_patt_214.810, tmp1632, vect_patt_213.811
	vpslld	$2, %ymm1, %ymm1	#, vect_patt_213.811, tmp1634
	vpsubd	%ymm1, %ymm0, %ymm0	# tmp1634, vect__177.807, vect_patt_212.812
# benchmark_2mm.c:84:             D[i*nl + j] = (double) (i*(j+2) % nk) / nk;
	vcvtdq2pd	%xmm0, %ymm1	# vect_patt_212.812, vect__179.813
	vextracti128	$0x1, %ymm0, %xmm0	# vect_patt_212.812, tmp1639
# benchmark_2mm.c:84:             D[i*nl + j] = (double) (i*(j+2) % nk) / nk;
	vdivpd	%ymm6, %ymm1, %ymm1	# tmp1789, vect__179.813, vect__186.814
# benchmark_2mm.c:84:             D[i*nl + j] = (double) (i*(j+2) % nk) / nk;
	vcvtdq2pd	%xmm0, %ymm0	# tmp1639, vect__179.813
# benchmark_2mm.c:84:             D[i*nl + j] = (double) (i*(j+2) % nk) / nk;
	vdivpd	%ymm6, %ymm0, %ymm0	# tmp1789, vect__179.813, vect__186.814
# benchmark_2mm.c:84:             D[i*nl + j] = (double) (i*(j+2) % nk) / nk;
	vmovupd	%ymm1, -64(%rax)	# vect__186.814, MEM <vector(4) double> [(double *)_642]
	vmovupd	%ymm0, -32(%rax)	# vect__186.814, MEM <vector(4) double> [(double *)_642 + 32B]
	cmpq	%rax, %rdx	# ivtmp.922, ivtmp.931
	jne	.L368	#,
# benchmark_2mm.c:82:     for (int i = 0; i < ni; i++)
	incl	%ecx	# i
# benchmark_2mm.c:82:     for (int i = 0; i < ni; i++)
	addq	$1280, %rsi	#, ivtmp.930
	addq	$1280, %rdx	#, ivtmp.931
	cmpl	$100, %ecx	#, i
	jne	.L367	#,
# /usr/include/x86_64-linux-gnu/bits/string_fortified.h:79:   return __builtin___strcpy_chk (__dest, __src, __glibc_objsize (__dest));
	vmovdqa	.LC91(%rip), %xmm0	#, tmp1794
	movl	$29548, %eax	#,
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:112:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	leaq	.LC54(%rip), %rdi	#, tmp1646
# /usr/include/x86_64-linux-gnu/bits/string_fortified.h:79:   return __builtin___strcpy_chk (__dest, __src, __glibc_objsize (__dest));
	movl	$5066034, -16480(%rbp)	#, MEM <char[1:4]> [(void *)&config]
	movl	$1701737061, -16400(%rbp)	#, MEM <char[1:23]> [(void *)&config + 64B]
	movw	%ax, -16396(%rbp)	#, MEM <char[1:23]> [(void *)&config + 64B]
	movb	$0, -16394(%rbp)	#, MEM <char[1:23]> [(void *)&config + 64B]
# benchmark_2mm.c:365:     config.num_strategies = 6;
	movl	$6, -16352(%rbp)	#, config.num_strategies
# /usr/include/x86_64-linux-gnu/bits/string_fortified.h:79:   return __builtin___strcpy_chk (__dest, __src, __glibc_objsize (__dest));
	vmovdqa	%xmm0, -16416(%rbp)	# tmp1794, MEM <char[1:23]> [(void *)&config + 64B]
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:112:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	vzeroupper
	call	puts@PLT	#
# benchmark_metrics.h:131:     volatile double dummy = 0.0;
	movq	$0x000000000, -16664(%rbp)	#, dummy
# benchmark_metrics.h:132:     double start = omp_get_wtime();
	call	omp_get_wtime@PLT	#
	vmovsd	%xmm0, -16680(%rbp)	# tmp1801, %sfp
.L370:
# benchmark_metrics.h:133:     while (omp_get_wtime() - start < 0.1) {
	call	omp_get_wtime@PLT	#
# benchmark_metrics.h:133:     while (omp_get_wtime() - start < 0.1) {
	vsubsd	-16680(%rbp), %xmm0, %xmm0	# %sfp, tmp1802, tmp1650
# benchmark_metrics.h:133:     while (omp_get_wtime() - start < 0.1) {
	vmovsd	.LC56(%rip), %xmm3	#, tmp1875
	movq	.LC55(%rip), %rax	#, tmp1960
	vcomisd	%xmm0, %xmm3	# tmp1650, tmp1875
	vmovq	%rax, %xmm2	# tmp1960, tmp1793
	vxorps	%xmm6, %xmm6, %xmm6	# tmp1807
	jbe	.L401	#,
# benchmark_metrics.h:134:         for (int i = 0; i < 1000000; i++) {
	xorl	%eax, %eax	# i
	.p2align 4,,10
	.p2align 3
.L371:
# benchmark_metrics.h:135:             dummy += i * 0.0001;
	vcvtsi2sdl	%eax, %xmm6, %xmm0	# i, tmp1807, tmp1808
# benchmark_metrics.h:135:             dummy += i * 0.0001;
	vmovsd	-16664(%rbp), %xmm1	# dummy, dummy.93_124
# benchmark_metrics.h:134:         for (int i = 0; i < 1000000; i++) {
	incl	%eax	# i
# benchmark_metrics.h:135:             dummy += i * 0.0001;
	vfmadd132sd	%xmm2, %xmm1, %xmm0	# tmp1793, dummy.93_124, _125
	vmovsd	%xmm0, -16664(%rbp)	# _125, dummy
# benchmark_metrics.h:134:         for (int i = 0; i < 1000000; i++) {
	cmpl	$1000000, %eax	#, i
	jne	.L371	#,
	jmp	.L370	#
.L401:
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:112:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	leaq	.LC57(%rip), %rdi	#, tmp1652
	call	puts@PLT	#
	movl	$160, %r9d	#,
	movl	$140, %r8d	#,
	movl	$120, %ecx	#,
	movl	$100, %edx	#,
	leaq	.LC58(%rip), %rsi	#, tmp1653
	movl	$1, %edi	#,
	xorl	%eax, %eax	#
	call	__printf_chk@PLT	#
	movl	$7200000, %edx	#,
	leaq	.LC59(%rip), %rsi	#, tmp1654
	movl	$1, %edi	#,
	xorl	%eax, %eax	#
	call	__printf_chk@PLT	#
	vmovsd	.LC60(%rip), %xmm0	#,
	leaq	.LC61(%rip), %rsi	#, tmp1656
	movl	$1, %edi	#,
	movl	$1, %eax	#,
	call	__printf_chk@PLT	#
# /usr/include/x86_64-linux-gnu/bits/string_fortified.h:29:   return __builtin___memcpy_chk (__dest, __src, __len,
	movl	$128000, %edx	#,
	movq	%r15, %rsi	# D_ref,
	movq	%r12, %rdi	# D,
	call	memcpy@PLT	#
# benchmark_2mm.c:383:     double start = omp_get_wtime();
	call	omp_get_wtime@PLT	#
# benchmark_2mm.c:384:     kernel_2mm_sequential(NI, NJ, NK, NL, ALPHA, BETA, A, B, C, D, tmp);
	subq	$8, %rsp	#,
	pushq	-16704(%rbp)	# %sfp
	movq	.LC62(%rip), %rax	#, tmp1876
	movq	%r13, %r9	# B,
	pushq	%r12	# D
	vmovq	%rax, %xmm1	# tmp1876,
	movq	.LC63(%rip), %rax	#, tmp1877
	pushq	-16696(%rbp)	# %sfp
	movq	%r14, %r8	# A,
	movl	$160, %ecx	#,
	movl	$140, %edx	#,
	movl	$120, %esi	#,
	movl	$100, %edi	#,
# benchmark_2mm.c:383:     double start = omp_get_wtime();
	vmovq	%xmm0, %rbx	# tmp1803, start
# benchmark_2mm.c:384:     kernel_2mm_sequential(NI, NJ, NK, NL, ALPHA, BETA, A, B, C, D, tmp);
	vmovq	%rax, %xmm0	# tmp1877,
	call	kernel_2mm_sequential	#
# benchmark_2mm.c:385:     double serial_time = omp_get_wtime() - start;
	addq	$32, %rsp	#,
	call	omp_get_wtime@PLT	#
# benchmark_2mm.c:385:     double serial_time = omp_get_wtime() - start;
	vmovq	%rbx, %xmm3	# start, start
	vsubsd	%xmm3, %xmm0, %xmm0	# start, tmp1804, serial_time
# benchmark_2mm.c:387:            serial_time, total_flops / (serial_time * 1e9));
	vmovsd	.LC64(%rip), %xmm3	#, tmp1880
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:112:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	leaq	.LC66(%rip), %rsi	#, tmp1668
# benchmark_2mm.c:387:            serial_time, total_flops / (serial_time * 1e9));
	vmulsd	%xmm0, %xmm3, %xmm1	# serial_time, tmp1880, tmp1664
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:112:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	vmovsd	.LC65(%rip), %xmm3	#, tmp1882
	movl	$1, %edi	#,
	movl	$2, %eax	#,
# benchmark_2mm.c:385:     double serial_time = omp_get_wtime() - start;
	vmovsd	%xmm0, -16744(%rbp)	# serial_time, %sfp
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:112:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	vdivsd	%xmm1, %xmm3, %xmm1	# tmp1664, tmp1882,
	call	__printf_chk@PLT	#
	leaq	.LC73(%rip), %rax	#, tmp1883
	pushq	%rax	# tmp1883
	leaq	.LC74(%rip), %rax	#, tmp1884
# benchmark_2mm.c:390:     int thread_counts[] = {2, 4, 8, 16};
	vmovdqa	.LC67(%rip), %xmm0	#, tmp1669
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:112:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	pushq	%rax	# tmp1884
	leaq	.LC68(%rip), %r9	#,
	leaq	.LC69(%rip), %r8	#,
	leaq	.LC70(%rip), %rcx	#, tmp1672
	leaq	.LC71(%rip), %rdx	#, tmp1673
	leaq	.LC72(%rip), %rsi	#, tmp1674
	movl	$1, %edi	#,
	xorl	%eax, %eax	#
# benchmark_2mm.c:390:     int thread_counts[] = {2, 4, 8, 16};
	vmovdqa	%xmm0, -16656(%rbp)	# tmp1669, MEM <vector(4) int> [(int *)&thread_counts]
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:112:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	call	__printf_chk@PLT	#
	leaq	.LC78(%rip), %rax	#, tmp1885
	pushq	%rax	# tmp1885
	leaq	.LC79(%rip), %rax	#, tmp1886
	leaq	.LC75(%rip), %r9	#,
	pushq	%rax	# tmp1886
	leaq	.LC76(%rip), %r8	#,
	movq	%r9, %rcx	#, tmp1677
	movq	%r8, %rdx	#, tmp1678
	leaq	.LC77(%rip), %rsi	#, tmp1679
	movl	$1, %edi	#,
	xorl	%eax, %eax	#
	call	__printf_chk@PLT	#
# benchmark_2mm.c:405:     } strategies[] = {
	leaq	.LC80(%rip), %rax	#, tmp1887
	movq	%rax, -16640(%rbp)	# tmp1887, strategies[0].name
	leaq	kernel_2mm_basic_parallel(%rip), %rax	#, tmp1888
	movq	%rax, -16632(%rbp)	# tmp1888, strategies[0].func
	leaq	.LC81(%rip), %rax	#, tmp1889
	movq	%rax, -16624(%rbp)	# tmp1889, strategies[1].name
	leaq	kernel_2mm_collapsed(%rip), %rax	#, tmp1890
	movq	%rax, -16616(%rbp)	# tmp1890, strategies[1].func
	leaq	.LC82(%rip), %rax	#, tmp1891
	movq	%rax, -16608(%rbp)	# tmp1891, strategies[2].name
	leaq	kernel_2mm_tiled(%rip), %rax	#, tmp1892
	movq	%rax, -16600(%rbp)	# tmp1892, strategies[2].func
	leaq	.LC83(%rip), %rax	#, tmp1893
	movq	%rax, -16592(%rbp)	# tmp1893, strategies[3].name
	leaq	kernel_2mm_simd(%rip), %rax	#, tmp1894
	movq	%rax, -16584(%rbp)	# tmp1894, strategies[3].func
	leaq	.LC84(%rip), %rax	#, tmp1895
	movq	%rax, -16576(%rbp)	# tmp1895, strategies[4].name
	leaq	kernel_2mm_tasks(%rip), %rax	#, tmp1896
	movq	%rax, -16568(%rbp)	# tmp1896, strategies[4].func
	leaq	-16640(%rbp), %rax	#, _862
	movq	%rax, -16752(%rbp)	# _862, %sfp
	movq	%rax, -16688(%rbp)	# _862, %sfp
	leaq	-16560(%rbp), %rax	#, ivtmp.894
	movq	%rax, -16736(%rbp)	# ivtmp.894, %sfp
	leaq	-16480(%rbp), %rax	#, _450
	movq	%rax, -16712(%rbp)	# _450, %sfp
	leaq	-16656(%rbp), %rax	#, ivtmp.901
	movq	%rax, -16760(%rbp)	# ivtmp.901, %sfp
	addq	$32, %rsp	#,
.L372:
	movq	-16760(%rbp), %rax	# %sfp, ivtmp.901
	movq	%rax, -16720(%rbp)	# ivtmp.901, %sfp
.L383:
# benchmark_2mm.c:416:             int num_threads = thread_counts[t];
	movq	-16720(%rbp), %rax	# %sfp, ivtmp.901
	movq	%r14, %rbx	# A, A
	movl	(%rax), %eax	# MEM[(int *)_902], num_threads
# benchmark_2mm.c:417:             omp_set_num_threads(num_threads);
	movl	%eax, %edi	# num_threads,
# benchmark_2mm.c:416:             int num_threads = thread_counts[t];
	movl	%eax, -16724(%rbp)	# num_threads, %sfp
# benchmark_2mm.c:417:             omp_set_num_threads(num_threads);
	call	omp_set_num_threads@PLT	#
# /usr/include/x86_64-linux-gnu/bits/string_fortified.h:29:   return __builtin___memcpy_chk (__dest, __src, __len,
	movl	$128000, %edx	#,
	movq	%r15, %rsi	# D_ref,
	movq	%r12, %rdi	# D,
	call	memcpy@PLT	#
	movq	-16736(%rbp), %rax	# %sfp, ivtmp.894
	movq	%rax, %r14	# ivtmp.894, ivtmp.894
	.p2align 4,,10
	.p2align 3
.L373:
	movl	$128000, %edx	#,
	movq	%r15, %rsi	# D_ref,
	movq	%r12, %rdi	# D,
	call	memcpy@PLT	#
# benchmark_2mm.c:426:                 start = omp_get_wtime();
	call	omp_get_wtime@PLT	#
# benchmark_2mm.c:427:                 strategies[s].func(NI, NJ, NK, NL, ALPHA, BETA, A, B, C, D, tmp);
	subq	$8, %rsp	#,
	pushq	-16704(%rbp)	# %sfp
	movq	.LC62(%rip), %rax	#, tmp1904
# benchmark_2mm.c:426:                 start = omp_get_wtime();
	vmovsd	%xmm0, -16680(%rbp)	# tmp1805, %sfp
# benchmark_2mm.c:427:                 strategies[s].func(NI, NJ, NK, NL, ALPHA, BETA, A, B, C, D, tmp);
	pushq	%r12	# D
	vmovq	%rax, %xmm1	# tmp1904,
	movq	.LC63(%rip), %rax	#, tmp1905
	pushq	-16696(%rbp)	# %sfp
	vmovq	%rax, %xmm0	# tmp1905,
	movq	-16688(%rbp), %rax	# %sfp, ivtmp.908
	movq	%r13, %r9	# B,
	movq	%rbx, %r8	# A,
	movl	$160, %ecx	#,
	movl	$140, %edx	#,
	movl	$120, %esi	#,
	movl	$100, %edi	#,
	call	*8(%rax)	# MEM[(void (*<T1d2a>) (int, int, int, int, double, double, double *, double *, double *, double *, double *) *)_722 + 8B]
# benchmark_2mm.c:428:                 times[iter] = omp_get_wtime() - start;
	addq	$32, %rsp	#,
	call	omp_get_wtime@PLT	#
# benchmark_2mm.c:428:                 times[iter] = omp_get_wtime() - start;
	vsubsd	-16680(%rbp), %xmm0, %xmm0	# %sfp, tmp1806, tmp1703
# benchmark_2mm.c:424:             for (int iter = 0; iter < MEASUREMENT_ITERATIONS; iter++) {
	addq	$8, %r14	#, ivtmp.894
	vmovq	.LC86(%rip), %xmm5	#, tmp1773
# benchmark_2mm.c:428:                 times[iter] = omp_get_wtime() - start;
	vmovsd	%xmm0, -8(%r14)	# tmp1703, MEM[(double *)_42]
# benchmark_2mm.c:424:             for (int iter = 0; iter < MEASUREMENT_ITERATIONS; iter++) {
	cmpq	-16712(%rbp), %r14	# %sfp, ivtmp.894
	vxorps	%xmm6, %xmm6, %xmm6	# tmp1807
	jne	.L373	#,
# benchmark_2mm.c:434:                 avg_time += times[i];
	vxorpd	%xmm3, %xmm3, %xmm3	# tmp1909
	vaddsd	-16560(%rbp), %xmm3, %xmm7	# times[0], tmp1909, avg_time
	movq	%rbx, %r14	# A, A
# benchmark_2mm.c:436:             avg_time /= MEASUREMENT_ITERATIONS;
	movl	$1280, %edx	#, ivtmp.885
# benchmark_2mm.c:434:                 avg_time += times[i];
	vaddsd	-16552(%rbp), %xmm7, %xmm7	# times[1], avg_time, avg_time
# benchmark_2mm.c:89:     double max_error = 0.0;
	vxorpd	%xmm4, %xmm4, %xmm4	# max_error
# benchmark_2mm.c:434:                 avg_time += times[i];
	vaddsd	-16544(%rbp), %xmm7, %xmm7	# times[2], avg_time, avg_time
	vaddsd	-16536(%rbp), %xmm7, %xmm7	# times[3], avg_time, avg_time
	vaddsd	-16528(%rbp), %xmm7, %xmm7	# times[4], avg_time, avg_time
	vaddsd	-16520(%rbp), %xmm7, %xmm7	# times[5], avg_time, avg_time
	vaddsd	-16512(%rbp), %xmm7, %xmm7	# times[6], avg_time, avg_time
	vaddsd	-16504(%rbp), %xmm7, %xmm7	# times[7], avg_time, avg_time
	vaddsd	-16496(%rbp), %xmm7, %xmm7	# times[8], avg_time, avg_time
	vaddsd	-16488(%rbp), %xmm7, %xmm7	# times[9], avg_time, avg_time
# benchmark_2mm.c:436:             avg_time /= MEASUREMENT_ITERATIONS;
	vdivsd	.LC85(%rip), %xmm7, %xmm7	#, avg_time, avg_time
	.p2align 4,,10
	.p2align 3
.L377:
	leaq	-1280(%rdx), %rax	#, ivtmp.878
	.p2align 4,,10
	.p2align 3
.L376:
# benchmark_2mm.c:92:             double error = fabs(D_ref[i*nl + j] - D_test[i*nl + j]);
	vmovsd	(%r15,%rax), %xmm0	# MEM[(double *)D_ref_29 + ivtmp.878_295 * 1], MEM[(double *)D_ref_29 + ivtmp.878_295 * 1]
	vsubsd	(%r12,%rax), %xmm0, %xmm0	# MEM[(double *)D_31 + ivtmp.878_295 * 1], MEM[(double *)D_ref_29 + ivtmp.878_295 * 1], tmp1707
# benchmark_2mm.c:91:         for (int j = 0; j < nl; j++) {
	addq	$8, %rax	#, ivtmp.878
# benchmark_2mm.c:92:             double error = fabs(D_ref[i*nl + j] - D_test[i*nl + j]);
	vandpd	%xmm5, %xmm0, %xmm0	# tmp1773, tmp1707, error
	vmaxsd	%xmm4, %xmm0, %xmm4	# max_error, error, max_error
# benchmark_2mm.c:91:         for (int j = 0; j < nl; j++) {
	cmpq	%rdx, %rax	# ivtmp.885, ivtmp.878
	jne	.L376	#,
# benchmark_2mm.c:90:     for (int i = 0; i < ni; i++) {
	leaq	1280(%rax), %rdx	#, ivtmp.885
	cmpq	$128000, %rax	#, ivtmp.878
	jne	.L377	#,
# benchmark_2mm.c:442:             double speedup = serial_time / avg_time;
	vmovsd	-16744(%rbp), %xmm3	# %sfp, serial_time
# benchmark_2mm.c:443:             double efficiency = speedup / num_threads * 100.0;
	movl	-16724(%rbp), %ecx	# %sfp, num_threads
# benchmark_2mm.c:442:             double speedup = serial_time / avg_time;
	vdivsd	%xmm7, %xmm3, %xmm1	# avg_time, serial_time, speedup
# benchmark_2mm.c:443:             double efficiency = speedup / num_threads * 100.0;
	vcvtsi2sdl	%ecx, %xmm6, %xmm2	# num_threads, tmp1807, tmp1809
# benchmark_2mm.c:444:             double gflops = total_flops / (avg_time * 1e9);
	vmulsd	.LC64(%rip), %xmm7, %xmm3	#, avg_time, tmp1710
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:112:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	vmovsd	.LC65(%rip), %xmm5	#, tmp1912
	movq	-16688(%rbp), %rax	# %sfp, ivtmp.908
	vmovsd	%xmm7, %xmm7, %xmm0	# avg_time,
	movq	(%rax), %rdx	# MEM[(const char * *)_722],
	leaq	.LC88(%rip), %rsi	#,
	movl	$1, %edi	#,
	movl	$4, %eax	#,
# benchmark_2mm.c:442:             double speedup = serial_time / avg_time;
	vmovsd	%xmm4, -16680(%rbp)	# max_error, %sfp
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:112:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	vdivsd	%xmm3, %xmm5, %xmm3	# tmp1710, tmp1912,
# benchmark_2mm.c:443:             double efficiency = speedup / num_threads * 100.0;
	vdivsd	%xmm2, %xmm1, %xmm2	# tmp1714, speedup, tmp1715
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:112:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	vmulsd	.LC87(%rip), %xmm2, %xmm2	#, tmp1715,
	call	__printf_chk@PLT	#
# benchmark_2mm.c:449:             if (error > 1e-10) {
	vmovsd	-16680(%rbp), %xmm4	# %sfp, max_error
	vcomisd	.LC89(%rip), %xmm4	#, max_error
	ja	.L402	#,
.L399:
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:112:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	movl	$10, %edi	#,
	call	putchar@PLT	#
# benchmark_2mm.c:415:         for (int t = 0; t < num_thread_configs; t++) {
	addq	$4, -16720(%rbp)	#, %sfp
	movq	-16720(%rbp), %rax	# %sfp, ivtmp.901
	cmpq	-16752(%rbp), %rax	# %sfp, ivtmp.901
	jne	.L383	#,
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:112:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	movl	$10, %edi	#,
	call	putchar@PLT	#
# benchmark_2mm.c:414:     for (int s = 0; s < 5; s++) {
	addq	$16, -16688(%rbp)	#, %sfp
	movq	-16688(%rbp), %rax	# %sfp, ivtmp.908
	cmpq	%rax, -16736(%rbp)	# ivtmp.908, %sfp
	jne	.L372	#,
# benchmark_2mm.c:458:     free(A);
	movq	%r14, %rdi	# A,
	call	free@PLT	#
# benchmark_2mm.c:459:     free(B);
	movq	%r13, %rdi	# B,
	call	free@PLT	#
# benchmark_2mm.c:460:     free(C);
	movq	-16696(%rbp), %rdi	# %sfp,
	call	free@PLT	#
# benchmark_2mm.c:461:     free(D_ref);
	movq	%r15, %rdi	# D_ref,
	call	free@PLT	#
# benchmark_2mm.c:462:     free(D);
	movq	%r12, %rdi	# D,
	call	free@PLT	#
# benchmark_2mm.c:463:     free(tmp);
	movq	-16704(%rbp), %rdi	# %sfp,
	call	free@PLT	#
# benchmark_2mm.c:466: }
	movq	-56(%rbp), %rax	# D.38804, tmp1811
	subq	%fs:40, %rax	# MEM[(<address-space-1> long unsigned int *)40B], tmp1811
	jne	.L403	#,
	leaq	-48(%rbp), %rsp	#,
	popq	%rbx	#
	popq	%r10	#
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12	#
	popq	%r13	#
	popq	%r14	#
	popq	%r15	#
	xorl	%eax, %eax	#
	popq	%rbp	#
	leaq	-8(%r10), %rsp	#,
	.cfi_def_cfa 7, 8
	ret	
.L402:
	.cfi_restore_state
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:112:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	vmovsd	%xmm4, %xmm4, %xmm0	# max_error,
	leaq	.LC90(%rip), %rsi	#,
	movl	$1, %edi	#,
	movl	$1, %eax	#,
	call	__printf_chk@PLT	#
	jmp	.L399	#
.L403:
# benchmark_2mm.c:466: }
	call	__stack_chk_fail@PLT	#
	.cfi_endproc
.LFE5551:
	.size	main, .-main
	.section	.rodata.cst32,"aM",@progbits,32
	.align 32
.LC2:
	.long	0
	.long	1
	.long	2
	.long	3
	.long	4
	.long	5
	.long	6
	.long	7
	.align 32
.LC3:
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.align 32
.LC4:
	.long	1374389535
	.long	1374389535
	.long	1374389535
	.long	1374389535
	.long	1374389535
	.long	1374389535
	.long	1374389535
	.long	1374389535
	.align 32
.LC5:
	.byte	4
	.byte	5
	.byte	6
	.byte	7
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	12
	.byte	13
	.byte	14
	.byte	15
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	4
	.byte	5
	.byte	6
	.byte	7
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	12
	.byte	13
	.byte	14
	.byte	15
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.align 32
.LC6:
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	4
	.byte	5
	.byte	6
	.byte	7
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	12
	.byte	13
	.byte	14
	.byte	15
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	4
	.byte	5
	.byte	6
	.byte	7
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	12
	.byte	13
	.byte	14
	.byte	15
	.align 32
.LC7:
	.long	0
	.long	1079574528
	.long	0
	.long	1079574528
	.long	0
	.long	1079574528
	.long	0
	.long	1079574528
	.align 32
.LC8:
	.long	8
	.long	9
	.long	10
	.long	11
	.long	12
	.long	13
	.long	14
	.long	15
	.align 32
.LC9:
	.long	16
	.long	17
	.long	18
	.long	19
	.long	20
	.long	21
	.long	22
	.long	23
	.align 32
.LC10:
	.long	24
	.long	25
	.long	26
	.long	27
	.long	28
	.long	29
	.long	30
	.long	31
	.align 32
.LC11:
	.long	32
	.long	33
	.long	34
	.long	35
	.long	36
	.long	37
	.long	38
	.long	39
	.align 32
.LC12:
	.long	40
	.long	41
	.long	42
	.long	43
	.long	44
	.long	45
	.long	46
	.long	47
	.align 32
.LC13:
	.long	48
	.long	49
	.long	50
	.long	51
	.long	52
	.long	53
	.long	54
	.long	55
	.align 32
.LC14:
	.long	56
	.long	57
	.long	58
	.long	59
	.long	60
	.long	61
	.long	62
	.long	63
	.align 32
.LC15:
	.long	64
	.long	65
	.long	66
	.long	67
	.long	68
	.long	69
	.long	70
	.long	71
	.align 32
.LC16:
	.long	72
	.long	73
	.long	74
	.long	75
	.long	76
	.long	77
	.long	78
	.long	79
	.align 32
.LC17:
	.long	80
	.long	81
	.long	82
	.long	83
	.long	84
	.long	85
	.long	86
	.long	87
	.align 32
.LC18:
	.long	88
	.long	89
	.long	90
	.long	91
	.long	92
	.long	93
	.long	94
	.long	95
	.align 32
.LC19:
	.long	96
	.long	97
	.long	98
	.long	99
	.long	100
	.long	101
	.long	102
	.long	103
	.align 32
.LC20:
	.long	104
	.long	105
	.long	106
	.long	107
	.long	108
	.long	109
	.long	110
	.long	111
	.align 32
.LC21:
	.long	112
	.long	113
	.long	114
	.long	115
	.long	116
	.long	117
	.long	118
	.long	119
	.align 32
.LC22:
	.long	120
	.long	121
	.long	122
	.long	123
	.long	124
	.long	125
	.long	126
	.long	127
	.align 32
.LC23:
	.long	128
	.long	129
	.long	130
	.long	131
	.long	132
	.long	133
	.long	134
	.long	135
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC24:
	.long	136
	.long	137
	.long	138
	.long	139
	.set	.LC25,.LC3
	.set	.LC26,.LC4
	.set	.LC27,.LC5
	.set	.LC28,.LC6
	.set	.LC29,.LC7
	.section	.rodata.cst32
	.align 32
.LC30:
	.long	1
	.long	2
	.long	3
	.long	4
	.long	5
	.long	6
	.long	7
	.long	8
	.align 32
.LC31:
	.long	-2004318071
	.long	-2004318071
	.long	-2004318071
	.long	-2004318071
	.long	-2004318071
	.long	-2004318071
	.long	-2004318071
	.long	-2004318071
	.align 32
.LC32:
	.long	0
	.long	1079902208
	.long	0
	.long	1079902208
	.long	0
	.long	1079902208
	.long	0
	.long	1079902208
	.align 32
.LC33:
	.long	9
	.long	10
	.long	11
	.long	12
	.long	13
	.long	14
	.long	15
	.long	16
	.align 32
.LC34:
	.long	17
	.long	18
	.long	19
	.long	20
	.long	21
	.long	22
	.long	23
	.long	24
	.align 32
.LC35:
	.long	25
	.long	26
	.long	27
	.long	28
	.long	29
	.long	30
	.long	31
	.long	32
	.align 32
.LC36:
	.long	33
	.long	34
	.long	35
	.long	36
	.long	37
	.long	38
	.long	39
	.long	40
	.align 32
.LC37:
	.long	41
	.long	42
	.long	43
	.long	44
	.long	45
	.long	46
	.long	47
	.long	48
	.align 32
.LC38:
	.long	49
	.long	50
	.long	51
	.long	52
	.long	53
	.long	54
	.long	55
	.long	56
	.align 32
.LC39:
	.long	57
	.long	58
	.long	59
	.long	60
	.long	61
	.long	62
	.long	63
	.long	64
	.align 32
.LC40:
	.long	65
	.long	66
	.long	67
	.long	68
	.long	69
	.long	70
	.long	71
	.long	72
	.align 32
.LC41:
	.long	73
	.long	74
	.long	75
	.long	76
	.long	77
	.long	78
	.long	79
	.long	80
	.align 32
.LC42:
	.long	81
	.long	82
	.long	83
	.long	84
	.long	85
	.long	86
	.long	87
	.long	88
	.align 32
.LC43:
	.long	89
	.long	90
	.long	91
	.long	92
	.long	93
	.long	94
	.long	95
	.long	96
	.align 32
.LC44:
	.long	97
	.long	98
	.long	99
	.long	100
	.long	101
	.long	102
	.long	103
	.long	104
	.align 32
.LC45:
	.long	105
	.long	106
	.long	107
	.long	108
	.long	109
	.long	110
	.long	111
	.long	112
	.align 32
.LC46:
	.long	113
	.long	114
	.long	115
	.long	116
	.long	117
	.long	118
	.long	119
	.long	120
	.align 32
.LC47:
	.long	8
	.long	8
	.long	8
	.long	8
	.long	8
	.long	8
	.long	8
	.long	8
	.align 32
.LC48:
	.long	3
	.long	3
	.long	3
	.long	3
	.long	3
	.long	3
	.long	3
	.long	3
	.align 32
.LC49:
	.long	1717986919
	.long	1717986919
	.long	1717986919
	.long	1717986919
	.long	1717986919
	.long	1717986919
	.long	1717986919
	.long	1717986919
	.align 32
.LC50:
	.long	0
	.long	1080295424
	.long	0
	.long	1080295424
	.long	0
	.long	1080295424
	.long	0
	.long	1080295424
	.align 32
.LC51:
	.long	2
	.long	2
	.long	2
	.long	2
	.long	2
	.long	2
	.long	2
	.long	2
	.align 32
.LC52:
	.long	-368140053
	.long	-368140053
	.long	-368140053
	.long	-368140053
	.long	-368140053
	.long	-368140053
	.long	-368140053
	.long	-368140053
	.align 32
.LC53:
	.long	0
	.long	1080131584
	.long	0
	.long	1080131584
	.long	0
	.long	1080131584
	.long	0
	.long	1080131584
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC55:
	.long	-350469331
	.long	1058682594
	.align 8
.LC56:
	.long	-1717986918
	.long	1069128089
	.align 8
.LC60:
	.long	0
	.long	1072100096
	.align 8
.LC62:
	.long	858993459
	.long	1072902963
	.align 8
.LC63:
	.long	0
	.long	1073217536
	.align 8
.LC64:
	.long	0
	.long	1104006501
	.align 8
.LC65:
	.long	0
	.long	1096513344
	.section	.rodata.cst16
	.align 16
.LC67:
	.long	2
	.long	4
	.long	8
	.long	16
	.section	.rodata.cst8
	.align 8
.LC85:
	.long	0
	.long	1076101120
	.section	.rodata.cst16
	.align 16
.LC86:
	.long	-1
	.long	2147483647
	.long	0
	.long	0
	.set	.LC87,.LC7
	.section	.rodata.cst8
	.align 8
.LC89:
	.long	-640172613
	.long	1037794527
	.section	.rodata.cst16
	.align 16
.LC91:
	.quad	7002378758270118252
	.quad	7723499029868668780
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04.2) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
