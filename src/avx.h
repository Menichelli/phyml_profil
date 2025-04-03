/*

PHYML :  a program that  computes maximum likelihood  phylogenies from
DNA or AA homologous sequences

Copyright (C) Stephane Guindon. Oct 2003 onward

All parts of  the source except where indicated  are distributed under
the GNU public licence.  See http://www.opensource.org for details.

*/

#include <config.h>

#include <immintrin.h>
#include <stdlib.h>


#ifndef AVX_H
#define AVX_H

#include "utilities.h"
#include "optimiz.h"
#include "models.h"
#include "free.h"
#include "times.h"
#include "mixt.h"

#include <immintrin.h>
#include <stdlib.h>
#include <stdint.h>


#if (defined(__AVX__) || defined(__AVX2__))
#define ALIGNED(ptr) __builtin_assume_aligned((ptr), 32)
#define IS_LIKELY(x) __builtin_expect((x), 1)
#define IS_UNLIKELY(x) __builtin_expect((x), 0)
#define UNROLL4 _Pragma("GCC unroll 4")
#define PREFETCH(ptr) __builtin_prefetch((ptr), 0, 3)


static inline double mm256_reduce_max_pd(__m256d v) {
    __m128d vlow = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1);
    vlow = _mm_max_pd(vlow, vhigh);
    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return _mm_cvtsd_f64(_mm_max_pd(vlow, high64));
}
void AVX_Update_Partial_Lk(t_tree *tree,t_edge *b_fcus,t_node *n);
void AVX_Update_Eigen_Lr(t_edge *b, t_tree *tree);
phydbl AVX_Lk_Core_One_Class_Eigen_Lr(const phydbl *dot_prod, const phydbl *expl, const unsigned int ns);
void AVX_Lk_dLk_Core_One_Class_Eigen_Lr(const phydbl *dot_prod, const phydbl *expl, const unsigned int ns, phydbl *lk, phydbl *dlk);
phydbl AVX_Lk_Core_One_Class_No_Eigen_Lr(const phydbl *p_lk_left, const phydbl *p_lk_rght, const phydbl *Pij, const phydbl *tPij, const phydbl *pi, const int ns, const int ambiguity_check, const int observed_state);  
phydbl AVX_Vect_Norm(__m256d _z);
phydbl AVX_Lk_Core(int state, int ambiguity_check, t_edge *b, t_tree *tree);
phydbl AVX_Lk_Core_Nucl(int state, int ambiguity_check, t_edge *b, t_tree *tree);
phydbl AVX_Lk_Core_AA(int state, int ambiguity_check, t_edge *b, t_tree *tree);
void AVX_Partial_Lk_Exex(const __m256d *_tPij1, const int state1, const __m256d *_tPij2, const int state2, const int ns, __m256d *plk0);
void AVX_Partial_Lk_Exin(const __m256d *_tPij1, const int state1, const __m256d *_tPij2, const phydbl *_plk2, __m256d *_pmat2plk2, const int ns, __m256d *_plk0);
void AVX_Partial_Lk_Inin(const __m256d *_tPij1, const phydbl *plk1, __m256d *_pmat1plk1, const __m256d *_tPij2, const phydbl *plk2, __m256d *_pmat2plk2, const int ns, __m256d *_plk0);
void AVX_Matrix_Vect_Prod(const __m256d *_m_transpose,  const phydbl *_v, const int ns, __m256d *res);
__m256d AVX_Horizontal_Add(const __m256d x[4]);
phydbl AVX_Lk_Core_One_Class_No_Eigen_Lr_Block(const phydbl *p_lk_left, const phydbl *p_lk_rght, const phydbl *Pij, const phydbl *tPij, const phydbl *pi, const int ns, const int ambiguity_check, const int observed_state);
phydbl AVX_Lk_Core_One_Class_No_Eigen_Lr_No_Block(const phydbl *p_lk_left, const phydbl *p_lk_rght, const phydbl *Pij, const phydbl *tPij, const phydbl *pi, const int ns, const int ambiguity_check, const int observed_state);


#endif
#endif
