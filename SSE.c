#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <float.h>
#include <immintrin.h>
#include <emmintrin.h>
#include <pthread.h>

#define MINSNPS_B 5
#define MAXSNPS_E 20
double gettime(void);
float randpval (void);

double gettime(void)
{
  struct timeval ttime;
  gettimeofday(&ttime , NULL);
  return ttime.tv_sec + ttime.tv_usec * 0.000001;
}

float randpval (void)
{
  int vr = rand();
  int vm = rand()%vr;
  float r = ((float)vm)/(float)vr;
  assert(r>=0.0f && r<=1.00001f);
  return r;
}

int main(int argc, char ** argv)
{
  assert(argc==2);
  double timeTotalMainStart = gettime();
  float avgF = 0.0f;
  float maxF = 0.0f;
  float minF = 100000;
  unsigned int N = (unsigned int)atoi(argv[1]);
  unsigned int iters = 1;
  srand(1);
  float * mVec = (float*)malloc(sizeof(float)*N);
  assert(mVec!=NULL);
  float * nVec = (float*)malloc(sizeof(float)*N);
  assert(nVec!=NULL);
  float * LVec = (float*)malloc(sizeof(float)*N);
  assert(LVec!=NULL);
  float * RVec = (float*)malloc(sizeof(float)*N);
  assert(RVec!=NULL);
  float * CVec = (float*)malloc(sizeof(float)*N);
  assert(CVec!=NULL);
  float * FVec = (float*)malloc(sizeof(float)*N);
  assert(FVec!=NULL);

  __m128 rVec_t;
  __m128 lVec_t;
  __m128 mVec_t;
  __m128 nVec_t;
  __m128 num;
  __m128 den;
  __m128 den_0;
  __m128 den_1;
  __m128 F_result;
  __m128 cond_1;
  __m128 v1;
  __m128 cond_2;
  __m128 v2;
  __m128 maxF_t;
  __m128 minF_t;
  const __m128 one= _mm_set1_ps(1.0f);
  const __m128 two= _mm_set1_ps(2.0f);
  const __m128 zerozeroone = _mm_set1_ps(0.01f);

  for(unsigned int i=0;i<N;i++)
  {
    mVec[i] =(float)(MINSNPS_B+rand()%MAXSNPS_E);
    nVec[i] =(float)(MINSNPS_B+rand()%MAXSNPS_E);
    LVec[i] =randpval()*mVec[i];
    RVec[i] =randpval()*nVec[i];
    CVec[i] =randpval()*mVec[i]*nVec[i];
    FVec[i] =0.0;
    assert(mVec[i]>=MINSNPS_B && mVec[i]<=(MINSNPS_B+MAXSNPS_E));
    assert(nVec[i]>=MINSNPS_B && nVec[i]<=(MINSNPS_B+MAXSNPS_E));
    assert(LVec[i]>=0.0f && LVec[i]<=1.0f*mVec[i]);
    assert(RVec[i]>=0.0f && RVec[i]<=1.0f*nVec[i]);
    assert(CVec[i]>=0.0f && CVec[i]<=1.0f*mVec[i]*nVec[i]);
  }
  double timeOmegaTotalStart = gettime();
  for(unsigned int j=0;j<iters;j++)
  {
    avgF = 0.0f;
    maxF = 0.0f;
    minF = 10000;
    for(unsigned int i=0;i<N-(N%4);i+=4)
    {
      rVec_t = _mm_load_ps(RVec+i);
      lVec_t = _mm_load_ps(LVec+i);
      mVec_t = _mm_load_ps(mVec+i);
      nVec_t = _mm_load_ps(nVec+i);
      num = _mm_add_ps(_mm_div_ps(_mm_mul_ps(mVec_t,_mm_sub_ps(mVec_t,one)),two),_mm_div_ps(_mm_mul_ps(nVec_t,_mm_sub_ps(nVec_t,one)),two));
      num = _mm_div_ps(_mm_add_ps(lVec_t,rVec_t),num);
      den = _mm_div_ps( _mm_sub_ps(_mm_sub_ps(_mm_load_ps(CVec+i),lVec_t),rVec_t),_mm_mul_ps(mVec_t,nVec_t));
      F_result = _mm_div_ps(num,_mm_add_ps(den,zerozeroone));
      _mm_store_ps(FVec+i,F_result);
      maxF_t = _mm_load_ps1(&maxF);
      minF_t = _mm_load_ps1(&minF);
      cond_1 = _mm_cmpgt_ps(F_result, maxF_t);
      v1 = _mm_sub_ps(F_result,maxF_t);
      v1 = _mm_and_ps(v1, cond_1);
      maxF_t = _mm_add_ps(maxF_t,v1);
      maxF_t = _mm_max_ps(maxF_t, _mm_shuffle_ps(maxF_t, maxF_t, _MM_SHUFFLE(2, 1, 0, 3)));
      maxF_t = _mm_max_ps(maxF_t, _mm_shuffle_ps(maxF_t, maxF_t, _MM_SHUFFLE(1, 0, 3, 2)));
      _mm_store_ss(&maxF,maxF_t);
      cond_2 = _mm_cmplt_ps(F_result, minF_t);
      v2 = _mm_sub_ps(minF_t,F_result);
      v2 = _mm_and_ps(v2, cond_2);
      minF_t = _mm_sub_ps(minF_t,v2);
      minF_t = _mm_min_ps(minF_t, _mm_shuffle_ps(minF_t, minF_t, _MM_SHUFFLE(2, 1, 0, 3)));
      minF_t = _mm_min_ps(minF_t, _mm_shuffle_ps(minF_t, minF_t, _MM_SHUFFLE(1, 0, 3, 2)));
      _mm_store_ss(&minF,minF_t);
      F_result = _mm_hadd_ps(F_result,F_result);
      F_result = _mm_hadd_ps(F_result,F_result);
      _mm_store_ss(&avgF, _mm_add_ps(_mm_load_ps1(&avgF),F_result));
    }
    for(unsigned int i = N-(N%4);i<N;i++){
      rVec_t = _mm_load_ss(RVec+i);
      lVec_t = _mm_load_ss(LVec+i);
      mVec_t = _mm_load_ss(mVec+i);
      nVec_t = _mm_load_ss(nVec+i);
      num = _mm_add_ss(_mm_div_ss(_mm_mul_ss(mVec_t,_mm_sub_ss(mVec_t,one)),two),_mm_div_ss(_mm_mul_ss(nVec_t,_mm_sub_ss(nVec_t,one)),two));
      num = _mm_div_ss(_mm_add_ss(lVec_t,rVec_t),num);
      den = _mm_div_ss( _mm_sub_ss(_mm_sub_ss(_mm_load_ss(CVec+i),lVec_t),rVec_t),_mm_mul_ss(mVec_t,nVec_t));
      F_result = _mm_div_ss(num,_mm_add_ss(den,zerozeroone));
      _mm_store_ss(FVec+i,F_result);
      maxF_t = _mm_load_ss(&maxF);
      minF_t = _mm_load_ss(&minF);
      cond_1 = _mm_cmpgt_ss(F_result, maxF_t);
      v1 = _mm_sub_ps(F_result,maxF_t);
      v1 = _mm_and_ps(v1, cond_1);
      maxF_t = _mm_add_ss(maxF_t,v1);
      _mm_store_ss(&maxF,maxF_t);
      cond_2 = _mm_cmplt_ss(F_result, minF_t);
      v2 = _mm_sub_ss(minF_t,F_result);
      v2 = _mm_and_ps(v2, cond_2);
      minF_t = _mm_sub_ss(minF_t,v2);
      _mm_store_ss(&minF,minF_t);
      _mm_store_ss(&avgF, _mm_add_ss(_mm_load_ss(&avgF),F_result));
    }
  }
  double timeOmegaTotal = gettime()-timeOmegaTotalStart;
  double timeTotalMainStop = gettime();
  printf("Omega time %fs - Total time %fs - Min %e - Max %e - Avg %e\n",timeOmegaTotal/iters, timeTotalMainStop-timeTotalMainStart, (double)minF, (double)maxF,(double)avgF/N);
  free(mVec);
  free(nVec);
  free(LVec);
  free(RVec);
  free(CVec);
  free(FVec);
}
