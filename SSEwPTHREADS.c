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
float randpval(void);

double gettime(void) {
  struct timeval ttime;
  gettimeofday( & ttime, NULL);
  return ttime.tv_sec + ttime.tv_usec * 0.000001;
}

typedef struct THREAD_ARGUMENTS{
  int t_id;
  int from_index;
  int to_index;
  unsigned int N;
  float *mVec;
  float *nVec;
  float *RVec;
  float *LVec;
  float *CVec;
  float *FVec;
}t_args;

typedef struct THREAD_RESULTS{
  float avgF;
  float minF;
  float maxF;
}t_res;

float randpval(void) {
  int vr = rand();
  int vm = rand() % vr;
  float r = ((float) vm) / (float) vr;
  assert(r >= 0.0f && r <= 1.00001f);
  return r;
}

void process_workload(float* avgF,float* minF,float* maxF,int num_threads,pthread_t *thread_handles,t_args* main_args);
void *omega_calc(void * args);
t_args* spawn_threads(pthread_t *thread_handles,int num_threads,unsigned int N,float *mVec,float *nVec,float *RVec,float *LVec,float *CVec,float *FVec);

pthread_cond_t cond1 = PTHREAD_COND_INITIALIZER;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
int wait =0;



int main(int argc, char ** argv) {
  wait =0;
  t_args* main_args;
  double test;
  float avgF = 0.0f;
  float maxF = 0.0f;
  float minF = FLT_MAX;
  float *mVec;
  float *nVec;
  float *RVec;
  float *LVec;
  float *CVec;
  float *FVec;
  t_res* thread_res;
  unsigned int iters = 10;
  unsigned int N = (unsigned int) atoi(argv[1]);
  unsigned int num_threads = (unsigned int) atoi(argv[2]);
  assert(argc == 3);

  pthread_t thread_handles[num_threads];

  double timeTotalMainStart = gettime();
  srand(1);
  mVec = (float * ) malloc(sizeof(float) * N);
  assert(mVec != NULL);
  nVec = (float * ) malloc(sizeof(float) * N);
  assert(nVec != NULL);
  LVec = (float * ) malloc(sizeof(float) * N);
  assert(LVec != NULL);
  RVec = (float * ) malloc(sizeof(float) * N);
  assert(RVec != NULL);
  CVec = (float * ) malloc(sizeof(float) * N);
  assert(CVec != NULL);
  FVec = (float * ) malloc(sizeof(float) * N);
  assert(FVec != NULL);
  main_args = spawn_threads(thread_handles,num_threads,N,mVec,nVec,RVec,LVec,CVec,FVec);
  for (unsigned int i = 0; i < N; i++) {
    mVec[i] = (float)(MINSNPS_B + rand() % MAXSNPS_E);
    nVec[i] = (float)(MINSNPS_B + rand() % MAXSNPS_E);
    LVec[i] = randpval() * mVec[i];
    RVec[i] = randpval() * nVec[i];
    CVec[i] = randpval() * mVec[i] * nVec[i];
    FVec[i] = 0.0;
    assert(mVec[i] >= MINSNPS_B && mVec[i] <= (MINSNPS_B + MAXSNPS_E));
    assert(nVec[i] >= MINSNPS_B && nVec[i] <= (MINSNPS_B + MAXSNPS_E));
    assert(LVec[i]>=0.0f && LVec[i]<=1.0f*mVec[i]);
    assert(RVec[i]>=0.0f && RVec[i]<=1.0f*nVec[i]);
    assert(CVec[i]>=0.0f && CVec[i]<=1.0f*mVec[i]*nVec[i]);
  }
  wait = 1;
  pthread_cond_broadcast(&cond1);
  double timeOmegaTotalStart = gettime();
  for(unsigned int j=0;j<iters;j++){
    process_workload(&avgF,&minF,&maxF,num_threads,thread_handles,main_args);
    main_args = spawn_threads(thread_handles,num_threads,N,mVec,nVec,RVec,LVec,CVec,FVec);
  }
  process_workload(&avgF,&minF,&maxF,num_threads,thread_handles,main_args);
  double timeOmegaTotal = gettime() - timeOmegaTotalStart;
  double timeTotalMainStop = gettime();

  printf("Omega time %fs - Total time %fs - Min %e - Max %e - Avg %e\n",timeOmegaTotal / iters, timeTotalMainStop - timeTotalMainStart, (double) minF, (double) maxF,  (double) avgF / N);

  free(mVec);
  free(nVec);
  free(LVec);
  free(RVec);
  free(CVec);
  free(FVec);
}

void process_workload(float* avgF,float* minF,float* maxF,int num_threads,pthread_t *thread_handles,t_args* main_args){
  t_res* thread_res = (t_res*)omega_calc((void*)main_args);
  *avgF = ((t_res*)thread_res)->avgF;
  *minF = ((t_res*)thread_res)->minF;
  *maxF = ((t_res*)thread_res)->maxF;
  for(unsigned int t=0;t<num_threads;t++){
    pthread_join(thread_handles[t],(void*)&thread_res);
    if(thread_res->minF < *minF){*minF = thread_res->minF;}
    if(thread_res->maxF > *maxF){*maxF = thread_res->maxF;}
    *avgF += thread_res->avgF;
  }
}

t_args* spawn_threads(pthread_t *thread_handles,int num_threads,unsigned int N,float *mVec,float *nVec,float *RVec,float *LVec,float *CVec,float *FVec){
  int count = 0;
  t_args *main_args,*args;
  int nt;
  int c = (float)N/((float)num_threads+1);
  c = c -(c%4);
  for(unsigned int t=0; t<=num_threads;t++){
    args = (t_args*)malloc(sizeof(t_args));
    args->t_id = t;
    args->mVec = mVec;
    args->nVec = nVec;
    args->LVec = LVec;
    args->RVec = RVec;
    args->CVec = CVec;
    args->FVec = FVec;
    if(t == 0){
      args->from_index = 0;
      args->to_index = c;
      count += c;
      main_args = args;
    }
    else
    {
      if(t<num_threads){
        args->from_index = count;
        args->to_index = count+c;
        count+= c;
      }
      else{
        args->from_index = count;
        args->to_index = N;
      }
      pthread_create(&thread_handles[t-1],NULL,omega_calc,(void*)args);
    }
  }
  return main_args;
}

void *omega_calc(void * args){
  if(wait ==0){

    pthread_mutex_lock(&lock);
    pthread_cond_wait(&cond1, &lock);
    pthread_mutex_unlock(&lock);

  }
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
  t_res * result = (t_res*)malloc(sizeof(t_res));
  result->avgF = 0.0f;
  result->minF = 10000;
  result->maxF = 0.0f;
  const __m128 one= _mm_set1_ps(1.0f);
  const __m128 two= _mm_set1_ps(2.0f);
  const __m128 zerozeroone = _mm_set1_ps(0.01f);
  int x = ((t_args*)args)->from_index;
  int t = (((t_args*)args)->to_index - ((((t_args*)args)->to_index-((t_args*)args)->from_index)%4));
  for(unsigned int i=x;i<t;i+=4){
    rVec_t = _mm_load_ps(&(((t_args*)args)->RVec[i]));
    lVec_t = _mm_load_ps(&(((t_args*)args)->LVec[i]));
    mVec_t = _mm_load_ps(&(((t_args*)args)->mVec[i]));
    nVec_t = _mm_load_ps(&(((t_args*)args)->nVec[i]));

    num = _mm_add_ps(_mm_div_ps(_mm_mul_ps(mVec_t,_mm_sub_ps(mVec_t,one)),two),_mm_div_ps(_mm_mul_ps(nVec_t,_mm_sub_ps(nVec_t,one)),two));
    num = _mm_div_ps(_mm_add_ps(lVec_t,rVec_t),num);
    den = _mm_div_ps( _mm_sub_ps(_mm_sub_ps(_mm_load_ps(&(((t_args*)args)->CVec[i])),lVec_t),rVec_t),_mm_mul_ps(mVec_t,nVec_t));

    F_result = _mm_div_ps(num,_mm_add_ps(den,zerozeroone));
    _mm_store_ps(&(((t_args*)args)->FVec[i]),F_result);
    maxF_t = _mm_load_ps1(&(result->maxF));
    minF_t = _mm_load_ps1(&(result->minF));

    cond_1 = _mm_cmpgt_ps(F_result, maxF_t);
    v1 = _mm_sub_ps(F_result,maxF_t);
    v1 = _mm_and_ps(v1, cond_1);
    maxF_t = _mm_add_ps(maxF_t,v1);
    maxF_t = _mm_max_ps(maxF_t, _mm_shuffle_ps(maxF_t, maxF_t, _MM_SHUFFLE(2, 1, 0, 3)));
    maxF_t = _mm_max_ps(maxF_t, _mm_shuffle_ps(maxF_t, maxF_t, _MM_SHUFFLE(1, 0, 3, 2)));
    _mm_store_ss(&(result->maxF),maxF_t);
    cond_2 = _mm_cmplt_ps(F_result, minF_t);
    v2 = _mm_sub_ps(minF_t,F_result);
    v2 = _mm_and_ps(v2, cond_2);
    minF_t = _mm_sub_ps(minF_t,v2);
    minF_t = _mm_min_ps(minF_t, _mm_shuffle_ps(minF_t, minF_t, _MM_SHUFFLE(2, 1, 0, 3)));
    minF_t = _mm_min_ps(minF_t, _mm_shuffle_ps(minF_t, minF_t, _MM_SHUFFLE(1, 0, 3, 2)));
    _mm_store_ss(&(result->minF),minF_t);
    F_result = _mm_hadd_ps(F_result,F_result);
    F_result = _mm_hadd_ps(F_result,F_result);
    _mm_store_ss(&(result->avgF), _mm_add_ps(_mm_load_ps1(&(result->avgF)),F_result));
  }

  for(unsigned int i = t;i<((t_args*)args)->to_index;i++){
    rVec_t = _mm_load_ss(&(((t_args*)args)->RVec[i]));
    lVec_t = _mm_load_ss(&(((t_args*)args)->LVec[i]));
    mVec_t = _mm_load_ss(&(((t_args*)args)->mVec[i]));
    nVec_t = _mm_load_ss(&(((t_args*)args)->nVec[i]));
    num = _mm_add_ss(_mm_div_ss(_mm_mul_ss(mVec_t,_mm_sub_ss(mVec_t,one)),two),_mm_div_ss(_mm_mul_ss(nVec_t,_mm_sub_ss(nVec_t,one)),two));
    num = _mm_div_ss(_mm_add_ss(lVec_t,rVec_t),num);
    den = _mm_div_ss( _mm_sub_ss(_mm_sub_ss(_mm_load_ss(&(((t_args*)args)->CVec[i])),lVec_t),rVec_t),_mm_mul_ss(mVec_t,nVec_t));
    F_result = _mm_div_ss(num,_mm_add_ss(den,zerozeroone));
    _mm_store_ss(&(((t_args*)args)->FVec[i]),F_result);
    maxF_t = _mm_load_ss(&(result->maxF));
    minF_t = _mm_load_ss(&(result->minF));
    cond_1 = _mm_cmpgt_ss(F_result, maxF_t);
    v1 = _mm_sub_ps(F_result,maxF_t);
    v1 = _mm_and_ps(v1, cond_1);
    maxF_t = _mm_add_ss(maxF_t,v1);
    _mm_store_ss(&(result->maxF),maxF_t);
    cond_2 = _mm_cmplt_ss(F_result, minF_t);
    v2 = _mm_sub_ss(minF_t,F_result);
    v2 = _mm_and_ps(v2, cond_2);
    minF_t = _mm_sub_ss(minF_t,v2);
    _mm_store_ss(&(result->minF),minF_t);
    _mm_store_ss(&(result->avgF), _mm_add_ss(_mm_load_ss(&(result->avgF)),F_result));
  }
  if(((t_args*)args)->t_id !=0){
    free(args);
    pthread_exit(result);
  }
  else{
    free(args);
    return (void*)result;
  }
}
