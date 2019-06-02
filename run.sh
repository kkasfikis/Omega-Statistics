clear
echo "The run script for the project"
sleep 1
echo "Executing the refference code for N=10000000"
gcc original.c -o orig
./orig 10000000
echo "finished.."
sleep 1
echo "Executing the SSE inplementation for N=10000000"
gcc SSE.c -o sse -msse4.2 -O3 
./sse 10000000
echo "finished.."
sleep 1
echo "Executing the SSE and pthreads inplementation for num_of_threrads = 2"
gcc SSEwPTHREADS.c -o sseP -msse4.2 -O3 -pthread 
./sseP 10000000 2
echo "finished.."
sleep 1
echo "Executing the SSE and pthreads inplementation for num_of_threrads = 4"
gcc SSEwPTHREADS.c -o sseP -msse4.2 -O3 -pthread 
./sseP 10000000 4
echo "finished.."
sleep 1
echo "Executing the SSE and pthreads and MPI inplementation for num_of_threrads = 2 and num_of_procs = 2"
mpicc SSEwPTHREADSwMPI.c -o sseWW -pthread -msse4.2 -O3
lamboot
mpiexec -n 2 ./sseWW 10000000 2
echo "finished.."
sleep 1
echo "Executing the SSE and pthreads and MPI inplementation for num_of_threrads = 2 and num_of_procs = 4"
mpicc SSEwPTHREADSwMPI.c -o sseWW -pthread -msse4.2 -O3
lamboot
mpiexec -n 4 ./sseWW 10000000 2
echo "finished.."
sleep 1
echo "Executing the SSE and pthreads and MPI inplementation for num_of_threrads = 4 and num_of_procs = 2"
mpicc SSEwPTHREADSwMPI.c -o sseWW -pthread -msse4.2 -O3
lamboot
mpiexec -n 2 ./sseWW 10000000 4
echo "finished.."
sleep 1
echo "Executing the SSE and pthreads and MPI inplementation for num_of_threrads = 4 and num_of_procs = 4"
mpicc SSEwPTHREADSwMPI.c -o sseWW -pthread -msse4.2 -O3
lamboot
mpiexec -n 4 ./sseWW 10000000 4
echo "finished.."
sleep 1
