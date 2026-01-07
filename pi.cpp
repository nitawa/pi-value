#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <mpi.h>
#include <chrono>
#include <unistd.h>
#include <climits>

int main(int argc, char** argv) {

  // start time measurement
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

  // Initialize MPI
  MPI_Init(&argc, &argv);

  int world_size, world_rank;
  // Get number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  // Get rank of the processes
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // check whether the number of processes is valid - ideally the number of processes should be even or 1
  if (world_size > 1 && world_size % 2 != 0) {
    if (world_rank == 0) {
      std::cerr << "Error: Number of processes must be 1 or an even number." << std::endl;
    }
    MPI_Finalize();
    return 1;
  } // end check for valid number of processes

  unsigned int NEvts        = UINT_MAX; // Total points to generate
  unsigned int NEvtsPerProc = NEvts / world_size;
  unsigned int hitsProc = 0;

  // one seed per process to ensure different sequences
  std::mt19937 gen(12345 + world_rank);
  // random float distribution between 0 and 1
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  for (unsigned int i = 0; i < NEvtsPerProc; i++) {
    double x = dist(gen);
    double y = dist(gen);
    double r = std::sqrt(x * x + y * y); // radius
    if (r <= 1.0) {
      hitsProc++;
    } // end if inside circle
  } // end for all events per process

  unsigned int hits = 0;
    if (world_size > 1) {
        // Parallel mode: Reduce all local_hits to Rank 0
        MPI_Reduce(&hitsProc, &hits, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        // Sequential mode
        hits = hitsProc;
    }

    if (world_rank == 0) {
        // Pi = 4 * (hits / total_samples)
        double piValue = 4.0 * (double)hits / (double)(NEvtsPerProc * world_size);
        // Output the result with high precision
        std::cout << "pi value: " << piValue << " - using " << world_size << " procs  and " << NEvts << " events" << std::endl;
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = t1 - t0;
        std::cout << "Elapsed time: " << elapsed_seconds.count() << " s" << std::endl;
    }

    MPI_Finalize();
    return 0;
}