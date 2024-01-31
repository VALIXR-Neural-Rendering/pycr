
#pragma once

#define POINTS_PER_THREAD 80
#define WORKGROUP_SIZE 128
#define POINTS_PER_WORKGROUP (POINTS_PER_THREAD * WORKGROUP_SIZE)
// Adjust this to be something in the order of 1 million points
#define MAX_POINTS_PER_BATCH (100 * POINTS_PER_WORKGROUP)