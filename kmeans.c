#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "kmeans.h"

#define MAX_ITER 15
#define MIN_RSS_ITER (2 * MAX_ITER) / 3

static void AddData(Delta* head_ref, Point* data) {
  Delta newNode = (Delta) malloc(sizeof(Delta));
  newNode->datapoint = data;
  newNode->next = *head_ref;
  *head_ref = newNode;
}

static void PrintPoint(Point* datapoint, int numfeatures) {
  int i;  
  printf("%ld\n", datapoint->ID);
  for (i = 0; i < numfeatures; i++) {
    printf("%f ", datapoint->features[i]);
  }
  printf("\n");
}

void PrintCentroid(Centroid* centroid, FILE* fp, int numfeatures) {
  int i;
  for (i = 0; i < numfeatures-1; i++) {
    fprintf(fp, "%f,", centroid->make_believes[i]);
  }
  fprintf(fp, "%f", centroid->make_believes[i]);
  fprintf(fp, "\n");
}

static float GetLabel(Point* datapoint, int numfeatures) {
  return datapoint->features[numfeatures - 1];
}

static void PrintCluster(Delta datanode, float label, int numfeatures) {
  printf("======= Cluster with label: %f =======\n", label);
  while (datanode != NULL) {
    Point* point = datanode->datapoint;
    float ilabel = GetLabel(point, numfeatures);
    if (ilabel == label) {
      printf("%ld\t", point->ID);
    }
    datanode = datanode->next;
  }
  printf("\n");
}

float* GetFeatures(int numfeatures, char* string) {
  float* tokens = (float*) malloc(sizeof(float) * numfeatures);
  char* delim = ",";
  char* token;
  int i = 0;
  
  token = strsep(&string, delim);
  while (token != NULL && i<numfeatures) {
    tokens[i] = strtof(token, NULL);
    i++;
    token = strsep(&string, delim);
  }
  
  return tokens;
}

static Point* CreatePoint(int numfeatures, char* string) {
  Point* datapoint = (Point*) malloc(sizeof(Point));
  
  char* id = strsep(&string, ",");
  datapoint->ID = strtol(id, NULL, 0);
  datapoint->features = GetFeatures(numfeatures, string);
  datapoint->centroid_ref = NULL;
  return datapoint;
}

Delta StoreData(FILE* fp, int numfeatures) {
  Delta head = NULL;  
  char buf[100];
  
  while (!feof(fp)) {
    fgets(buf, sizeof(buf), fp);
    size_t nl = strlen(buf) - 1;
    if (buf[nl] == '\n') {
      buf[nl] = '\0';
    }
    AddData(&head, CreatePoint(numfeatures, buf));
  }
  return head;
}

static Centroid* CreateCentroid() {
  return (Centroid*) malloc(sizeof(Centroid));  
}

static Centroid** StoreCentroids() {
  Centroid** C = (Centroid**) malloc(sizeof(Centroid*) * 2);
  C[0] = CreateCentroid();
  C[1] = CreateCentroid();
  return C;
}

static void SortPoint(R distance, Point* datapoint, Centroid** troids, int k, int numfeatures) {
  int i;
  double min_dist = (*distance)(datapoint, troids[0]);
  Centroid* closest = troids[0];
  for (i=1; i<k; i++) {
    double cand_dist = (*distance)(datapoint, troids[i]);
    if (cand_dist < 0) {
      continue;
    }
    if (cand_dist < min_dist ||
  	(cand_dist == min_dist && troids[i]->cluster_count < closest->cluster_count)) {
      min_dist = cand_dist;
      closest = troids[i];
    }
  }
  datapoint->centroid_ref = closest;
  for (i=0; i<numfeatures; i++) {
    closest->cluster_aggregate[i] += datapoint->features[i];
  }
  closest->cluster_RSS += pow(min_dist, 2);
  closest->cluster_count++;
}

static void PartitionData(Delta datanode, R distance, Centroid** centroids, int k, int numfeatures) {
  while (datanode != NULL) {
    SortPoint(distance, datanode->datapoint, centroids, k, numfeatures);
    datanode = datanode->next;
  }
}

static double RSS(Centroid** centroids, int k) {
  int i;
  double RSS = 0;
  for (i=0; i<k; i++) {
    RSS += centroids[i]->cluster_RSS;
  }
  return RSS;
}

static float* NewMakeBelieve(Centroid* centroid, int numfeatures) {
  int i;
  float* new_mb = (float*) malloc(sizeof(float) * numfeatures);
  long count = centroid->cluster_count;
  double* aggr = centroid->cluster_aggregate;
  for (i=0; i<numfeatures; i++) {
    new_mb[i] = (float) (aggr[i] / count);
  }
  return new_mb;
}

static int IsCentroidDifferent(float* prev, float* new, int numfeatures) {
  int i;
  for (i=0; i<numfeatures; i++) {
    if (prev[i] != new[i]) {
      return 1;
    }
  }
  return 0;
}

double* InitAggregates(int numfeatures) {
  int i;
  double* aggr = (double*) malloc(sizeof(double)*numfeatures);
  for (i=0; i<numfeatures; i++) {
    aggr[i] = 0;
  }
  return aggr;
}

static void UpdateCentroids(Centroid** centroids, float** new_mbs, int k, int numfeatures) {
  int i;
  for (i=0; i<k; i++) {
    centroids[i]->make_believes = new_mbs[i];
    centroids[i]->cluster_aggregate = InitAggregates(numfeatures);
    centroids[i]->cluster_RSS = 0;
    centroids[i]->cluster_count = 0;
  }
}

static int UpdateAndContinue(int iter, Centroid** centroids, int k, int numfeatures, double RSS_min) {
  int i, changed_centroids = 0;
  float** new_mbs = (float**) malloc(sizeof(float*) * k);
  
  for (i=0; i<k; i++) {
    float* prev = centroids[i]->make_believes;
    new_mbs[i] = NewMakeBelieve(centroids[i], numfeatures);
    changed_centroids += IsCentroidDifferent(prev, new_mbs[i], numfeatures);
  }
  
  if (!changed_centroids ||
      (RSS_min == RSS(centroids, k) && iter >= MIN_RSS_ITER) ||
      iter >= MAX_ITER) {
    return 0;
  }
  
  UpdateCentroids(centroids, new_mbs, k, numfeatures);
  return 1;
}

void KMeans(Delta data, R distance, Centroid** centroids, int k, int numfeatures) {
  int i = 0;
  double RSS_min, new_RSS;
  do {
    printf("Entering iteration %d..\n", i);
    PartitionData(data, distance, centroids, k, numfeatures);
    new_RSS = RSS(centroids, k);
    if (RSS_min) {
      RSS_min = new_RSS <= RSS_min ? new_RSS : RSS_min;
    } else {
      RSS_min = new_RSS;
    }
    i++;
  } while (UpdateAndContinue(i, centroids, k, numfeatures, RSS_min));
}

double TestDistance(R distance, Point* point, Centroid* centroid) {
  return (*distance)(point, centroid);
}
