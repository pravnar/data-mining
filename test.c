#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUMCOLUMN 11

typedef struct { 
  float* make_believes;
  double cluster_aggregate;
  long cluster_count;
} Centroid;

typedef struct {
  long ID;
  float* features;
  Centroid* centroid_ref;
} Point;

typedef struct deltanode* Delta;
struct deltanode {
  Point* datapoint;
  Delta next;
};

typedef float (*R)(Point*, Centroid*);

static void AddData(Delta* head_ref, Point* data) {
  Delta newNode = (Delta) malloc(sizeof(Delta));
  newNode->datapoint = data;
  newNode->next = *head_ref;
  *head_ref = newNode;
}

static void PrintPoint(Point* datapoint) {
  int i;  
  printf("%ld\n", datapoint->ID);
  for (i = 0; i < NUMCOLUMN-1; i++) {
    printf("%f ", datapoint->features[i]);
  }
  printf("\n");
}

static float GetLabel(Point* datapoint) {
  return datapoint->features[NUMCOLUMN - 2];
}

static void PartitionData(Centroid** centroids, Delta datanode) {
  while (datanode != NULL) {
    Point* point = datanode->datapoint;
    float ilabel = GetLabel(point);
    if (ilabel == 2) {
      point->centroid_ref = centroids[0];
    } else if (ilabel == 4) {
      point->centroid_ref = centroids[1];
    } else {
      printf ("Wat: %f\n", ilabel);
    }
    datanode = datanode->next;
  }
}

static void PrintCluster(Delta datanode, float label) {
  printf("======= Cluster with label: %f =======\n", label);
  while (datanode != NULL) {
    Point* point = datanode->datapoint;
    float ilabel = GetLabel(point);
    if (ilabel == label) {
      printf("%ld\t", point->ID);
    }
    datanode = datanode->next;
  }
  printf("\n");
}

static float* GetFeatures(int numfeatures, char* string) {
  float* tokens = (float*) malloc(sizeof(float) * numfeatures);
  char* delim = ",";
  char* token;
  int i = 0;
  
  token = strsep(&string, delim);
  while (token != NULL) {
    tokens[i] = strtof(token, NULL);
    i++;
    token = strsep(&string, delim);
  }
  
  return tokens;
}

static Point* CreatePoint(int numcolumns, char* string) {
  Point* datapoint = (Point*) malloc(sizeof(Point));
  
  char* id = strsep(&string, ",");
  datapoint->ID = strtol(id, NULL, 0);
  /* Let's say the ID is not a feature */
  datapoint->features = GetFeatures(numcolumns-1, string);
  datapoint->centroid_ref = NULL;
  return datapoint;
}

static Delta StoreData(FILE* fp) {  
  Delta head = NULL;  
  char buf[100];
  
  while (!feof(fp)) {
    fgets(buf, sizeof(buf), fp);
    size_t nl = strlen(buf) - 1;
    if (buf[nl] == '\n') {
      buf[nl] = '\0';
    }
    AddData(&head, CreatePoint(NUMCOLUMN, buf));
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

static void SortPoint(R distance, Point* datapoint, Centroid** troids, int k) {
  int i;
  float min_dist = (*distance)(datapoint, troids[0]);
  Centroid* closest = troids[0];
  for (i=1; i<k; i++) {
    float cand_dist = (*distance)(datapoint, troids[i]);
    if (cand_dist < min_dist ||
	(cand_dist == min_dist && troids[i]->cluster_count < closest->cluster_count)) {
      min_dist = cand_dist;
      closest = troids[i];
    }
  }
  datapoint->centroid_ref = closest;
  closest->cluster_aggregate += min_dist;
  closest->cluster_count++;
}

int main() {
  FILE* fp = fopen("breast-cancer-wisconsin.data", "r");
  
  Delta datanode = StoreData(fp);
  Centroid** C = StoreCentroids();
  PartitionData(C, datanode);
  PrintCluster(datanode, 2);
  
  return 0;
}
