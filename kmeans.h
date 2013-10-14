typedef struct {
  float* make_believes;
  double* cluster_aggregate;
  double cluster_RSS;
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

typedef double (*R)(Point*, Centroid*);

Delta StoreData(FILE* fp, int numfeatures);

void KMeans(Delta data, R distance, Centroid** centroids, int k, int numfeatures);

double TestDistance(R distance, Point* point, Centroid* centroid);

void PrintCentroid(Centroid* centroid, FILE* fp, int numfeatures);

double* InitAggregates(int numfeatures);

float* GetFeatures(int numfeatures, char* string);
