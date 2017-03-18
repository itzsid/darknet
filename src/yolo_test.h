#ifndef YOLO_TEST_H
#define YOLO_TEST_H
#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"


/** @brief Test YOLO v1 network */
void test_yolo_v1(network* net, char *filename, char** names, image** alphabet, char* output_file, float thresh, int visualize);

/** @brief Test YOLO v2 network */
void test_yolo_v2(network* net, char *filename, char** names, image** alphabet, char* output_file, float thresh, int visualize);

#endif
